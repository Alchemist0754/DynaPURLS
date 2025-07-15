import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from utils import TBLogger
import time
from collections import defaultdict
import copy


def load_callbacks(args):
    """Load PyTorch Lightning callbacks"""
    callbacks = []
    # Checkpoint callback
    callbacks.append(plc.ModelCheckpoint(
        dirpath=args.work_dir,
        monitor='val-loss',  # Use validation set's loss
        filename='best',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs=args.save_interval,
    ))
    return callbacks


def get_trainer(args):
    """Get PyTorch Lightning trainer"""
    logger = TBLogger(save_dir=args.tb_folder, 
                      name=args.work_dir_name, 
                      default_hp_metric=False)
                
    args.callbacks = load_callbacks(args)
    args.logger = logger
        
    return Trainer(
        devices=args.gpus if args.gpus > 0 else None,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        strategy='ddp_find_unused_parameters_true', 
        callbacks=args.callbacks,
        logger=logger,
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_interval,
        num_sanity_val_steps=0,
    )


def find_sample_index(buffer, sample_to_find):
    """Find sample index in buffer"""
    for idx, sample in enumerate(buffer):
        if (sample['label'] == sample_to_find['label'] and
            sample['confidence'] == sample_to_find['confidence']):
            return idx
    return None


def generate_text_features(model):
    """Generate text features"""
    # Assume model.model.text_features are already set
    text_features = model.model.text_features.clone().detach().to(model.device)
    # Now, text_features has shape (num_classes, num_parts+1, feat_dim)
    return text_features


def update_buffer_with_sample(buffer, label, sample_loss, confidence, buffer_size):
    """Update buffer with new sample"""
    sample = {
        'loss': sample_loss,  # Loss with grad_fn
        'confidence': confidence
    }

    if len(buffer[label]) < buffer_size:
        buffer[label].append(sample)
    else:
        # Replace the lowest confidence sample if the new one has higher confidence
        min_conf_index = min(
            range(len(buffer[label])),
            key=lambda idx: buffer[label][idx]['confidence']
        )
        if confidence > buffer[label][min_conf_index]['confidence']:
            buffer[label][min_conf_index] = sample


def collect_buffer_losses(buffer):
    """Collect losses from buffer for optimization"""
    buffer_losses = []
    for samples in buffer.values():
        buffer_losses.extend([item['loss'] for item in samples])
    return buffer_losses


def compute_logits_and_predictions(skel_features, text_features):
    """Compute logits and predictions"""
    logits = torch.sum(skel_features * text_features, axis=-1)
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    return logits, probs, max_probs, preds


def apply_scale_and_shift(scale_parameters, shift_parameters, text_features):
    """Apply scale and shift parameters"""
    scaled_features = scale_parameters * text_features
    shifted_text_features = F.normalize(scaled_features + shift_parameters, dim=-1)
    return shifted_text_features


def initialize_optimization_parameters(text_features):
    """Initialize optimization parameters"""
    scale_parameters = nn.Parameter(torch.ones_like(text_features)).cuda()
    shift_parameters = nn.Parameter(torch.zeros_like(text_features)).cuda()
    return scale_parameters, shift_parameters


def setup_optimizer_and_scheduler(scale_parameters, shift_parameters, lr, steps):
    """Setup optimizer and scheduler"""
    optimizer = torch.optim.Adam([scale_parameters, shift_parameters], 
                                lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=steps, 
                                                        eta_min=lr * 0.1)
    return optimizer, scheduler


def process_batch_adaptation(model, vis_emb_batch, target_batch, target2_batch, 
                           unseen_text_features, unseen_inds, scale_parameters, 
                           shift_parameters, optimizer, scheduler, buffer, 
                           confidence_threshold, buffer_size, steps):
    """Process batch adaptation procedure"""
    for step in range(steps):
        # Apply scale and shift
        shifted_text_features = apply_scale_and_shift(scale_parameters, shift_parameters, unseen_text_features)

        # Compute pseudo-labels and confidence (no gradient)
        with torch.no_grad():
            cls_idx = unseen_inds
            losses, vals = model.model.loss(
                vis_emb_batch,
                target=target_batch,
                target2=target2_batch,
                cls_idx=cls_idx,
                curr_epoch=0,
                is_train=False,
                is_test=True,
                out_text_features=shifted_text_features
            )

            skel_features = vals['skel_emb']
            text_features = vals['text_emb']
            logits, probs, max_probs, preds = compute_logits_and_predictions(skel_features, text_features)

        # Identify high-confidence samples
        confident_mask = max_probs > confidence_threshold
        if confident_mask.sum() == 0:
            continue  # Skip if no high-confidence samples

        # Prepare high-confidence samples
        vis_emb_confident = vis_emb_batch[confident_mask]
        preds_confident = preds[confident_mask]
        max_probs_confident = max_probs[confident_mask]

        # Compute loss for high-confidence samples (requires gradient)
        model.model.train()
        losses_sample, vals_sample = None, None
        if vis_emb_confident.size(0) >= 2:  # Ensure at least 2 samples
            losses_sample, vals_sample = model.model.loss(
                vis_emb_confident,
                target=preds_confident,
                target2=None,
                cls_idx=cls_idx,
                curr_epoch=0,
                is_train=False,
                is_test=True,
                out_text_features=shifted_text_features
            )
        else:
            continue  # Skip iteration if insufficient samples
        
        skel_features_sample = vals_sample['skel_emb']
        text_features_sample = vals_sample['text_emb']
        
        sample_logits = torch.sum(skel_features_sample * text_features_sample, axis=-1)
        sample_losses = F.cross_entropy(sample_logits, preds_confident, reduction='none')

        # Update buffer
        batch_size_confident = vis_emb_confident.size(0)
        for i in range(batch_size_confident):
            label = preds_confident[i].item()
            sample_loss = sample_losses[i]
            confidence = max_probs_confident[i].item()
            update_buffer_with_sample(buffer, label, sample_loss, confidence, buffer_size)

        # Check if buffer has enough samples for optimization
        total_samples_in_buffer = sum(len(samples) for samples in buffer.values())
        min_batch_size = 2  # Adjustable as needed
        if total_samples_in_buffer < min_batch_size:
            continue

        # Collect losses from buffer for optimization
        buffer_losses = collect_buffer_losses(buffer)

        # Compute mean loss from buffer
        if buffer_losses:
            mean_loss = torch.mean(torch.stack(buffer_losses))

            optimizer.zero_grad()
            mean_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()


def evaluate_batch(model, vis_emb_batch, target_batch, target2_batch, 
                  unseen_text_features, unseen_inds, scale_parameters, shift_parameters):
    """Evaluate batch performance"""
    with torch.no_grad():
        shifted_text_features = apply_scale_and_shift(scale_parameters, shift_parameters, unseen_text_features)
        cls_idx = unseen_inds

        valid_mask = (target_batch != -1)
        if valid_mask.sum() > 0:
            losses, vals = model.model.loss(
                vis_emb_batch[valid_mask],
                target=target_batch[valid_mask],
                target2=target2_batch[valid_mask],
                cls_idx=cls_idx,
                curr_epoch=0,
                is_train=False,
                is_test=True,
                out_text_features=shifted_text_features
            )
            skel_features = vals['skel_emb']
            text_features = vals['text_emb']
            logits, probs, max_probs, preds = compute_logits_and_predictions(skel_features, text_features)
            correct = (preds == target_batch[valid_mask]).sum().item()
            
            return correct, valid_mask.sum().item(), target2_batch[valid_mask].cpu().numpy(), [unseen_inds[p.item()] for p in preds.cpu()]
        
        return 0, 0, [], []


def run_dynamic_refinement(model, test_loader, hparams, steps=20, lr=0.001, 
                         buffer_size=256, confidence_threshold=0.5):
    """Run dynamic refinement process"""
    model = model.cuda()
    unseen_inds = hparams.unseen_inds
    text_features = generate_text_features(model)
    unseen_text_features = text_features[unseen_inds].clone().detach().cuda()

    # Initialize optimization parameters
    scale_parameters, shift_parameters = initialize_optimization_parameters(unseen_text_features)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(scale_parameters, shift_parameters, lr, steps)

    total_correct = 0
    total_samples = 0

    results = {
        'true_labels': [], 
        'predicted_labels': [],
        'confusion_matrix': None
    }
    buffer = defaultdict(list)  # Class-balanced buffer

    start_time = time.time()

    for batch in test_loader:
        vis_emb_batch, target_batch, target2_batch = batch
        vis_emb_batch = vis_emb_batch.cuda()
        target_batch = target_batch.cuda()
        target2_batch = target2_batch.cuda()

        # Process batch adaptation
        process_batch_adaptation(model, vis_emb_batch, target_batch, target2_batch, 
                               unseen_text_features, unseen_inds, scale_parameters, 
                               shift_parameters, optimizer, scheduler, buffer, 
                               confidence_threshold, buffer_size, steps)

        # Evaluate on current batch after adaptation
        correct, valid_samples, true_labels, pred_labels = evaluate_batch(
            model, vis_emb_batch, target_batch, target2_batch, 
            unseen_text_features, unseen_inds, scale_parameters, shift_parameters)
        
        total_correct += correct
        total_samples += valid_samples
        results['true_labels'].extend(true_labels)
        results['predicted_labels'].extend(pred_labels)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate overall accuracy
    test_acc = total_correct / total_samples
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/total_samples} seconds")

    return test_acc, total_time, total_time / total_samples, results


def run_evaluation(model, trainer, test_loader):
    """Run standard evaluation"""
    model.zsl()
    results = trainer.test(model, test_loader)
    return results[0]['test-acc'] 
