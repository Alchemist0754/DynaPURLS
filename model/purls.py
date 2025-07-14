import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from model import Base, init_weights
import numpy as np
from model.utils.losses import contrastive_loss, accuracy
from model.utils.attention import MultiHeadSelfAttention
from model.utils.layers import HiddenLayer
from model.utils.contrastive_learning import get_rank

local_names = ['head', 'hands', 'torso', 'legs', 'start', 'middle', 'end', 'global']

class Purls(Base):
    def __init__(self, num_epoch, input_size=256, 
                 emb_dim = 512, bp_num=4, t_num=3, 
                 proj_hidden_size=1024, n_hidden_layers=1, 
                 n_head=2, k_dim=150, local_type = '', cls_labels=None,
                 use_d = False):
        """
        Argument explanations:
        num_epoch: total number of running epochs
        input_size: input skeleton feature size
        emb_dim: text embedding size for descriptions
        bp_num: number of body-part-based representations
        t_num: number of temporal-interval-based representations
        proj_hidden_size: hidden layer output size for skel->txt projection
        n_hidden_layers: number of hidden layers for skel->txt projection
        n_head: number of head for attention module
        k_dim: transformed feature size for attention module
        local_type: types of local/global representation matching ('bp', 'tp', 'g')
        cls_labels: (extended) label names of each action (or its part-based/temporal-based local descriptions)
        use_d: activate weighted learning for each contrastive loss
        """
        
        init_dict = locals().copy()
        init_dict.pop('self')
        super().__init__(**init_dict)
        """
        configs
        """
        self.loss_func = contrastive_loss
        """
        model
        """
        # feature skel extractors for glob & local
        if self.local_type == 'bp': # only learn body-part-based local representations
            encode_input_size = self.input_size * (self.bp_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.bp_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.bp_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.bp_num + 1,device='cuda'), requires_grad=True)
            
        elif self.local_type == 'tp': # only learn temporal-based local representations
            encode_input_size = self.input_size * (self.t_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.t_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.t_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.t_num + 1,device='cuda'), requires_grad=True)
            
        elif self.local_type == 'g': # only learn global-description-based representations
            encode_input_size = self.input_size # 2048
            encode_output_size = self.emb_dim # 4096
            self.tmps = nn.Parameter(torch.ones([1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(1,device='cuda'), requires_grad=True)
            
        else: # learn global-description, body-part-based, temporal-based local representations
            encode_input_size = self.input_size * (self.bp_num + self.t_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.bp_num + self.t_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.bp_num + self.t_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.bp_num + self.t_num + 1,device='cuda'), requires_grad=True)
        
        
        # skeleton representation projection
        self.attention = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        self.skel_encoder = nn.Sequential(
                                        nn.BatchNorm1d(encode_input_size), 
                                        nn.Dropout(.5),
                                        nn.Linear(encode_input_size, self.proj_hidden_size),
                                        nn.SiLU(),
                                        HiddenLayer(self.n_hidden_layers, self.proj_hidden_size),
                                        nn.BatchNorm1d(self.proj_hidden_size),
                                        nn.Dropout(.5),
                                        nn.Linear(self.proj_hidden_size, encode_output_size),
                                        nn.Tanh()
                                    )
        # initialization
        self.attention.apply(init_weights)
        self.skel_encoder.apply(init_weights)
        
        # Remove CLIP-related code
        self.text_features = None

    def set_text_features(self, text_features):
        self.text_features = text_features

    def configure_optimizers(self, monitor1, monitor2, lr=1e-3):
        params = []
        params += self.skel_encoder.parameters()
        params += [self.tmps]
        if self.use_d:
            params += [self.d]
        
        params += self.attention.parameters()
        optimizer = optim.Adam(params, lr=lr)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, 
                                        cooldown=3, verbose=True)
        scheduler = {'scheduler': scheduler,
                    'monitor': monitor1,
                    'mode': 'min'
                    }
        return [optimizer], [scheduler]


    def loss(self, vis_emb, target, target2, 
             cls_idx, curr_epoch, is_train=False, 
             is_test=False, out_text_features=None, **params):
        """
        Argument explanations:
        vis_emb: Pre-trained skeleton features of the batch examples from backbones
        target: Label index for the batch examples (re-labeled according to available candidate classes)
        target2: Original label index for the batch examples (i.e. not yet re-labeled)
        cls_idx: All class index for the current task (e.g. For a NTU 55/5 split, cls_idx = [10, 11, 19, 26, 56])
        curr_epoch: Current epoch no.
        is_train: Whether the function is used during the training stage 
        is_test: Whether the function is used during the testing stage. 
            is_train = True -> Training Stage (on seen classes)
            is_train = False, is_test = False -> Validation Stage (on seen classes)
            is_train = False, is_test = True -> Testing Stage (on unseen classes)
        """
        # prepare one-hot contrastive learning labels
        batch_labels = len(vis_emb) * get_rank() + torch.arange(
                                                    len(vis_emb), device= vis_emb.device
                                                )
        # Use precomputed text features
        if out_text_features is None:
            text_features = self.text_features.to(vis_emb.device)
            with torch.no_grad():
                class_text_features = text_features[cls_idx]
                valid_target = torch.where(
                    target >= len(class_text_features),
                    torch.randint(0, len(class_text_features), target.shape, device=target.device),
                    target
                )
                bp_emb = class_text_features[valid_target]
        else:    
            class_text_features = out_text_features
            # Ensure target doesn't exceed bounds
            valid_target = torch.where(
                target >= len(class_text_features),
                torch.randint(0, len(class_text_features), target.shape, device=target.device),
                target
            )
            bp_emb = class_text_features[valid_target]
        # get body part adaptive features
        bs, t, j, d = vis_emb.shape
        vis_emb = vis_emb.view(bs, -1, d) # bs, tp, d
        vis_emb = vis_emb.to(bp_emb.device).to(bp_emb.dtype)
        
        local_vis_emb = self.attention(vis_emb, bp_emb)

        local_bp_emb = bp_emb
        
        # Handle feature concatenation based on local_type
        if self.local_type == 'bp':
            # Use only body-part-based features, first bp_num+1 features
            total_vis_emb = local_vis_emb[:, :self.bp_num+1, :]  # bs, bp_num+1, d
            local_bp_emb = local_bp_emb[:, :self.bp_num+1, :]    # bs, bp_num+1, d
        elif self.local_type == 'tp':
            # Use only temporal-based features, first t_num+1 features
            total_vis_emb = local_vis_emb[:, :self.t_num+1, :]   # bs, t_num+1, d
            local_bp_emb = local_bp_emb[:, :self.t_num+1, :]     # bs, t_num+1, d
        elif self.local_type == 'g':
            # Use only global features, take last feature
            total_vis_emb = local_vis_emb[:, -1:, :]  # bs, 1, d
            local_bp_emb = local_bp_emb[:, -1:, :]    # bs, 1, d
        else:
            # Use all local representations (bp + tp + global)
            total_vis_emb = local_vis_emb
        
        # calculate loss
        loss = 0.
        bs, p, d = total_vis_emb.shape
        total_vis_emb = total_vis_emb.view(bs, -1).to(total_vis_emb.device)

        output_emb = self.skel_encoder(total_vis_emb)
        output_emb = output_emb.view(bs, p, -1).to(total_vis_emb.device) # bs, p, 512

        # each representation contrastive learning loss (include global)
        for i in range(output_emb.shape[1]):
            curr_output_emb = output_emb[:, i, :]
            curr_att_emb = local_bp_emb[:, i, :]
            curr_loss, _ =  self.loss_func(curr_output_emb, curr_att_emb, None, 
                                                    self.tmps[i], batch_labels) #, reduction='none')
            if self.use_d:
                loss += curr_loss * self.d[i]
            else:
                loss += curr_loss

        global_emb = output_emb[:, -1, :]
        acc, preds, final_scores = accuracy(global_emb, target, class_text_features[:, -1, :], is_test, False)
        class_emb = class_text_features[:, -1, :]
        skel_emb = global_emb.unsqueeze(1).expand([global_emb.shape[0], class_text_features.shape[0]] + 
                                                list(global_emb.shape[1:]))
        text_emb = class_emb.unsqueeze(0).expand([global_emb.shape[0], class_emb.shape[0]] +
                                        list(class_emb.shape[1:])) # bs, cls, (4,) 512
        return {'loss': loss}, \
                {
                    'acc': acc, 
                    'preds': preds, 
                    'final_scores': final_scores,
                    'skel_emb': skel_emb,
                    'text_emb': text_emb,
                    'out_emb': output_emb
                }