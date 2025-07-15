import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import yaml
from model import MInterface
from data import DInterface
from data.utils import load_labels
from utils import str2bool, TBLogger
import sys
import numpy as np
import torch
import warnings
import shutil
from collections import Counter
warnings.filterwarnings("ignore")
import copy
import argparse
import torch.nn.functional as F
import torch.nn as nn
import time
import clip  # Import the CLIP package

# Import optimized utils
from model.utils.test_time_utils import (
    load_callbacks, get_trainer, find_sample_index, generate_text_features,
    run_dynamic_refinement, run_evaluation
)

torch.set_float32_matmul_precision('high')

class DynaPURLSProcessor:
    def __init__(self, model, data_module, hparams):
        self.model = model
        self.data_module = data_module
        self.hparams = hparams
        self.best_acc = -1
        self.test_acc = -1

        # Use a single prompt per class
        self.generate_text_features()

    def generate_text_features(self):
        # Use the optimized function from utils
        self.text_features = generate_text_features(self.model)

    def load_data(self):
        self.test_loader = self.data_module.zsl_test_dataloader()

    def find_sample_index(self, buffer, sample_to_find):
        # Use the optimized function from utils
        return find_sample_index(buffer, sample_to_find)

    def dynamic_refinement(self, steps=20, lr=0.001, buffer_size=256, confidence_threshold=0.5):
        # Use the optimized function from utils
        return run_dynamic_refinement(
            self.model, self.test_loader, self.hparams,
            steps=steps, lr=lr, buffer_size=buffer_size, 
            confidence_threshold=confidence_threshold
        )

    def evaluate(self):
        # Use the optimized function from utils
        trainer = get_trainer(self.hparams)
        return run_evaluation(self.model, trainer, self.test_loader)

def main(args):
    hparams = copy.deepcopy(args)
    hparams.tb_folder = '/'.join((hparams.work_dir).split('/')[:-1]) + '/tensorboard'
    hparams.work_dir_name = (hparams.work_dir).split('/')[-1]
    print("Current Experiment Configs: {}".format(hparams))
    pl.seed_everything(hparams.seed)
    hparams.num_class, hparams.emb_dim, hparams.unseen_inds, \
    hparams.seen_inds, hparams.cls_labels \
        = load_labels(hparams.root, hparams.split, hparams.dataloader, hparams.model_name)
    hparams.seen_labels = [hparams.cls_labels[i] for i in hparams.seen_inds]
    hparams.unseen_labels = [hparams.cls_labels[i] for i in hparams.unseen_inds]
    hparams.bp_num = 4
    hparams.t_num = 3
    print("Load data module.")
    data_module = DInterface(**vars(hparams))
    data_module.setup()
    print("Load model module.")
    model = MInterface(**vars(hparams))
    model.set_text_features(data_module.text_features)
    dynapurls_processor = DynaPURLSProcessor(model, data_module, hparams)
    dynapurls_processor.load_data()
    best_model_path = f'{hparams.work_dir}/best.ckpt'
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    model.zsl()
    dynapurls_processor.evaluate()

    lr_range = [1e-2]
    buffer_size_range = [16]  # Increased minimum buffer size
    confidence_threshold_range = [0.1]
    step_range = [1]

    best_acc = 0
    best_config = {}

    # Grid search over all hyperparameter combinations
    for lr in lr_range:
        for buffer_size in buffer_size_range:
            for confidence_threshold in confidence_threshold_range:
                for steps in step_range:
                    print(f"Confidence Threshold: {confidence_threshold}")
                    new_model = copy.deepcopy(model)
                    dynapurls_processor = DynaPURLSProcessor(new_model, data_module, hparams)
                    dynapurls_processor.load_data()
                    
                    acc, total_time, avg_time, results = dynapurls_processor.dynamic_refinement(
                        steps=steps,
                        lr=lr,
                        buffer_size=buffer_size,
                        confidence_threshold=confidence_threshold
                    )
                    
                    print(f"Test Accuracy: {acc:.4f}")
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {
                            'lr': lr,
                            'buffer_size': buffer_size,
                            'confidence_threshold': confidence_threshold,
                            'steps': steps
                        }

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic arguments
    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # Processor
    parser.add_argument('--gpus', type=str2bool, default=-1, help='use GPUs or not')
    parser.add_argument('--num_epoch', type=int, help='stop training in which epoch')

    # Visualize and debug
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')

    # Data
    parser.add_argument('--root', help='root repo to load data')
    parser.add_argument('--root2', help='feature data path')
    parser.add_argument('--dataset', help='type of dataset: shift_5_r')
    parser.add_argument('--dataloader', help='class of the dataloader: ntu60')
    parser.add_argument('--data_type', help='how to process the input skeleton data')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('-ss', '--split', type=int, help='Which split to use: 5 or 12')
    parser.add_argument('-b', '--backbone', default='shift', help='encoder backbone')
    
    # Model
    parser.add_argument('--model_name', help='select the training config for a specific model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_args', default=dict(), help='model configs')
    
    # Ablation
    parser.add_argument('--activate_train', type=str2bool, default=False, help='activate training process')
    parser.add_argument('--test_p', default=False, type=str2bool, help='activate hyperparameter testing')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume training')
    parser.add_argument('--accumulate_grad_batches', default=0, help='accumulate grad batches')
    
    p = parser.parse_args(sys.argv[1:])
    if p.config is not None:
        with open(p.config, 'r') as f:
            input_args = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in input_args.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**input_args)
    args = parser.parse_args()
    if not args.test_p:
        main(args)
    else:
        pass