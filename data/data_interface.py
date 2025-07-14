import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split, ConcatDataset
import clip
import torch
import os

class DInterface(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.load_data_module()
        self.text_features = None
        

    def divide_by_proportions(self, proportions, dataset):
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        return lengths

    def setup(self, stage=None):
        # seen class train
        self.seen_train = self.instancialize(sample_type = 'seen_train') # seen class train
        self.seen_val =  self.instancialize(sample_type = 'seen_val') 
        self.zsl_test = self.instancialize(sample_type = 'zsl_val') 
        self.gzsl_test = ConcatDataset([self.seen_val, self.zsl_test])
        
        self.prepare_text_features()

    # training dataloaders
    def seen_train_dataloader(self):
        return DataLoader(self.seen_train, pin_memory=True, batch_size=self.batch_size, shuffle=True)
            
    # validate dataloaders
    def seen_val_dataloader(self):
        return DataLoader(self.seen_val, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def zsl_test_dataloader(self):
        return DataLoader(self.zsl_test, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def gzsl_test_dataloader(self):
        return DataLoader(self.gzsl_test, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def load_data_module(self):
        # decide training type
        name = self.dataloader
        # Change the `snake_case.py` file name to `CamelCase` class name.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        self.data_module = getattr(importlib.import_module(
            '.'+name, package=__package__), camel_name)
    
    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.__dict__.keys()
        args1 = {}
        for arg in class_args: # update default arg under Dinterface
            if arg in inkeys:
                args1[arg] = self.__dict__[arg]
        args1.update(other_args) # update specialized arg from functional calls
        return self.data_module(**args1)

    def prepare_text_features(self):
        # Build feature cache file path
        cache_path = os.path.join(self.work_dir, 'text_features.pt')
        
        # Load directly if cache file exists
        if os.path.exists(cache_path):
            self.text_features = torch.load(cache_path)
            return
        
        # Extract features if cache doesn't exist
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        cls_tokens = [clip.tokenize(self.cls_labels[:, i]) for i in range(self.bp_num + self.t_num + 1)]
        text_features = []
        for tokens in cls_tokens:
            curr_text_features = clip_model.encode_text(tokens.to(self.device)).float()
            curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
            text_features.append(curr_text_features.unsqueeze(0))
        self.text_features = torch.cat(text_features, dim=0).to(self.device)
        self.text_features = self.text_features.permute(1, 0, 2).contiguous()
        
        # Save features to cache file
        torch.save(self.text_features, cache_path)