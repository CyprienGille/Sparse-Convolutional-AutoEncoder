# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:14:35 2021

Ensuring reproducibility when using PyTorch

@author: YFGI6212
"""
import torch
import numpy as np
import random

def seed_all(seed, cuda = True):
    if not seed:
        seed = 10
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """ Necessary step in order to ensure data balancing between 
        dataloaders workers. This is only necessary when more than
        one worker is used.
        
        Typical usage: 
               training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                                 batch_size     = bs
                                                                 shuffle        = shuf,
                                                                 pin_memory     = pin,
                                                                 num_workers    = 2,
                                                                 worker_init_fn = seed_worker,
                                                                 drop_last      = dl) # ensure reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(0)
    random.seed(0)

