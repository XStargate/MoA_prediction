import numpy as np
import torch

class Config(object):
    def __init__(self, device_name='auto'):
        if (device_name == 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_name

        self.epochs = 24
        # self.epochs = 1
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.weight_decay = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
        self.max_lr = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
        self.div_factor = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
        self.pct_start = 0.1
        self.nfolds = 7
        # self.nfolds = 2
        self.early_stopping_steps = 10
        self.early_stop = False
        self.hidden_size = 1500

        self.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.seeds = [0]
        self.loss_smooth = 0.001
        
        self.ncomp_g = 600
        self.ncomp_c = 60
        
        self.drug_thresh = 18
