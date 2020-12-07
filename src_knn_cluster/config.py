import numpy as np
import torch

class Config(object):
    def __init__(self, device_name='auto'):
        if (device_name == 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_name

        self.epochs = 35
        # self.epochs = 1
        self.batch_size = 256
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5
        self.nfolds = 7
        # self.nfolds = 2
        self.early_stopping_steps = 15
        self.early_stop = True
        self.hidden_size = 2048
        # self.hidden_size = 128

        # self.seeds = [42, 44, 88, 38, 27, 90, 98, 18, 68, 86]
        self.seeds = [16, 26, 36, 46, 56, 66, 76, 86, 96, 60]
        # self.seeds = [0, 1, 2, 3, 4, 5, 6]
        # self.seeds = [0]
        self.loss_smooth = 0.001
        
        self.ncomp_g = 600
        self.ncomp_c = 50
