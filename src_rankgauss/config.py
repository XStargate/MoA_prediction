import numpy as np
import torch

class Config(object):
    def __init__(self, device_name='auto'):
        if (device_name == 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_name

        self.epochs = 25
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.nfolds = 5
        self.early_stopping_steps = 10
        self.early_stop = False
        self.hidden_size = 1500

        self.seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.loss_smooth = 0.001
