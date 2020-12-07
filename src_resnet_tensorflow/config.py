import numpy as np


class Config(object):
    def __init__(self, ncomp_g=463, ncomp_c=60):
        self.verbose=True
        self.nfolds=10
        # self.nfolds=2
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.hidden_size = 1024
        # self.hidden_size = 64
        self.epochs_ns = 50
        self.epochs_sc = 50
        # self.epochs_ns = 1
        # self.epochs_sc = 1
        self.patience_ns = 15
        self.patience_sc = 15
        self.early_stop = True
        self.batchsize = 128
        self.num_ensembling = 1
        # self.seeds = [42, 44, 48, 84, 88, 442, 444, 448, 484, 488]
        self.seeds = [23, 228, 1488, 1998, 2208, 2077, 404]
        # self.seeds = [0]

        # save
        self.save_name = "resnet"

        # self.ncomp_g = 600
        # self.ncomp_c = 60
        self.ncomp_g = 2
        self.ncomp_c = 100
        self.label_smoothing = 0.0005
