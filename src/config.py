import numpy as np
import torch

class Config(object):
    def __init__(self, num_class, cat_tr, numerical_tr_num):
        # self.num_class = targets_tr.shape[1]
        self.num_class = num_class
        self.verbose=False
        self.seed = 0
        self.SPLITS=10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EPOCHS = 200
        self.num_ensembling = 1
        # Parameters model
        self.cat_emb_dim=[1] * cat_tr.shape[1] #to choose
        self.cats_idx = list(range(cat_tr.shape[1]))
        self.cat_dims = [len(np.unique(cat_tr[:, i])) for i in range(cat_tr.shape[1])]
        self.num_numericals= numerical_tr_num
    
        # save
        self.save_name = "tabnet_raw_step1"
        
        self.strategy = "KFOLD" # 
