import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Config_TabNet(object):
    def __init__(self, device_name='auto'):
        if (device_name == 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_name

        self.max_epochs = 1000
        self.batch_size = 1024
        self.virtual_batch_size = 32

        self.n_d = 32
        self.n_a = 128
        self.n_independent = 1
        self.n_shared = 1
        self.n_steps = 1
        self.gamma = 1.5
        self.lambda_sparse = 0.0

        self.optimizer_fn = optim.Adam
        self.optimizer_params = dict(lr=2e-2, weight_decay=1e-5)
        self.scheduler_params = dict(mode="min", patience=30, min_lr=1e-5, factor=0.9)
        self.mask_type = "entmax"
        self.scheduler_fn = ReduceLROnPlateau
        self.fit_patience = 80
        self.verbose = 20
        self.labelsmooth_rate = 0.0004


class Config_FeatureEngineer(object):
    def __init__(self):

        self.scale = "rankgauss"
        self.scale_n_quantiles = 100

        self.decompo = "PCA"
        self.pca_drop_orig = False
        self.genes_variance = 0.9
        self.cells_variance = 0.9

        self.n_clusters_g = 22
        self.n_clusters_c = 4

        self.feature_select = "variancethreshold"
        self.variancethreshold_for_FS = 0.8

        self.regularization_ls = False
        self.ls_rate = 0.0004

        self.encoding = "dummy"

        self.seed = 42

        self.seeds = [18, 19, 20, 21, 22]
        self.nfolds = 7
        self.drug_thresh = 18
