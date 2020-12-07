from scaling import scaling
from labelsmoothing import ls_manual
from utils import seed_everything, onehot_encoding, remove_ctl, make_cv_folds
from train_TabNet_kfold import train_tabnet_kfold, pred_tabnet_kfold
from fe_stats import fe_stats
from cluster_kmeans import fe_cluster
from config import Config_FeatureEngineer, Config_TabNet
from sklearn.metrics import log_loss
import warnings
import datetime
from time import time
import os
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# read config
cfg_fe = Config_FeatureEngineer()
seed_everything(seed_value=cfg_fe.seed)

data_dir = '/kaggle/input/lish-moa/'
save_path = './'
load_path = '/kaggle/input/moatabnetmultimodelkfold/'
runty = 'eval'

# read data
train = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
train_drug = pd.read_csv(os.path.join(data_dir, 'train_drug.csv'))
submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))


x_train = train.copy()
x_test = test.copy()
y_train = targets_scored.copy()

cp_features = ['cp_type', 'cp_time', 'cp_dose']
genes_features = [column for column in train.columns if 'g-' in column]
cells_features = [column for column in train.columns if 'c-' in column]


# scale the data, like RankGauss
x_train, x_test = scaling(x_train, x_test, scale=cfg_fe.scale,
                          n_quantiles=cfg_fe.scale_n_quantiles, seed=cfg_fe.seed)

# fe_stats
x_train, x_test = fe_stats(x_train, x_test, genes_features, cells_features)
x_train.head()

# group the drug using kmeans
if runty == 'traineval':
    x_train, x_test = fe_cluster(x_train, x_test, genes_features, cells_features,
                                 n_cluster_g=cfg_fe.n_clusters_g, n_cluster_c=cfg_fe.n_clusters_c, seed=cfg_fe.seed, runty=runty, path=save_path)
elif runty == 'eval':
    x_train, x_test = fe_cluster(x_train, x_test, genes_features, cells_features,
                                 n_cluster_g=cfg_fe.n_clusters_g, n_cluster_c=cfg_fe.n_clusters_c, seed=cfg_fe.seed, runty=runty, path=load_path)

# one-hot encoding
x_train = onehot_encoding(x_train)
x_test = onehot_encoding(x_test)

feature_cols = [c for c in x_train.columns if (str(c)[0:5] != 'kfold' and c not in [
    'sig_id', 'drug_id', 'cp_type', 'cp_time', 'cp_dose'])]
target_cols = [x for x in y_train.columns if x != 'sig_id']


# label smoothing
if cfg_fe.regularization_ls:
    y_train = ls_manual(y_train, ls_rate=cfg_fe.ls_rate)

# merge drug_id and labels
x_train = x_train.merge(y_train, on='sig_id')
x_train = x_train.merge(train_drug, on='sig_id')

# remove sig_id
# x_train, x_test, y_train = remove_ctl(x_train, x_test, y_train)

# make CVs
target_cols = target_cols = [x for x in targets_scored.columns if x != 'sig_id']
x_train = make_cv_folds(x_train, cfg_fe.seeds, cfg_fe.nfolds, cfg_fe.drug_thresh, target_cols)


# train or make prediction
begin_time = datetime.datetime.now()
if (runty == 'traineval'):
    test_preds_all = train_tabnet_kfold(x_train, y_train, x_test, submission,
                                        feature_cols, target_cols, cfg_fe.seeds, cfg_fe.nfolds, save_path)
    y_train = targets_scored[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test_pred_final = pred_tabnet_kfold(x_train, y_train, x_test, submission, feature_cols,
                                        target_cols, cfg_fe.seeds, cfg_fe.nfolds, load_path='./', stacking=False)
elif (runty == 'eval'):
    y_train = targets_scored[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test_pred_final = pred_tabnet_kfold(x_train, y_train, x_test, submission, feature_cols,
                                        target_cols, cfg_fe.seeds, cfg_fe.nfolds, load_path, stacking=False)

time_diff = datetime.datetime.now() - begin_time
print(f'Total time is {time_diff}')

# make submission
all_feat = [col for col in submission.columns if col not in ["sig_id"]]
# To obtain the same lenght of test_preds_all and submission
sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop=True)
tmp = pd.DataFrame(test_pred_final, columns=all_feat)
tmp["sig_id"] = sig_id

submission = pd.merge(test[["sig_id"]], tmp, on="sig_id", how="left")
submission.fillna(0, inplace=True)

submission.to_csv("submission_tabnet.csv", index=None)
