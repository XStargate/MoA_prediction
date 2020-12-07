from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import random
import pandas as pd
import torch
import os


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def onehot_encoding(data):
    print("one-hot encoding on cp_type, cp_time and cp_dose")
    data = pd.get_dummies(data, columns=['cp_type', 'cp_time', 'cp_dose'])
    return data


def remove_ctl(x_train, x_test, y_train):

    y_train = y_train[x_train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    x_train = x_train[x_train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    x_test = x_test[x_test['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    x_train = x_train.drop('cp_type', axis=1)
    x_test = x_test.drop('cp_type', axis=1)

    return x_train, x_test, y_train


def sub_clip(submission, minval=0.001, maxval=0.995):
    """
    Clip the submission
    """

    tmp = submission.drop(['sig_id'], axis=1)
    tmp_c = tmp.clip(minval, maxval)
    sub_ = pd.concat([submission['sig_id'], tmp_c], axis=1)
    # sub_.loc[test_features['cp_type']=='ctl_vehicle', sub_.columns[1:]] = 0

    return sub_


def make_cv_folds(x_train, seeds, nfolds, drug_thresh, target_cols):
    vc = x_train.drug_id.value_counts()
    vc1 = vc.loc[vc <= drug_thresh].index.sort_values()
    vc2 = vc.loc[vc > drug_thresh].index.sort_values()

    for seed_id in seeds:
        kfold_col = 'kfold_{}'.format(seed_id)

        # STRATIFY DRUGS 18X OR LESS
        dct1 = {}
        dct2 = {}

        skf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed_id)
        tmp = x_train.groupby('drug_id')[target_cols].mean().loc[vc1]

        for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
            dd = {k: fold for k in tmp.index[idxV].values}
            dct1.update(dd)

        # STRATIFY DRUGS MORE THAN 18X
        skf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed_id)
        tmp = x_train.loc[x_train.drug_id.isin(vc2)].reset_index(drop=True)

        for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
            dd = {k: fold for k in tmp.sig_id[idxV].values}
            dct2.update(dd)

        # ASSIGN FOLDS
        x_train[kfold_col] = x_train.drug_id.map(dct1)
        x_train.loc[x_train[kfold_col].isna(), kfold_col] = x_train.loc[x_train[kfold_col].isna(),
                                                                        'sig_id'].map(dct2)
        x_train[kfold_col] = x_train[kfold_col].astype('int8')

    return x_train
