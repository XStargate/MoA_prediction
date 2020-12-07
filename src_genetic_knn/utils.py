from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import random
import pandas as pd
from sklearn.model_selection import KFold
import torch
import os

from pdb import set_trace

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

def process_data_cp(data):
    data.cp_dose = data.cp_dose.map({'D1':-1,'D2':1})
    data.cp_time = data.cp_time.map({24:-1, 48:0, 72:1})
    data.cp_type = data.cp_type.map({'trt_cp':-1, 'ctl_vehicle':1})
    
    return data

def sub_clip(sub, test_features, minval=0.0005, maxval=0.999):
    """
    Clip the submission
    """

    tmp = sub.drop(['sig_id'], axis=1)
    tmp_c = tmp.clip(minval, maxval)
    sub_ = pd.concat([sub['sig_id'], tmp_c], axis=1)
    # sub_.loc[test_features['cp_type']=='ctl_vehicle', sub_.columns[1:]] = 0

    return sub_


def make_folds(drug, scored, folds, random_state, stratify=True, drug_thresh=18):
    targets = scored.columns[1:]
    scored = scored.merge(drug, on='sig_id', how='left')
    
    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc<=drug_thresh].index.sort_values()
    vc2 = vc.loc[vc>drug_thresh].index.sort_values()
    
    # STRATIFY DRUGS 18 OR LESS
    dct1 = {}; dct2 = {}
    if stratify:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, 
                                        random_state=random_state)
    else:
        skf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)
    
    # STRATIFY DRUGS MORE THAN 18
    if stratify:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, 
                                        random_state=random_state)
    else:
        skf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)
    
    # ASSIGN FOLDS
    scored['fold'] = np.nan
    scored['fold'] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(),'fold'] = scored.loc[scored.fold.isna(),'sig_id'].map(dct2)
    scored.fold = scored.fold.astype('int8')
    
    return scored[['sig_id','fold']].copy()
