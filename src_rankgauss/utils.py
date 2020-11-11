import numpy as np
import pandas as pd
import random
import pandas as pd
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

def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data

def sub_clip(sub, test_features, minval=0.001, maxval=0.995):
    """
    Clip the submission
    """

    tmp = sub.drop(['sig_id'], axis=1)
    tmp_c = tmp.clip(minval, maxval)
    sub_ = pd.concat([sub['sig_id'], tmp_c], axis=1)
    sub_.loc[test_features['cp_type']=='ctl_vehicle', sub_.columns[1:]] = 0

    return sub_



