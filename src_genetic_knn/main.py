import os

import numpy as np
import pandas as pd

from config import Config
from utils import process_data_cp, make_folds

from pdb import set_trace

def main():
    cfg = Config()
    
    data_dir = '../../data'
    
    train = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    targets = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    drug = pd.read_csv(os.path.join(data_dir, 'train_drug.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    
    train = process_data_cp(train)
    test = process_data_cp(test)
    
    ff = make_folds(drug=drug, folds=cfg.folds, random_state=cfg.seed, 
                    stratify=True, scored=targets)
    
    train['fold'] = ff.fold.values
    targets['fold'] = ff.fold.values
    
    set_trace()
    # initialize
    oof = np.zeros((len(train), targets.shape[1]-2))
    dna = np.random.uniform(0,1,(cfg.population,875))**2.0
    cvs = np.zeros((cfg.population))

if __name__ == '__main__':
    main()