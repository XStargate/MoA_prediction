import os
from time import time


import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import Config
from pca_selection import process
from rankgauss import rankGauss
from train_orig import train_test
from utils import seed_everything, process_data, make_cv_folds

from pdb import set_trace

def main():
    
    seed_everything(seed_value=42)
    cfg = Config()
    
    data_dir = '../../data'
    save_path = './'
    load_path = './'
    runty = 'traineval'
    assert runty == 'traineval' or runty == 'eval',  \
        "Run type is wrong. Should be 'traineval' or 'eval'"
    
    train_features = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    train_drug = pd.read_csv(os.path.join(data_dir, 'train_drug.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    train_features, test_features = rankGauss(
        train_features=train_features, test_features=test_features)

    train, test, targets_scored, targets_nonscored =   \
        process(train_features=train_features, test_features=test_features, 
                train_targets_scored=train_targets_scored,
                train_targets_nonscored=train_targets_nonscored,
                train_drug=train_drug, runty=runty, save_path=save_path, 
                load_path=load_path)
    
    target_cols = [x for x in train_targets_scored.columns if x != 'sig_id']
    
    train = make_cv_folds(train, cfg.seeds, cfg.nfolds, cfg.drug_thresh, target_cols)
    
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    trte = train_test(train, test, targets_scored, targets_nonscored, save_path, load_path, runty)
    
    time_begin = time()
    
    for seed in cfg.seeds:
        if (runty == 'traineval'):
            oof_, predictions_ = trte.run_k_fold(seed)
            oof += oof_ / len(cfg.seeds)
            predictions += predictions_ / len(cfg.seeds)
        elif (runty == 'eval'):
            predictions_ = trte.run_k_fold(seed)
            predictions += predictions_ / len(cfg.seeds)
        
    time_diff = time() - time_begin
    
    train[target_cols] = oof
    test[target_cols] = predictions
    
    valid_results = train_targets_scored.drop(columns=target_cols).merge(
        train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

    y_true = train_targets_scored[target_cols].values
    y_pred = valid_results[target_cols].values

    if (runty == 'traineval'):
        score = 0
        for i in range(len(target_cols)):
            score += log_loss(y_true[:, i], y_pred[:, i])

        print("CV log_loss: ", score / y_pred.shape[1])
    
    sub = submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
    sub.to_csv('submission.csv', index=False)
    

if __name__ == '__main__':
    main()
    
    
    