import json
import os

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import tensorflow as tf

from config import Config
from model import train_test
from loss import logloss
from pca_selection import process, process_input2
from rankgauss import rankGauss
from utils import preprocessor, preprocessor_2

from pdb import set_trace


def main():

    cfg = Config()

    data_dir = '../../data'
    save_path = './'
    load_path = './'
    runty = 'eval'
    assert runty == 'traineval' or runty == 'eval',  \
        "Run type is wrong. Should be 'traineval' or 'eval'"

    train_features = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    train_targets_scored = train_targets_scored.drop(['sig_id'], axis=1)
    train_targets_nonscored = train_targets_nonscored.drop(['sig_id'], axis=1)
    
    non_ctl_idx = train_features.loc[train_features['cp_type']!='ctl_vehicle'].index.to_list()
    train_features = train_features.drop(['sig_id', 'cp_type','cp_time','cp_dose'], axis=1)
    train_features = train_features.iloc[non_ctl_idx]
    train_targets_scored = train_targets_scored.iloc[non_ctl_idx]
    train_targets_nonscored = train_targets_nonscored.iloc[non_ctl_idx]
    test_features = test_features.drop(['sig_id', 'cp_dose', 'cp_time'], axis=1)
    
    set_trace()
    
    gs = train_features.columns.str.startswith('g-')
    cs = train_features.columns.str.startswith('c-')
    
    # read the main predictors
    with open('./t-test-pca-rfe-logistic-regression/main_predictors.json') as f:
        tmp = json.load(f)
        preds = tmp['start_predictors']

    oof = tf.constant(0.0)
    predictions = np.zeros((test_features.shape[0], train_targets_scored.shape[1]))

    for seed in cfg.seeds:
        
        mskf = MultilabelStratifiedKFold(n_splits=cfg.nfolds, shuffle=True, 
                                         random_state=seed)

        for f, (t_idx, v_idx) in enumerate(mskf.split(X=train_features, y=train_targets_scored)):
            x_train, x_valid = preprocessor(train_features.iloc[t_idx].values, 
                                            train_features.iloc[v_idx].values, gs, cs)
            _, data_test = preprocessor(train_features.iloc[t_idx].values, 
                                        test_features.drop('cp_type', axis=1).values, 
                                        gs, cs)
            x_train_2, x_valid_2 =   \
                preprocessor_2(train_features.iloc[t_idx][preds].values, 
                               train_features.iloc[v_idx][preds].values)
            _, data_test_2 = preprocessor_2(train_features.iloc[t_idx][preds].values, 
                                           test_features[preds].values)
            y_train_sc = train_targets_scored.iloc[t_idx].values
            y_train_ns = train_targets_nonscored.iloc[t_idx].values
            y_valid_sc = train_targets_scored.iloc[v_idx].values
            y_valid_ns = train_targets_nonscored.iloc[v_idx].values
            n_features = x_train.shape[1]
            n_features_2 = x_train_2.shape[1]
            
            trte = train_test(x_train=x_train, x_valid=x_valid, data_test=data_test,
                              x_train_2=x_train_2, x_valid_2=x_valid_2,
                              data_test_2=data_test_2, y_train_sc=y_train_sc,
                              y_train_ns=y_train_ns, y_valid_sc=y_valid_sc,
                              y_valid_ns=y_valid_ns, save_path=save_path,
                              load_path=load_path, fold=f, runty=runty)
            
            y_val, predictions_ = trte.run_k_fold(seed)
            oof += logloss(tf.constant(y_valid_sc, dtype=tf.float32), 
                           tf.constant(y_val, dtype=tf.float32)) / (cfg.nfolds*len(cfg.seeds))
            predictions += predictions_ / (cfg.nfolds * len(cfg.seeds))

    print("CV log_loss: ", oof)
    
    target_cols = train_targets_scored.columns
    
    sub.iloc[:, 1:] = predictions
    sub.loc[test_features['cp_type'] == 'ctl_vehicle', sub.columns[1:]] = 0

    # clip the submission
    # sub_c = sub_clip(sub, test_features)
    # sub_c.to_csv('submission.csv', index=False)
        
    # sub.loc[test_features['cp_type']=='ctl_vehicle', submission.columns[1:]] = 0
    sub.to_csv('submission.csv', index=False)
    
    """ if (runty == 'train'):
        train1[target_cols] = oof
        valid_results = train_targets_scored.drop(columns=target_cols).merge(
            train1[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        y_true = train_targets_scored[target_cols].values
        y_pred = valid_results[target_cols].values

        score = 0
        for i in range(len(target_cols)):
            score_ = log_loss(y_true[:, i], y_pred[:, i])
            score += score_ / target.shape[1]

        print("CV log_loss: ", score)

    elif (runty == 'eval'):
        test1[target_cols] = predictions

        sub = submission.drop(columns=target_cols).merge(
            test1[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        # clip the submission
        # sub_c = sub_clip(sub, test_features)
        # sub_c.to_csv('submission.csv', index=False)
        
        sub.loc[test_features['cp_type']=='ctl_vehicle', submission.columns[1:]] = 0
        sub.to_csv('submission.csv', index=False) """

if __name__ == '__main__':
    main()
