from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import os

from sklearn.metrics import log_loss

from config import Config
from pca_selection import process
from rankgauss import rankGauss
from train import train_test
from utils import seed_everything, process_data, sub_clip

from pdb import set_trace

def main():

    seed_everything(seed_value=42)
    cfg = Config()

    data_dir = '../../data'
    save_path = './'
    load_path = './'
    runty = 'traineval'

    train_features = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    train_features, test_features = rankGauss(
        train_features=train_features, test_features=test_features)

    train, test, targets_scored =   \
        process(train_features=train_features, test_features=test_features, 
                train_targets_scored=train_targets_scored,
                train_targets_nonscored=train_targets_nonscored,
                runty=runty, save_path=save_path, load_path=load_path)
        
    target = targets_scored
    folds = train.copy()

    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for seed in cfg.seeds:
        mskf = MultilabelStratifiedKFold(n_splits=cfg.nfolds, shuffle=True, random_state=seed)
        for fold, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
            folds.loc[v_idx, 'kfold'] = int(fold)
        folds['kfold'] = folds['kfold'].astype(int)
        
        trte = train_test(folds, test, target, save_path, load_path, runty=runty)
        
        if (runty == 'train'):
            oof_ = trte.run_k_fold(seed)
            oof += oof_ / len(cfg.seeds)
        elif (runty == 'eval'):
            predictions_ = trte.run_k_fold(seed)
            predictions += predictions_ / len(cfg.seeds)
        elif (runty == 'traineval'):
            oof_, predictions_ = trte.run_k_fold(seed)
            oof += oof_ / len(cfg.seeds)
            predictions += predictions_ / len(cfg.seeds)

        # oof_, predictions_ = trte.run_k_fold(seed)
        # oof += oof_ / len(cfg.seed)
        # predictions += predictions_ / len(cfg.seed)

    if (runty == 'train'):
        train[target_cols] = oof
        valid_results = train_targets_scored.drop(columns=target_cols).merge(
            train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        y_true = train_targets_scored[target_cols].values
        y_pred = valid_results[target_cols].values

        score = 0
        for i in range(len(target_cols)):
            score_ = log_loss(y_true[:, i], y_pred[:, i])
            score += score_ / (target.shape[1]-1)

        print("CV log_loss: ", score)

    elif (runty == 'eval'):
        test[target_cols] = predictions

        sub = submission.drop(columns=target_cols).merge(
            test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        # clip the submission
        # sub_c = sub_clip(sub, test_features)
        # sub_c.to_csv('submission.csv', index=False)

        sub.to_csv('submission.csv', index=False)
        
    elif (runty == 'traineval'):
        train[target_cols] = oof
        valid_results = train_targets_scored.drop(columns=target_cols).merge(
            train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        y_true = train_targets_scored[target_cols].values
        y_pred = valid_results[target_cols].values

        score = 0
        for i in range(len(target_cols)):
            score_ = log_loss(y_true[:, i], y_pred[:, i])
            score += score_ / (target.shape[1]-1)

        print("CV log_loss: ", score)
        
        test[target_cols] = predictions

        sub = submission.drop(columns=target_cols).merge(
            test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

        # clip the submission
        # sub_c = sub_clip(sub, test_features)
        # sub_c.to_csv('submission.csv', index=False)

        sub.to_csv('submission.csv', index=False)

    # train[target_cols] = oof
    # test[target_cols] = predictions

    # valid_results = train_targets_scored.drop(columns=target_cols).merge(
    #     train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

    # y_true = train_targets_scored[target_cols].values
    # y_pred = valid_results[target_cols].values

    # score = 0
    # for i in range(len(target_cols)):
    #     score_ = log_loss(y_true[:, i], y_pred[:, i])
    #     score += score_ / target.shape[1]

    # print("CV log_loss: ", score)

    # sub = submission.drop(columns=target_cols).merge(
    #     test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

    # # clip the submission
    # sub_c = sub_clip(sub, test_features)
    # sub_c.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
