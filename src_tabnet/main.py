from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

import os

from config import Config
from dataloader import get_ratio_labels, transform_data
from model import TabNetRegressor
from loss import (log_loss_score, log_loss_multi, auc_multi, LabelSmoothingLoss, SmoothBCEwLogits)
from utils import seed_everything, check_targets

from pdb import set_trace

def main(runtype):
    assert runtype == 'train' or runtype == 'eval'

    seed_everything(42)

    data_dir = '../../data'
    train = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    remove_vehicle = True

    if remove_vehicle:
        train_features = train.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_scored = train_targets_scored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_nonscored = train_targets_nonscored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
    else:
        train_features = train

    col_features = list(train_features.columns)[1:]
    cat_tr, cat_test, numerical_tr, numerical_test = \
        transform_data(train_features, test_features, col_features,
                       normalize=False, removed_vehicle=remove_vehicle)

    columns, ratios = get_ratio_labels(train_targets_scored)
    columns_nonscored, ratios_nonscored = get_ratio_labels(train_targets_nonscored)
    targets_tr = train_targets_scored[columns].values.astype(np.float32)
    targets2_tr = train_targets_nonscored[columns_nonscored].values.astype(np.float32)

    cfg = Config(num_class = targets_tr.shape[1], cat_tr = cat_tr, numerical_tr_num=numerical_tr.shape[1])

    X_test = np.concatenate([cat_test, numerical_test], axis=1)

    if cfg.strategy == "KFOLD":
        oof_preds_all = []
        oof_targets_all = []
        scores_all =  []
        scores_auc_all= []
        preds_test = []
        for seed in range(cfg.num_ensembling):
            print("## SEED : ", seed)
            mskf = MultilabelStratifiedKFold(n_splits=cfg.SPLITS, random_state=cfg.seed+seed, shuffle=True)
            oof_preds = []
            oof_targets = []
            scores = []
            scores_auc = []
            p = []
            for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
                print("FOLDS : ", j)

                ## model
                X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
                X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
                model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, gamma=1.3, lambda_sparse=0, cat_dims=cfg.cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cfg.cats_idx, optimizer_fn=torch.optim.Adam,
                                        optimizer_params=dict(lr=2e-2, weight_decay=1e-5), mask_type='entmax', device_name=cfg.device, scheduler_params=dict(milestones=[ 100,150], gamma=0.9), scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)
                #'sparsemax'

                if (runtype == 'train'):
                    model.fit(X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,max_epochs=cfg.EPOCHS, patience=50, batch_size=512, virtual_batch_size=32,
                          num_workers=0, drop_last=False, loss_fn=LabelSmoothingLoss(classes=y_val.shape[1], smoothing=0.01))
                    # torch.nn.functional.binary_cross_entropy_with_logits)
                    model.load_best_model()

                name = cfg.save_name + f"_fold{j}_{seed}"
                if (runtype == 'train'):
                    model.save_model(name)
                if (runtype == 'eval'):
                    model.load_model_infer(name)

                # preds on val
                preds = model.predict(X_val)
                preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
                score = log_loss_multi(y_val, preds)

                # preds on test
                if (runtype == 'eval'):
                    temp = model.predict(X_test)
                    p.append(torch.sigmoid(torch.as_tensor(temp)).detach().cpu().numpy())
                ## save oof to compute the CV later
                oof_preds.append(preds)
                oof_targets.append(y_val)
                scores.append(score)
                scores_auc.append(auc_multi(y_val,preds))
                print(f"validation fold {j} : {score}")
            if (runtype == 'eval'):
                p = np.stack(p)
                preds_test.append(p)
            oof_preds_all.append(np.concatenate(oof_preds))
            oof_targets_all.append(np.concatenate(oof_targets))
            scores_all.append(np.array(scores))
            scores_auc_all.append(np.array(scores_auc))
        if (runtype == 'eval'):
            preds_test = np.stack(preds_test)

        for i in range(cfg.num_ensembling):
            print("CV score fold : ", log_loss_multi(oof_targets_all[i], oof_preds_all[i]))
            print("auc mean : ", sum(scores_auc_all[i]) / len(scores_auc_all[i]))

    else:
        i = 0
        mskf = MultilabelStratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state=0)
        oof_preds = []
        oof_targets = []
        scores = []
        scores_auc = []
        for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
            if i == cfg.SPLITS:
                break
            if not check_targets(targets_tr[train_idx]):
                continue
            print("FOLDS : ", i, j)

            ## model
            X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
            X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
            model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_dims=cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cats_idx, optimizer_fn=torch.optim.Adam,
                               optimizer_params=dict(lr=1e-3, amsgrad=True), mask_type="sparsemax", device_name=cfg.device)

            if (runtype == 'train'):
                model.fit(X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,max_epochs=cfg.EPOCHS, patience=50, batch_size=512, virtual_batch_size=32,
                    num_workers=0, drop_last=False, loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
                model.load_best_model()

            name = cfg.save_name + f"_{j}"
            if (runtype == 'train'):
                model.save_model(name)
            if (runtype == 'eval'):
                model.load_model_infer(name)

            # preds on val
            preds = model.predict(X_val)
            preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
            score = log_loss_multi(y_val, preds)

            # preds on test
            if (runtype == 'eval'):
                temp = model.predict(X_test)
                p.append(torch.sigmoid(torch.as_tensor(temp)).detach().cpu().numpy())

            ## save oof to compute the CV later
            oof_preds.append(preds)
            oof_targets.append(y_val)
            scores.append(score)
            scores_auc.append(auc_multi(y_val,preds))
            print(f"validation fold {j} : {score}")
            i+=1
            if (runtype == 'eval'):
                p = np.stack(p)
                preds_test.append(p)
            #break

        if (runtype == 'eval'):
            preds_test = np.stack(preds_test)
        print("auc mean : ", sum(scores_auc)/len(scores_auc))
        print("CV score : ", log_loss_multi(np.concatenate(oof_targets) , np.concatenate(oof_preds)))

    if (runtype == 'eval'):
        # clip the submission
        sub = np.clip(preds_test.mean(1).mean(0), 0.001, 0.995)
        submission[columns] = sub
        submission.loc[test_features['cp_type']=='ctl_vehicle', submission.columns[1:]] = 0
        submission.to_csv("submission.csv", index=False)

def train():
    seed_everything(42)

    data_dir = '../../data'
    train = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    remove_vehicle = True

    if remove_vehicle:
        train_features = train.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_scored = train_targets_scored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_nonscored = train_targets_nonscored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
    else:
        train_features = train

    col_features = list(train_features.columns)[1:]
    cat_tr, cat_test, numerical_tr, numerical_test = \
        transform_data(train_features, test_features, col_features,
                       normalize=False, removed_vehicle=remove_vehicle)

    columns, ratios = get_ratio_labels(train_targets_scored)
    columns_nonscored, ratios_nonscored = get_ratio_labels(train_targets_nonscored)
    targets_tr = train_targets_scored[columns].values.astype(np.float32)
    targets2_tr = train_targets_nonscored[columns_nonscored].values.astype(np.float32)

    cfg = Config(num_class = targets_tr.shape[1], cat_tr = cat_tr, numerical_tr_num=numerical_tr.shape[1])

    X_test = np.concatenate([cat_test, numerical_test], axis=1)

    if cfg.strategy == "KFOLD":
        oof_preds_all = []
        oof_targets_all = []
        scores_all =  []
        scores_auc_all= []
        for seed in range(cfg.num_ensembling):
            print("## SEED : ", seed)
            mskf = MultilabelStratifiedKFold(n_splits=cfg.SPLITS, random_state=cfg.seed+seed, shuffle=True)
            oof_preds = []
            oof_targets = []
            scores = []
            scores_auc = []
            for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
                print("FOLDS : ", j)

                ## model
                X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
                X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
                model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, gamma=1.3, lambda_sparse=0, cat_dims=cfg.cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cfg.cats_idx, optimizer_fn=torch.optim.Adam,
                                        optimizer_params=dict(lr=2e-2, weight_decay=1e-5), mask_type='entmax', device_name=cfg.device, scheduler_params=dict(milestones=[ 100,150], gamma=0.9), scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)
                #'sparsemax'

                model.fit(X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,max_epochs=cfg.EPOCHS, patience=50, batch_size=1024, virtual_batch_size=32,
                          num_workers=0, drop_last=False, loss_fn=SmoothBCEwLogits(smoothing=0.001))
                # torch.nn.functional.binary_cross_entropy_with_logits)
                model.load_best_model()
                preds = model.predict(X_val)
                preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
                score = log_loss_multi(y_val, preds)
                name = cfg.save_name + f"_fold{j}_{seed}"
                model.save_model(name)
                ## save oof to compute the CV later
                oof_preds.append(preds)
                oof_targets.append(y_val)
                scores.append(score)
                scores_auc.append(auc_multi(y_val,preds))
                print(f"validation fold {j} : {score}")
            oof_preds_all.append(np.concatenate(oof_preds))
            oof_targets_all.append(np.concatenate(oof_targets))
            scores_all.append(np.array(scores))
            scores_auc_all.append(np.array(scores_auc))

        for i in range(cfg.num_ensembling):
            print("CV score fold : ", log_loss_multi(oof_targets_all[i], oof_preds_all[i]))
            print("auc mean : ", sum(scores_auc_all[i]) / len(scores_auc_all[i]))

    else:
        i = 0
        mskf = MultilabelStratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state=0)
        oof_preds = []
        oof_targets = []
        scores = []
        scores_auc = []
        for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
            if i == cfg.SPLITS:
                break
            if not check_targets(targets_tr[train_idx]):
                continue
            print("FOLDS : ", i, j)

            ## model
            X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
            X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
            model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_dims=cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cats_idx, optimizer_fn=torch.optim.Adam,
                               optimizer_params=dict(lr=1e-3, amsgrad=True), mask_type="sparsemax", device_name=cfg.device)

            model.fit(X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,max_epochs=cfg.EPOCHS, patience=50, batch_size=512, virtual_batch_size=32,
                    num_workers=0, drop_last=False, loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
            model.load_best_model()
            preds = model.predict(X_val)
            preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
            score = log_loss_multi(y_val, preds)
            name = cfg.save_name + f"_{j}"
            model.save_model(name)
            ## save oof to compute the CV later
            oof_preds.append(preds)
            oof_targets.append(y_val)
            scores.append(score)
            scores_auc.append(auc_multi(y_val,preds))

            i+=1
            #break

        print("auc mean : ", sum(scores_auc)/len(scores_auc))
        print("CV score : ", log_loss_multi(np.concatenate(oof_targets) , np.concatenate(oof_preds)))

def eval():
    seed_everything(42)

    data_dir = '../../data'
    train = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(data_dir, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    remove_vehicle = True

    if remove_vehicle:
        train_features = train.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_scored = train_targets_scored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
        train_targets_nonscored = train_targets_nonscored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
    else:
        train_features = train

    col_features = list(train_features.columns)[1:]
    cat_tr, cat_test, numerical_tr, numerical_test = \
        transform_data(train_features, test_features, col_features,
                       normalize=False, removed_vehicle=remove_vehicle)

    columns, ratios = get_ratio_labels(train_targets_scored)
    columns_nonscored, ratios_nonscored = get_ratio_labels(train_targets_nonscored)
    targets_tr = train_targets_scored[columns].values.astype(np.float32)
    targets2_tr = train_targets_nonscored[columns_nonscored].values.astype(np.float32)

    cfg = Config(num_class = targets_tr.shape[1], cat_tr = cat_tr, numerical_tr_num=numerical_tr.shape[1])
    cfg.save_name = './tabnet_batchsize/tabnet_raw_step1'

    X_test = np.concatenate([cat_test, numerical_test], axis=1)

    if cfg.strategy == "KFOLD":
        oof_preds_all = []
        oof_targets_all = []
        scores_all =  []
        scores_auc_all= []
        preds_test = []
        for seed in range(cfg.num_ensembling):
            print("## SEED : ", seed)
            mskf = MultilabelStratifiedKFold(n_splits=cfg.SPLITS, random_state=cfg.seed+seed, shuffle=True)
            oof_preds = []
            oof_targets = []
            scores = []
            scores_auc = []
            p = []
            for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
                print("FOLDS : ", j)

                ## model
                X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
                X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
                model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, gamma=1.3, lambda_sparse=0, cat_dims=cfg.cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cfg.cats_idx, optimizer_fn=torch.optim.Adam,
                                   optimizer_params=dict(lr=2e-2), mask_type='entmax', device_name=cfg.device, scheduler_params=dict(milestones=[ 50,100,150], gamma=0.9), scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)
                #'sparsemax'
            
                name = cfg.save_name + f"_fold{j}_{seed}"
                model.load_model_infer(name)
                # preds on val
                preds = model.predict(X_val)
                preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
                score = log_loss_multi(y_val, preds)
            
                # preds on test
                temp = model.predict(X_test)
                p.append(torch.sigmoid(torch.as_tensor(temp)).detach().cpu().numpy())
                ## save oof to compute the CV later
                oof_preds.append(preds)
                oof_targets.append(y_val)
                scores.append(score)
                scores_auc.append(auc_multi(y_val,preds))
                print(f"validation fold {j} : {score}")
            p = np.stack(p)
            preds_test.append(p)
            oof_preds_all.append(np.concatenate(oof_preds))
            oof_targets_all.append(np.concatenate(oof_targets))
            scores_all.append(np.array(scores))
            scores_auc_all.append(np.array(scores_auc))
        preds_test = np.stack(preds_test)

        for i in range(cfg.num_ensembling): 
            print("CV score fold : ", log_loss_multi(oof_targets_all[i], oof_preds_all[i]))
            print("auc mean : ", sum(scores_auc_all[i])/len(scores_auc_all[i]))

    if cfg.strategy != "KFOLD":
        i = 0
        mskf = MultilabelStratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state=0)
        oof_preds = []
        oof_targets = []
        scores = []
        scores_auc = []
        for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(cat_tr)), targets_tr)):
            if i == cfg.SPLITS:
                break
            
            if not check_targets(targets_tr[train_idx]):
                continue
            print("FOLDS : ", i, j)

            ## model
            X_train, y_train = torch.as_tensor(np.concatenate([cat_tr[train_idx], numerical_tr[train_idx] ], axis=1)), torch.as_tensor(targets_tr[train_idx])
            X_val, y_val = torch.as_tensor(np.concatenate([cat_tr[val_idx], numerical_tr[val_idx] ], axis=1)), torch.as_tensor(targets_tr[val_idx])
            model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_dims=cat_dims, cat_emb_dim=cfg.cat_emb_dim, cat_idxs=cats_idx, optimizer_fn=torch.optim.Adam,
                               optimizer_params=dict(lr=1e-3, amsgrad=True), mask_type="sparsemax", device_name=cfg.device)
        
            name = cfg.save_name + f"_{j}"
            model.load_model(name)
            # preds on val
            preds = model.predict(X_val)
            preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()
            score = log_loss_multi(y_val, preds)

            # preds on test
            temp = model.predict(X_test)
            p.append(torch.sigmoid(torch.as_tensor(temp)).detach().cpu().numpy())
            ## save oof to compute the CV later
            oof_preds.append(preds)
            oof_targets.append(y_val)
            scores.append(score)
            scores_auc.append(auc_multi(y_val,preds))
            print(f"validation fold {j} : {score}")
            i+=1
            p = np.stack(p)
            preds_test.append(p)

        preds_test = np.stack(preds_test)

        print("auc mean : ", sum(scores_auc)/len(scores_auc))
        print("CV score : ", log_loss_multi(np.concatenate(oof_targets) , np.concatenate(oof_preds)))

    # clip the submission
    sub = np.clip(preds_test.mean(1).mean(0), 0.001, 0.995)
    submission[columns] = sub
    submission.loc[test_features['cp_type']=='ctl_vehicle', submission.columns[1:]] = 0
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    train()
