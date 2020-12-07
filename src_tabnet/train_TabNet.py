import numpy as np
import os
import pandas as pd

# Deep Learning
import torch

# Tabnet
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from config import Config_FeatureEngineer, Config_TabNet
from loss import SmoothBCEwLogits, LogitsLogLoss, Logloss, BCEwLogitsSmooth
from utils import seed_everything
from sklearn.ensemble import StackingClassifier
from torch.nn import functional as F
import shutil
from sklearn.linear_model import LinearRegression


def train_tabnet(x_train, y_train, x_test, submission, feature_cols, target_cols, seeds, nfolds, save_path):

    cfg_fe = Config_FeatureEngineer()
    seed_everything(seed_value=cfg_fe.seed)

    cfg_tabnet = Config_TabNet()

    test_cv_preds = []
    oof_preds = []
    scores = []

    for seed in seeds:
        kfold_col = f'kfold_{seed}'
        print("seed: {}".format(seed))
        print('*' * 60)

        for fold in range(nfolds):

            oof_preds_fold = y_train.copy()
            oof_preds_fold.iloc[:, :] = 0

            print('*' * 60)
            print("FOLD: {}".format(fold + 1))
            print('*' * 60)

            trn_idx = x_train[x_train[kfold_col] != fold].index
            val_idx = x_train[x_train[kfold_col] == fold].index

            train_df = x_train[x_train[kfold_col] != fold].reset_index(drop=True)
            valid_df = x_train[x_train[kfold_col] == fold].reset_index(drop=True)

            x_tr, y_tr = train_df[feature_cols].values, train_df[target_cols].values
            x_val, y_val = valid_df[feature_cols].values, valid_df[target_cols].values

            # tabnet model
            model = TabNetRegressor(n_d=cfg_tabnet.n_d,
                                    n_a=cfg_tabnet.n_a,
                                    n_steps=cfg_tabnet.n_steps,
                                    n_independent=cfg_tabnet.n_independent,
                                    n_shared=cfg_tabnet.n_shared,
                                    gamma=cfg_tabnet.gamma,
                                    lambda_sparse=cfg_tabnet.lambda_sparse,
                                    optimizer_fn=cfg_tabnet.optimizer_fn,
                                    optimizer_params=cfg_tabnet.optimizer_params,
                                    mask_type=cfg_tabnet.mask_type,
                                    scheduler_params=cfg_tabnet.scheduler_params,
                                    scheduler_fn=cfg_tabnet.scheduler_fn,
                                    seed=seed,
                                    verbose=cfg_tabnet.verbose)

            # fit model
            model.fit(
                X_train=x_tr,
                y_train=y_tr,
                eval_set=[(x_val, y_val)],
                eval_name=["val"],
                eval_metric=["logits_ll"],
                max_epochs=cfg_tabnet.max_epochs,
                patience=cfg_tabnet.fit_patience,
                batch_size=cfg_tabnet.batch_size,
                virtual_batch_size=cfg_tabnet.virtual_batch_size,
                num_workers=1,
                drop_last=False,
                # To use binary cross entropy because this is not a regression problem
                loss_fn=BCEwLogitsSmooth(smooth=cfg_tabnet.labelsmooth_rate)
            )

            print('-' * 60)

            # save model
            model.save_model(os.path.join(save_path, f"TabNet_seed{seed}_FOLD{fold}"))
            print('*' * 60)

            # Predict on validation
            preds_val = model.predict(x_val)
            # Apply sigmoid to the predictions
            preds = 1 / (1 + np.exp(-preds_val))
            score = np.min(model.history["val_logits_ll"])

            oof_preds.append(preds)
            scores.append(score)

            # Save OOF for CV
            preds_tr = model.predict(x_train[feature_cols].values)
            preds = 1 / (1 + np.exp(-preds_tr))
            oof_preds_fold.loc[:, target_cols] = preds
            oof_preds_fold.to_csv(
                path_or_buf=f"./TabNet_oof_preds_seed{seed}_FOLD{fold}.csv", sep=',', index=False)

            # Predict on test
            preds_test = model.predict(x_test[feature_cols].values)
            preds_test = 1 / (1 + np.exp(-preds_test))
            test_cv_preds.append(preds_test)
            test_cv_preds_fold = pd.DataFrame(preds_test, columns=target_cols)
            test_cv_preds_fold["sig_id"] = x_test["sig_id"]
            test_cv_preds_fold.to_csv(
                path_or_buf=f"./TabNet_test_preds_seed{seed}_FOLD{fold}.csv", sep=',', index=False)

    oof_preds_all = np.concatenate(oof_preds)
    test_preds_all = np.stack(test_cv_preds)
    print("Averaged Best Score for CVs is: {}".format(np.mean(scores)))

    return test_preds_all


def pred_tabnet(x_train, y_train, x_test, submission, feature_cols, target_cols, seeds, nfolds, load_path, stacking=False):

    cfg_tabnet = Config_TabNet()
    test_cv_preds = []
    oof_preds = []
    scores = []

    for seed in seeds:

        print('*' * 60)
        kfold_col = f'kfold_{seed}'
        print("seed: {}".format(seed))
        print('*' * 60)

        for fold in range(nfolds):

            oof_preds_fold = y_train.copy()
            oof_preds_fold.iloc[:, :] = 0
            test_cv_preds_fold = submission.copy()
            test_cv_preds_fold.iloc[:, :] = 0

            print("FOLD: {}".format(fold + 1))
            print('*' * 60)

            train_df = x_train[x_train[kfold_col] != fold].reset_index(drop=True)
            valid_df = x_train[x_train[kfold_col] == fold].reset_index(drop=True)

            x_val, y_val = valid_df[feature_cols].values, valid_df[target_cols].values

            x_tot, y_tot = x_train[feature_cols].values, y_train[target_cols].values

            # tabnet model
            model = TabNetRegressor()

            # save model
            path = os.path.join(load_path, f"TabNet_seed{seed}_FOLD{fold}")
            if os.path.exists(path+".zip"):
                model.load_model(path+".zip")
            else:
                tmppath = os.path.join("./", f"TabNet_seed{seed}_FOLD{fold}")
                shutil.make_archive(tmppath, "zip", path)
                model.load_model(tmppath+".zip")
                os.remove(tmppath+".zip")

            # Predict on validation
            preds_val = model.predict(x_val)

            # Apply sigmoid to the predictions
            preds = 1 / (1 + np.exp(-preds_val))
            score = Logloss(y_val, preds)
            scores.append(score)
            print(f"TabNet, seed{seed}, FOLD{fold}, CV predict loss: {score}")
            print('*' * 60)

            # predict on the whole train set for sacking
            preds_tot = model.predict(x_tot)
            preds_tot = 1 / (1 + np.exp(-preds_tot))
            oof_preds.append(preds_tot)

            # Predict on test
            preds_test = model.predict(x_test[feature_cols].values)
            preds_test = 1 / (1 + np.exp(-preds_test))
            test_cv_preds.append(preds_test)

    oof_preds_all = np.stack(oof_preds)
    test_preds_all = np.stack(test_cv_preds)
    print("Averaged Best Score for CVs is: {}".format(np.mean(scores)))

    if not stacking:
        test_pred_final = test_preds_all.mean(axis=0)
    else:
        print("stacking...")
        num_models = len(seeds)*nfolds
        test_pred_final = np.zeros(test_preds_all.shape[1:])
        weights = np.zeros(num_models)

        # stacking method
        oof_preds_all = np.array(oof_preds_all)
        oof_preds_all = np.reshape(oof_preds_all, (num_models, -1))
        y_target = np.array(y_tot)
        y_target = np.reshape(y_target, (y_target.shape[0]*y_target.shape[1], -1))
        oof_preds_all = oof_preds_all.T
        print(f"oof shape is {oof_preds_all.shape}")
        print(f"targets is {y_target.shape}")

        # calculate blend weights
        reg = LinearRegression().fit(oof_preds_all, y_target)
        weights = reg.coef_[0]
        intercept = reg.intercept_
        test_pred_final[:, :] = intercept
        print(f"intercept is {intercept}")
        print(f"weights are {weights}")

        for idx in range(num_models):
            test_pred_final += test_preds_all[idx]*weights[idx]

        test_pred_final = np.clip(test_pred_final, 0, 1)

    return test_pred_final
