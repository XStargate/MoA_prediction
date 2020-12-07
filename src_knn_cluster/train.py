import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os

from config import Config
from dataloader import MoADataset, TestDataset
from model import Model, Model_old
from loss import SmoothBCEwLogits
from utils import seed_everything, process_data

from pdb import set_trace

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        # valid_preds.append(outputs.detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())
        # preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds

class train_test():
    def __init__(self, folds, test, target, save_path, load_path, runty='train'):
        self.folds = folds
        self.test = test
        self.target = target
        self.save_path = save_path
        self.load_path = load_path

        assert runty == 'train' or runty == 'eval' or runty == 'traineval',  \
            "Run type is wrong. Should be 'train' or 'eval'. "

        self.runty = runty
        self.cfg = Config()

    def run_training(self, fold, seed):

        seed_everything(seed)

        train = process_data(self.folds)
        test_ = process_data(self.test)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True)
        valid_df = train[train['kfold'] == fold].reset_index(drop=True)

        target_cols = self.target.drop('sig_id', axis=1).columns.values.tolist()

        feature_cols = [c for c in process_data(self.folds).columns if c not in target_cols]
        feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]

        x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
        x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        model = Model_old(
            num_features=len(feature_cols),
            num_targets=len(target_cols),
            hidden_size=self.cfg.hidden_size,
        )

        model.to(self.cfg.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                                  max_lr=8e-3, epochs=self.cfg.epochs,
                                                  steps_per_epoch=len(trainloader))

        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=self.cfg.loss_smooth)

        early_stopping_steps = self.cfg.early_stopping_steps
        early_step = 0

        oof = np.zeros((len(train), self.target.iloc[:, 1:].shape[1]))
        best_loss = np.inf
        total_time = 0

        for epoch in range(self.cfg.epochs):
            starting_time = time.time()
            train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, self.cfg.device)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, self.cfg.device)

            if valid_loss < best_loss:

                best_loss = valid_loss
                oof[val_idx] = valid_preds
                if not os.path.exists(os.path.join(self.save_path, f"seed{seed}")):
                    os.mkdir(os.path.join(self.save_path, f"seed{seed}"))
                torch.save(model.state_dict(), os.path.join(self.save_path, f"seed{seed}", f"FOLD{fold}_.pth"))
                early_step = 0
                has_improved = True
                
            elif self.cfg.early_stop:

                early_step += 1
                has_improved = False
                if (early_step >= early_stopping_steps):
                    break
                
            total_time += time.time() - starting_time
            msg_epoch = f"FOLD: {fold}, EPOCH: {epoch}, "
            msg_epoch += f"train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, "
            msg_epoch += f"time: {np.round(total_time, 2):<10}"
            msg_epoch += f"Improved: {has_improved}"
            print (msg_epoch)

        # #--------------------- PREDICTION---------------------
        # x_test = test_[feature_cols].values
        # testdataset = TestDataset(x_test)
        # testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.cfg.batch_size, shuffle=False)

        # model = Model(
        #     num_features=len(feature_cols),
        #     num_targets=len(target_cols),
        #     hidden_size=self.cfg.hidden_size,
        # )

        # model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
        # model.to(self.cfg.device)

        # predictions = np.zeros((len(test_), self.target.iloc[:, 1:].shape[1]))
        # predictions = inference_fn(model, testloader, self.cfg.device)

        return oof

    def run_evaluate(self, fold, seed):
        test_ = process_data(self.test)
        feature_cols = [c for c in test_.columns if c not in ['sig_id']]
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.cfg.batch_size, shuffle=False)
        target_cols = self.target.drop('sig_id', axis=1).columns.values.tolist()

        model = Model_old(
            num_features=len(feature_cols),
            num_targets=len(target_cols),
            hidden_size=self.cfg.hidden_size,
        )

        """ model.load_state_dict(torch.load(os.path.join(
            self.load_path, f"seed{seed}", f"FOLD{fold}_.pth"), map_location=torch.device(self.cfg.device))) """
        model.load_state_dict(torch.load(os.path.join(
            self.load_path, f"SEED{seed}_FOLD{fold}_scored.pth"), map_location=torch.device(self.cfg.device)))
        model.to(self.cfg.device)

        predictions = np.zeros((len(test_), self.target.iloc[:, 1:].shape[1]))
        predictions = inference_fn(model, testloader, self.cfg.device)

        return predictions


    def run_k_fold(self, seed):
        oof = np.zeros((self.folds.shape[0], self.target.shape[1]-1))
        predictions = np.zeros((self.test.shape[0], self.target.shape[1]-1))

        for fold in range(self.cfg.nfolds):
            if (self.runty == 'train'):
                oof_ = self.run_training(fold, seed)
                oof += oof_
            elif (self.runty == 'eval'):
                pred_ = self.run_evaluate(fold, seed)
                predictions += pred_ / self.cfg.nfolds
            elif (self.runty == 'traineval'):
                oof_ = self.run_training(fold, seed)
                pred_ = self.run_evaluate(fold, seed)
                oof += oof_
                predictions += pred_ / self.cfg.nfolds
                

        if (self.runty == 'train'):
            return oof
        elif (self.runty == 'eval'):
            return predictions
        elif(self.runty == 'traineval'):
            return oof, predictions


# def run_training(fold, seed):

#     seed_everything(seed)

#     train = process_data(folds)
#     test_ = process_data(test)

#     trn_idx = train[train['kfold'] != fold].index
#     val_idx = train[train['kfold'] == fold].index

#     train_df = train[train['kfold'] != fold].reset_index(drop=True)
#     valid_df = train[train['kfold'] == fold].reset_index(drop=True)

#     x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
#     x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

#     train_dataset = MoADataset(x_train, y_train)
#     valid_dataset = MoADataset(x_valid, y_valid)
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     model = Model(
#         num_features=num_features,
#         num_targets=num_targets,
#         hidden_size=hidden_size,
#     )

#     model.to(DEVICE)

#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

#     loss_fn = nn.BCEWithLogitsLoss()
#     loss_tr = SmoothBCEwLogits(smoothing =0.001)

#     early_stopping_steps = EARLY_STOPPING_STEPS
#     early_step = 0

#     oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
#     best_loss = np.inf3

#     for epoch in range(EPOCHS):

#         train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
#         print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
#         valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
#         print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

#         if valid_loss < best_loss:

#             best_loss = valid_loss
#             oof[val_idx] = valid_preds
#             torch.save(model.state_dict(), f"FOLD{fold}_.pth")

#         elif(EARLY_STOP == True):

#             early_step += 1
#             if (early_step >= early_stopping_steps):
#                 break

#     #--------------------- PREDICTION---------------------
#     x_test = test_[feature_cols].values
#     testdataset = TestDataset(x_test)
#     testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

#     model = Model(
#         num_features=num_features,
#         num_targets=num_targets,
#         hidden_size=hidden_size,

#     )

#     model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
#     model.to(DEVICE)

#     predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
#     predictions = inference_fn(model, testloader, DEVICE)

#     return oof, predictions

# def run_k_fold(NFOLDS, seed):
#     oof = np.zeros((len(train), len(target_cols)))
#     predictions = np.zeros((len(test), len(target_cols)))

#     for fold in range(NFOLDS):
#         oof_, pred_ = run_training(fold, seed)

#         predictions += pred_ / NFOLDS
#         oof += oof_

#     return oof, predictions
