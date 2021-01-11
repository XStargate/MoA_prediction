import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataloader import MoADataset, TestDataset
from model import Model
from loss import SmoothBCEwLogits
from utils import seed_everything, process_data

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

    preds = np.concatenate(preds)

    return preds

class train_test():
    def __init__(self, folds, test, target):
        self.folds = folds
        self.test = test
        self.target = target

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

        model = Model(
            num_features=len(feature_cols),
            num_targets=len(target_cols),
            hidden_size=self.cfg.hidden_size,
        )

        model.to(self.cfg.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                                  max_lr=1e-2, epochs=self.cfg.epochs,
                                                  steps_per_epoch=len(trainloader))

        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=self.cfg.loss_smooth)

        early_stopping_steps = self.cfg.early_stopping_steps
        early_step = 0

        oof = np.zeros((len(train), self.target.iloc[:, 1:].shape[1]))
        best_loss = np.inf

        for epoch in range(self.cfg.epochs):

            train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, self.cfg.device)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, self.cfg.device)
            print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

            if valid_loss < best_loss:

                best_loss = valid_loss
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"FOLD{fold}_.pth")

            elif self.cfg.early_stop:

                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

        #--------------------- PREDICTION---------------------
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.cfg.batch_size, shuffle=False)

        model = Model(
            num_features=len(feature_cols),
            num_targets=len(target_cols),
            hidden_size=self.cfg.hidden_size,
        )

        model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
        model.to(self.cfg.device)

        predictions = np.zeros((len(test_), self.target.iloc[:, 1:].shape[1]))
        predictions = inference_fn(model, testloader, self.cfg.device)

        return oof, predictions

    def run_k_fold(self, seed):
        oof = np.zeros((self.folds.shape[0], self.target.shape[1]-1))
        predictions = np.zeros((self.test.shape[0], self.target.shape[1]-1))

        for fold in range(self.cfg.nfolds):
            oof_, pred_ = self.run_training(fold, seed)
            predictions += pred_ / self.cfg.nfolds
            oof += oof_

        return oof, predictions
