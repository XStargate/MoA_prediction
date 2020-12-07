import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataloader import MoADataset, TestDataset
from model import Model, FineTuneScheduler
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
    def __init__(self, folds, test, targets_scored, targets_nonscored, save_path, load_path, runty):
        self.folds = folds
        self.test = test
        self.targets_scored = targets_scored
        self.targets_nonscored = targets_nonscored
        self.save_path = save_path
        self.load_path = load_path
        
        assert runty == 'traineval' or runty == 'eval',  \
            "Run type is wrong. Should be 'traineval' or 'eval'"
        
        self.runty = runty
        self.cfg = Config()

    def _run_training(self, fold, seed):

        seed_everything(seed)

        train = process_data(self.folds)
        test_ = process_data(self.test)

        kfold_col = f'kfold_{seed}'
        trn_idx = train[train[kfold_col] != fold].index
        val_idx = train[train[kfold_col] == fold].index

        train_df = train[train[kfold_col] != fold].reset_index(drop=True)
        valid_df = train[train[kfold_col] == fold].reset_index(drop=True)
        
        target_cols = self.targets_scored.drop('sig_id', axis=1).columns.values.tolist()
        aux_target_cols = [x for x in self.targets_nonscored.columns if x != 'sig_id']
        all_target_cols = target_cols + aux_target_cols
        
        num_targets = len(target_cols)
        num_aux_targets = len(aux_target_cols)
        num_all_targets = len(all_target_cols)

        feature_cols = [c for c in process_data(self.folds).columns if c not in all_target_cols]
        feature_cols = [c for c in feature_cols if (str(c)[0:5] != 'kfold' and c not in ['sig_id', 'drug_id'])]
        num_features = len(feature_cols)
        
        def train_model(model, tag_name, target_cols_now, fine_tune_scheduler=None):
            x_train, y_train  = train_df[feature_cols].values, train_df[target_cols_now].values
            x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols_now].values
        
            train_dataset = MoADataset(x_train, y_train)
            valid_dataset = MoADataset(x_valid, y_valid)
            
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=self.cfg.weight_decay[tag_name])
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                      steps_per_epoch=len(trainloader),
                                                      pct_start=self.cfg.pct_start,
                                                      div_factor=self.cfg.div_factor[tag_name], 
                                                      max_lr=self.cfg.max_lr[tag_name],
                                                      epochs=self.cfg.epochs)
        
            loss_fn = nn.BCEWithLogitsLoss()
            loss_tr = SmoothBCEwLogits(smoothing=self.cfg.loss_smooth)

            oof = np.zeros((len(train), len(target_cols_now)))
            best_loss = np.inf
            
            for epoch in range(self.cfg.epochs):
                if fine_tune_scheduler is not None:
                    fine_tune_scheduler.step(epoch, model)

                train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, self.cfg.device)
                valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, self.cfg.device)
                print(f"SEED: {seed}, FOLD: {fold}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")

                if np.isnan(valid_loss):
                    break
            
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    oof[val_idx] = valid_preds
                    if not os.path.exists(os.path.join(self.save_path, f"seed{seed}")):
                        os.mkdir(os.path.join(self.save_path, f"seed{seed}"))
                    torch.save(model.state_dict(), 
                               os.path.join(self.save_path, f"seed{seed}", f"{tag_name}_FOLD{fold}_.pth"))

            return oof
            
        fine_tune_scheduler = FineTuneScheduler(self.cfg.epochs)

        pretrained_model = Model(num_features, num_all_targets)
        pretrained_model.to(self.cfg.device)
        
        # Train on scored + nonscored targets
        train_model(pretrained_model, 'ALL_TARGETS', all_target_cols)

        # Load the pretrained model with the best loss
        pretrained_model = Model(num_features, num_all_targets)
        pretrained_model.load_state_dict(torch.load(os.path.join(
            self.load_path, f"seed{seed}", f"ALL_TARGETS_FOLD{fold}_.pth"),
                                                    map_location=torch.device(self.cfg.device)))
        pretrained_model.to(self.cfg.device)
        
        # Copy model without the top layer
        final_model = fine_tune_scheduler.copy_without_top(pretrained_model, num_features, 
                                                           num_all_targets, num_targets)

        # Fine-tune the model on scored targets only
        oof = train_model(final_model, 'SCORED_ONLY', target_cols, fine_tune_scheduler)

        # Load the fine-tuned model with the best loss
        model = Model(num_features, num_targets)
        model.load_state_dict(torch.load(os.path.join(
            self.load_path, f"seed{seed}", f"SCORED_ONLY_FOLD{fold}_.pth"), 
                                         map_location=torch.device(self.cfg.device)))
        # model.load_state_dict(torch.load(f"SCORED_ONLY_FOLD{fold}_.pth"))
        model.to(self.cfg.device)

        #--------------------- PREDICTION---------------------
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.cfg.batch_size, shuffle=False)
    
        predictions = np.zeros((len(test_), num_targets))
        predictions = inference_fn(model, testloader, self.cfg.device)
        return oof, predictions
    
    def run_evaluate(self, fold, seed):
        test_ = process_data(self.test)
        feature_cols = [c for c in test_.columns if c not in ['sig_id']]
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.cfg.batch_size, shuffle=False)
        target_cols = self.targets_scored.drop('sig_id', axis=1).columns.values.tolist()
        
        model = Model(len(feature_cols), len(target_cols))
        model.load_state_dict(torch.load(os.path.join(
            self.load_path, f"seed{seed}", f"SCORED_ONLY_FOLD{fold}_.pth"), 
                                         map_location=torch.device(self.cfg.device)))
        model.to(self.cfg.device)
        
        predictions = np.zeros((len(test_), len(target_cols)))
        predictions = inference_fn(model, testloader, self.cfg.device)
        
        return predictions

    def run_k_fold(self, seed):
        oof = np.zeros((len(self.folds), self.targets_scored.shape[1]-1))
        predictions = np.zeros((len(self.test), self.targets_scored.shape[1]-1))

        for fold in range(self.cfg.nfolds):
            if (self.runty == 'traineval'):
                oof_, pred_ = self._run_training(fold, seed)
                predictions += pred_ / self.cfg.nfolds
                oof += oof_
            elif (self.runty == 'eval'):
                pred_ = self.run_evaluate(fold, seed)
                predictions += pred_ / self.cfg.nfolds

        if (self.runty == 'traineval'):
            return oof, predictions
        elif (self.runty == 'eval'):
            return predictions
