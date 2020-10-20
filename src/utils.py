import numpy as np
import random
import torch
import os

from sklearn.metrics import roc_auc_score

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

def inference_fn(model, X, verbose=True):
    with torch.no_grad():
        y_preds = model.predict(X)
        y_preds = torch.sigmoid(torch.as_tensor(y_preds)).numpy()

    return y_preds

def check_targets(targets):
    ### check if targets are all binary in training set

    for i in range(targets.shape[1]):
        if len(np.unique(targets[:,i])) != 2:
            return False
    return True

def auc_multi(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)
    for i in range(M):
        try:
            results[i] = roc_auc_score(y_true[:,i], y_pred[:,i])
        except:
            pass
    return results.mean()

