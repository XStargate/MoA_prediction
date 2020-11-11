import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


from pdb import set_trace

def log_loss_score(actual, predicted,  eps=1e-15):

    """
    :param predicted:   The predicted probabilities as floats between 0-1
    :param actual:      The binary labels. Either 0 or 1.
    :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
    :return:            The logarithmic loss between between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
    """

    p1 = actual * np.log(predicted+eps)
    p0 = (1-actual) * np.log(1-predicted+eps)
    loss = p0 + p1

    return -loss.mean()

def log_loss_multi(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)
    for i in range(M):
        results[i] = log_loss_score(y_true[:,i], y_pred[:,i])
    return results.mean()

def auc_multi(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)
    for i in range(M):
        try:
            results[i] = roc_auc_score(y_true[:,i], y_pred[:,i])
        except:
            pass
    return results.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # true_dist.scatter_(1, target.data.unsqueeze(1).type(torch.int64), self.confidence)
            true_dist[target.type(torch.int64) == 1] = self.confidence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# def log_loss_multi_smooth(target, pred, smooth=0.001, dim=-1):
#     cls = target.shape[1]
#     pred = pred.log_softmax(dim=dim)
#     with torch.no_grad():
#         true_dist = torch.zeros_like(pred)
#         true_dist.fill_(smooth / (cls - 1))
#         true_dist.scatter_(1, target.data.unsqueeze(1), 1.0-smooth)
#     return torch.mean(torch.sum(-true_dist * pred, dim=dim))

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    loss = LabelSmoothingLoss(classes=3, smoothing=0.01)
    pred = torch.tensor([[2.,2.,2.], [2.,2.,2.], [2.,2.,2.], [2.,2.,2.], [2.,2.,2.]])
    target = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]])
    loss.forward(pred, target)
