import numpy as np
from sklearn.metrics import roc_auc_score

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
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
