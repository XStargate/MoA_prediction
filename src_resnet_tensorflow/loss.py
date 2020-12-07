import tensorflow as tf
from tensorflow.keras import backend

from config import Config

def logloss(y_true, y_pred, p_min=0, p_max=1):
    cfg = Config()
    p_min = cfg.label_smoothing
    p_max = 1 - p_min
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))