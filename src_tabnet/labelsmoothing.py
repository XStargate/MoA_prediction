import numpy as np


def labelsmoothing(y_true, label_smoothing=0.0):
    for index in y_true.index:
        row = y_true.iloc[index, 1:]
        row_smooth = row * (1.0 - label_smoothing) + 0.5*label_smoothing
        y_true.iloc[index, 1:] = row_smooth

    return y_true


def ls_manual(y_train, ls_rate=0.0):
    print("Label Smoothing")
    print(y_train.shape)

    cols = [c for c in y_train.columns if c != "sig_id"]
    tmp = y_train[cols].values
    tmp = tmp * (1.0 - ls_rate) + ls_rate/len(cols)

    y_train.loc[:, cols] = tmp
    print(y_train.shape)

    return y_train
