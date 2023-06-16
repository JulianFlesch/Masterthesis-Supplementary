from sklearn.metrics import log_loss, accuracy_score
import numpy as np


def abs_delta(y_pred, y, mean=False):
    if len(y_pred) != len(y):
        print("[Warning] inputs of different length:", len(y_pred), len(y))
        return -1
    
    abs_diff = np.abs(y_pred - y)
    if mean:
        return np.mean(abs_diff)
    else:
        return np.sum(abs_diff)


def cross_entropy_loss(*args, **kwargs):
    return log_loss(*args, **kwargs)


def class_error(*args, **kwargs):
    return accuracy_score(*args, **kwargs)
