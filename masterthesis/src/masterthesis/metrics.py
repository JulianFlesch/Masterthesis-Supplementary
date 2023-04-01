from sklearn.metrics import log_loss, accuracy_score


def cross_entropy_loss(*args, **kwargs):
    log_loss(*args, **kwargs)


def class_error(*args, **kwargs):
    accuracy_score(*args, **kwargs)
