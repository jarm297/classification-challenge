"""
In this module we store functions to measuer the performance of our model.

"""
import numpy as np
from sklearn.metrics import f1_score, make_scorer


def get_metric_name_mapping():
    return {_f1(): f1_score}

def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn

def get_scoring_function(name: str, **params):
    mapping = {
        _f1(): make_scorer(f1_score, greater_is_better=False, **params),
    }
    return mapping[name]

def _f1():
    return "f1 score"
