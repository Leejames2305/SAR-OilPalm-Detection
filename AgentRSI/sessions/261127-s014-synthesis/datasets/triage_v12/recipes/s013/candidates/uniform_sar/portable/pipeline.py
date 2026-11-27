"""Offline spatial oil-palm triage; risk scores are inspection priorities."""
from engine import predict
CONFIG = {'patches': ['sar']}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
