"""Controlled synthesis: both_bag."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'patches': ['optical', 'sar'], 'bag': 32}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
