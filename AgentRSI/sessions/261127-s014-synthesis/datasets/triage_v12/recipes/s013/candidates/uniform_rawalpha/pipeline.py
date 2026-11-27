"""Controlled synthesis: uniform_rawalpha."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'alpha_std': False}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
