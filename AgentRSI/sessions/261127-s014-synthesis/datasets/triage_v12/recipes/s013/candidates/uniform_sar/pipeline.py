"""Controlled synthesis: uniform_sar."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'patches': ['sar']}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
