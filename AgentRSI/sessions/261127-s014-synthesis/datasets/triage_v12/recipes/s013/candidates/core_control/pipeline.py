"""Controlled synthesis: core_control."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'height': False, 'mixture': False}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
