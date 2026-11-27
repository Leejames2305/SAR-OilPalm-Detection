"""Controlled synthesis: uniform_blend."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'quality_blend': True}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
