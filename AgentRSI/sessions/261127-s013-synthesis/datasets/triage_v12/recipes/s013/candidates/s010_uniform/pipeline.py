"""Controlled synthesis: s010_uniform."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'pool': True, 'shared_cols': ['NDVI_contrast35', 'NDRE1_contrast35', 'ndvi_dip', 'ndvi_sd20', 'ndvi_edge', 'ndre_contrast'], 'quality_blend': True, 'optical': 'legacy', 'alpha_std': False, 'opt_scale': 0.3, 'alpha_scale': 0.3, 'height': False, 'mixture': False, 'patches': ['optical', 'sar'], 'C': 0.1}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
