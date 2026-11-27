"""Controlled synthesis: pooled_sar_nocoords."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'pool': True, 'shared_cols': ['qNDVI', 'qNDRE', 'qNDMI', 'qNIRv', 'log_green_red', 'NDVI_contrast35', 'NDRE1_contrast35', 'NDMI_contrast35', 'ndvi_dip', 'ndvi_sd20', 'ndvi_edge', 'ndre_contrast'], 'patches': ['sar'], 'no_coords': True}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
