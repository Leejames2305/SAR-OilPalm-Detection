"""Offline spatial oil-palm triage; risk scores are inspection priorities."""
from engine import predict
CONFIG = {'pool': True, 'shared_cols': ['qNDVI', 'qNDRE', 'qNDMI', 'qNIRv', 'log_green_red', 'NDVI_contrast35', 'NDRE1_contrast35', 'NDMI_contrast35', 'ndvi_dip', 'ndvi_sd20', 'ndvi_edge', 'ndre_contrast'], 'patches': ['sar']}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
