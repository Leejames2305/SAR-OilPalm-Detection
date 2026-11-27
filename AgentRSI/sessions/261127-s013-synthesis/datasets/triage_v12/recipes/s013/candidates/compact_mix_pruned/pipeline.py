"""Controlled synthesis: compact_mix_pruned."""
import sys
from pathlib import Path
sys.path.insert(0,"/app/research")
from synthesis import predict
CONFIG = {'mixcols': ['deblur_NDVI', 'deblur_NDRE', 'deblur_NDMI', 'deblur_NIRv', 'deblur_NDVI_contrast', 'deblur_NDRE_contrast', 'deblur_NDMI_contrast'], 'patches': ['sar']}
def fit_predict(labeled,unlabeled,ctx):
    return predict(labeled,unlabeled,ctx,CONFIG)
