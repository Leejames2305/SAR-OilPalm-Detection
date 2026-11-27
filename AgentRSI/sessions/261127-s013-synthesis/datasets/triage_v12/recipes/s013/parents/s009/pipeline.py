"""Offline estate-specific triage with registration-integrated optical texture.

Larger scores rank trees for inspection; they are not disease probabilities.
All label-fitted residual features use block-disjoint inner validation.
"""
import numpy as np
from threadpoolctl import threadpool_limits
from texture_jitter_helper import residual

def fit_predict(labeled,unlabeled,ctx):
    with threadpool_limits(limits=max(1,int(ctx['cpus']))):
        return np.asarray(residual(labeled,unlabeled,ctx,kind='texture_small'),dtype=np.float64)
