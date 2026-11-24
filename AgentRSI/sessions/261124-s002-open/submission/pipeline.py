from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def fit_predict(labeled, unlabeled, ctx):
    tr=labeled[labeled['label'].isin(['Healthy','Unhealthy'])].copy()
    y=(tr['label']=='Unhealthy').astype(int).to_numpy()
    ds=Path(ctx['datasets_dir'])/'alos_winstats'/'features.parquet'
    feat=pd.read_parquet(ds)
    keys=['estate','Long','Lat']
    # Only use extracted fields with support in this estate and remove unstable
    # high-order summaries; the r1/r2 means and robust quantiles are compact.
    cols=[c for c in feat.columns if c.startswith('rs_') and ('_r1_' in c or '_r2_' in c)]
    # omit valid-count columns and raw minima/maxima at r2 to control capacity
    cols=[c for c in cols if not c.endswith('_n')]
    mtr=tr[keys].merge(feat[keys+cols],on=keys,how='left',sort=False)
    mva=unlabeled[keys].merge(feat[keys+cols],on=keys,how='left',sort=False)
    # finite columns only; missing dates/families are imputed per fold
    Xtr=mtr[cols].to_numpy(float); Xva=mva[cols].to_numpy(float)
    # retain compact spatial context and NDVI
    lat0=float(tr.Lat.mean()); lon0=float(tr.Long.mean()); mpl=111320*np.cos(np.deg2rad(lat0))
    def xy(d): return np.c_[(d.Long.to_numpy(float)-lon0)*mpl,(d.Lat.to_numpy(float)-lat0)*110540,d.NDVI.to_numpy(float)]
    Xtr=np.c_[xy(tr),Xtr]; Xva=np.c_[xy(unlabeled),Xva]
    degree=2 if ctx.get('estate')!='AirHitam' else 1
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),*( [PolynomialFeatures(degree=degree,include_bias=False)] if degree==2 else []),LogisticRegression(C=0.01,class_weight='balanced',max_iter=3000))
    model.fit(Xtr,y)
    return np.asarray(model.decision_function(Xva),dtype=np.float64)
