"""Spatially honest incumbent plus data-driven Sentinel-2 feature blend."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

S2=['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

def _blocks(d):
    lat=float(d['Lat'].mean()); x=(d['Long'].to_numpy(float)-float(d['Long'].min()))*111320*np.cos(np.deg2rad(lat)); y=(d['Lat'].to_numpy(float)-float(d['Lat'].min()))*110540
    return np.floor(x/80).astype(np.int64)*100000+np.floor(y/80).astype(np.int64)

def _coords(d, ref):
    lat0=float(ref['Lat'].mean()); lon0=float(ref['Long'].mean())
    return np.column_stack(((d['Long'].to_numpy(float)-lon0)*111320*np.cos(np.deg2rad(lat0)), (d['Lat'].to_numpy(float)-lat0)*110540.0))

def _fit_score(Xt, y, Xv, degree, C):
    med=np.nanmedian(Xt,axis=0); Xt=np.where(np.isfinite(Xt),Xt,med); Xv=np.where(np.isfinite(Xv),Xv,med)
    m=make_pipeline(StandardScaler(), *( [PolynomialFeatures(degree=2,include_bias=False)] if degree==2 else []), LogisticRegression(C=C,class_weight='balanced',max_iter=1800))
    m.fit(Xt,y); return m.decision_function(Xv),m

def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    tr=labeled[labeled['label'].isin(['Healthy','Unhealthy'])].copy().reset_index(drop=True)
    y=(tr['label']=='Unhealthy').astype(int).to_numpy()
    ds=Path(ctx['datasets_dir'])/'gee_s2_stress'/'features.parquet'
    ext=pd.read_parquet(ds,columns=['estate','Long','Lat']+S2)
    tr=tr.merge(ext,on=['estate','Long','Lat'],how='left',suffixes=('','_e')); te=unlabeled.merge(ext,on=['estate','Long','Lat'],how='left',suffixes=('','_e'))
    # Feature name collisions in the source table are resolved to the GEE suffix.
    fs=[c+'_e' if c in ('B4','B8') else c for c in S2]
    # incumbent: AirHitam is linear; the other estates use degree 2 as in the verified recipe.
    deg0=1 if ctx.get('estate')=='AirHitam' else 2
    X0tr=np.c_[tr['NDVI'].to_numpy(float),_coords(tr,tr)]
    X0te=np.c_[te['NDVI'].to_numpy(float),_coords(te,tr)]
    X1tr=np.c_[tr[['NDVI']+fs].to_numpy(float),_coords(tr,tr)]
    X1te=np.c_[te[['NDVI']+fs].to_numpy(float),_coords(te,tr)]
    alpha=0.1
    s0,m0=_fit_score(X0tr,y,X0te,deg0,.1); s1,m1=_fit_score(X1tr,y,X1te,2,.03)
    out=(1-alpha)*s0+alpha*s1
    return np.asarray(out,dtype=np.float64)
