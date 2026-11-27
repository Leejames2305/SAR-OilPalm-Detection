"""Offline oil-palm triage with physical optical and historical canopy context.

Training uses equally trusted binary labels from the current estate. Compact
training-only AlphaEarth PCs supply prior-year context. Fixed optical mixture
covariates are label-free; historical height has a physical uncertainty floor.
Scores average32spatial-block bootstrap models for stable inspection ranking;
they are not calibrated disease probabilities.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

USE_HEIGHT = True
OPTICAL_VARIANT = 'deblur'
REGULARIZATION_C = .15

_KEYS=['estate','Long','Lat']
_OPTICAL=['qNDVI','qNDRE','qNDMI','qNIRv','log_green_red','NDVI_contrast35',
          'NDRE1_contrast35','NDMI_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
_HEIGHT=['height_mean6','height_sd6','height_contrast6']
_INDICES=['NDVI','NDRE','NDMI','NBR','NIRv','GRVI','NDVI_contrast','NDRE_contrast','NDMI_contrast']

@lru_cache(maxsize=1)
def _table(root):
    return pd.read_parquet(Path(root)/'triage_v12'/'features.parquet').set_index(_KEYS)

def _read(frame,root,columns):
    values=_table(str(root)).reindex(pd.MultiIndex.from_frame(frame[_KEYS]))[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite prepared covariates')
    return values

def _scaled(a,b):
    scaler=StandardScaler().fit(a)
    return scaler.transform(a),scaler.transform(b)

def _design(train,target,root):
    lat0,lon0=float(train.Lat.mean()),float(train.Long.mean())
    def base(frame):
        return np.column_stack([frame.NDVI.to_numpy(dtype=float),
                (frame.Long.to_numpy(dtype=float)-lon0)*111320*np.cos(np.deg2rad(lat0)),
                (frame.Lat.to_numpy(dtype=float)-lat0)*110540])
    a,b=_scaled(base(train),base(target))
    poly=PolynomialFeatures(2,include_bias=False)
    a,b=poly.fit_transform(a),poly.transform(b)
    a[:,3:]*=.3;b[:,3:]*=.3
    oa,ob=_scaled(_read(train,root,_OPTICAL),_read(target,root,_OPTICAL))
    cols=[f'c2025_A{i:02d}' for i in range(64)]
    ea,eb=_scaled(_read(train,root,cols),_read(target,root,cols))
    pca=PCA(n_components=min(8,len(train)-1),svd_solver='full').fit(ea)
    ea,eb=_scaled(pca.transform(ea),pca.transform(eb))
    a,b=np.column_stack([a,.5*oa,.5*ea]),np.column_stack([b,.5*ob,.5*eb])
    if USE_HEIGHT:
        ha,hb=_read(train,root,_HEIGHT),_read(target,root,_HEIGHT)
        mean=ha.mean(axis=0)
        denom=np.sqrt(ha.var(axis=0)+np.array([3.,1.,1.])**2)
        a,b=np.column_stack([a,.35*(ha-mean)/denom]),np.column_stack([b,.35*(hb-mean)/denom])
    if OPTICAL_VARIANT:
        cols=[f'{OPTICAL_VARIANT}_{c}' for c in _INDICES]
        ma,mb=_scaled(_read(train,root,cols),_read(target,root,cols))
        a,b=np.column_stack([a,.3*ma]),np.column_stack([b,.3*mb])
    return a,b

def _fit_predict(labeled,unlabeled,ctx):
    if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
    train=labeled[labeled.label.isin(['Healthy','Unhealthy'])].copy()
    y=(train.label=='Unhealthy').to_numpy(dtype=int)
    if len(np.unique(y))<2:
        return np.full(len(unlabeled),float(y.mean()) if len(y) else 0.,dtype=np.float64)
    a,b=_design(train,unlabeled,ctx['datasets_dir'])
    weights=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
    # Resample complete80m blocks, preserving within-block label dependence.
    # Shared training-only scaler/PCA design; per-draw class weights normalized
    # to total effective mass len(train), keeping regularization comparable.
    east=(train.Long.to_numpy(dtype=float)-float(train.Long.min()))*111320*np.cos(np.deg2rad(float(train.Lat.mean())))
    north=(train.Lat.to_numpy(dtype=float)-float(train.Lat.mean()))*110540
    north-=north.min()
    group=np.floor(east/80).astype(np.int64)*100000+np.floor(north/80).astype(np.int64)
    unique,inverse=np.unique(group,return_inverse=True)
    rng=np.random.default_rng(20261127)
    predictions=[]
    for _ in range(32):
        count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inverse]
        positive_mass=float((count*y).sum())
        negative_mass=float((count*(1-y)).sum())
        if min(positive_mass,negative_mass)==0:continue
        sample_weights=count*np.where(y==1,len(y)/(2*positive_mass),len(y)/(2*negative_mass))
        model=LogisticRegression(C=REGULARIZATION_C,solver='lbfgs',max_iter=2000,tol=1e-7)
        model.fit(a,y,sample_weight=sample_weights)
        predictions.append(expit(model.decision_function(b)))
    if not predictions:
        return np.full(len(unlabeled),float(y.mean()),dtype=np.float64)
    # Sigmoid values are used for stable model averaging, not clinical
    # probability calibration: all fitting likelihoods are class-balanced.
    return np.asarray(np.mean(predictions,axis=0),dtype=np.float64)

def fit_predict(labeled:pd.DataFrame,unlabeled:pd.DataFrame,ctx:dict)->np.ndarray:
    with threadpool_limits(limits=max(1,int(ctx['cpus']))):
        return _fit_predict(labeled,unlabeled,ctx)
