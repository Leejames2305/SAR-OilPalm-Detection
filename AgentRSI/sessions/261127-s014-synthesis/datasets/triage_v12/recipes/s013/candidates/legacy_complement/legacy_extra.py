"""Complement missing condition measurements around the uniform s010 core."""
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import legacy_core

KEY=['estate','Long','Lat']
QUALITY=['qNDRE','qNDMI','qNIRv','NDMI_contrast35']
HEIGHT=['height_mean6','height_sd6','height_contrast6']
MIX=['deblur_'+v for v in ['NDVI','NDRE','NDMI','NBR','NIRv','GRVI','NDVI_contrast','NDRE_contrast','NDMI_contrast']]
@lru_cache(maxsize=1)
def table(root):return pd.read_parquet(Path(root)/'triage_v12/features.parquet').set_index(KEY)
def read(d,ctx,cols):
    v=table(str(ctx['datasets_dir'])).reindex(pd.MultiIndex.from_frame(d[KEY]))[cols].to_numpy(float)
    if not np.isfinite(v).all():raise ValueError('Missing compact covariates')
    return v
def features(d,t,ctx,kind):
    cols=QUALITY if kind=='quality' else HEIGHT+MIX if kind=='structure' else QUALITY+HEIGHT+MIX
    a,b=read(d,ctx,cols),read(t,ctx,cols)
    den=np.maximum(a.std(0),1e-8)
    for j,col in enumerate(cols):
        if col in HEIGHT:den[j]=np.sqrt(den[j]**2+np.array([3.,1.,1.])[HEIGHT.index(col)]**2)
    return np.clip((a-a.mean(0))/den,-3,3),np.clip((b-a.mean(0))/den,-3,3)

def fit_predict(labeled,unlabeled,ctx,kind='all',offset=False):
    if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
    d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy(int)
    if len(np.unique(y))<2:return np.full(len(unlabeled),y.mean() if len(y) else 0.,dtype=np.float64)
    with threadpool_limits(limits=max(1,int(ctx['cpus']))):
        if not offset:
            context=legacy_core._context
            def augmented(a,b,c,radar=False):
                x,z=context(a,b,c,radar=radar);extra_x,extra_z=features(a,b,c,kind)
                return np.c_[x,extra_x],np.c_[z,extra_z]
            legacy_core._context=augmented
            try:return legacy_core.fit_predict(labeled,unlabeled,ctx)
            finally:legacy_core._context=context
        s=legacy_core.fit_predict(labeled,unlabeled,ctx)
        east=(d.Long-d.Long.min()).to_numpy()*111320*np.cos(np.deg2rad(d.Lat.mean()));north=(d.Lat-d.Lat.min()).to_numpy()*110540
        group=np.floor(east/80).astype(np.int64)*100000+np.floor(north/80).astype(np.int64)
        if len(np.unique(group))<2:return s
        o=np.zeros(len(d))
        for tr,va in GroupKFold(min(5,len(np.unique(group)))).split(d,groups=group):
            train=d.iloc[tr];yy=(train.label=='Unhealthy').to_numpy(int)
            if len(np.unique(yy))<2:o[va]=0.
            else:o[va]=legacy_core.fit_predict(train,d.iloc[va],ctx)
        x,z=features(d,unlabeled,ctx,kind);w=np.where(y==1,.5/y.mean(),.5/(1-y.mean()));w/=w.sum()
        def objective(beta):
            f=o+x@beta
            return np.sum(w*(np.logaddexp(0,f)-y*f))+.25*(beta@beta),x.T@(w*(expit(f)-y))+.5*beta
        beta=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B').x
        return np.asarray(s+.3*np.clip(z@beta,-1,1),dtype=np.float64)
