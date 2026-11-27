"""Reconstruction of s008 optical recipe; not the original source."""
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from scipy.special import expit

KEY=['estate','Long','Lat']
@lru_cache(maxsize=12)
def table(root,name):
 return pd.read_parquet(Path(root)/name/'features.parquet').set_index(KEY)

def cov(d,ctx,name,cols):
 return table(str(ctx['datasets_dir']),name).reindex(pd.MultiIndex.from_frame(d[KEY]))[cols].to_numpy(float)

def groups(d):
 x=(d.Long-d.Long.min())*111320*np.cos(np.deg2rad(d.Lat.mean()));y=(d.Lat-d.Lat.min())*110540
 return (np.floor(x/80)*100000+np.floor(y/80)).to_numpy(np.int64)

def base(labeled,unlabeled,ctx,quality=True,degree=None):
 d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].copy();y=(d.label=='Unhealthy').to_numpy(int)
 if len(np.unique(y))<2:return np.full(len(unlabeled),0.)
 lon=d.Long.mean();lat=d.Lat.mean()
 def f(a):
  nd=a.NDVI.to_numpy(float)
  if quality:nd=.5*nd+.5*cov(a,ctx,'gee_quality',['qNDVI_p75'])[:,0]
  return np.c_[nd,(a.Long-lon)*111320*np.cos(np.deg2rad(lat)),(a.Lat-lat)*110540]
 deg=degree if degree is not None else (1 if ctx.get('estate')=='AirHitam' else 2)
 m=make_pipeline(StandardScaler(),*([PolynomialFeatures(2,include_bias=False)] if deg==2 else []),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000))
 m.fit(f(d),y);return m.decision_function(f(unlabeled))

def contrast(d,ctx):return cov(d,ctx,'gee_canopy',['NDRE1_contrast35'])

SOURCE_CACHE={}
def transfer_coef(ctx):
 key=(str(ctx['datasets_dir']),tuple((e,len(d),int(pd.util.hash_pandas_object(d,index=True).sum())) for e,d in sorted(ctx['other_labeled'].items())))
 if key in SOURCE_CACHE:return SOURCE_CACHE[key]
 zs=[];ys=[];bs=[]
 for e,d in sorted(ctx['other_labeled'].items()):
  d=d.reset_index(drop=True);c=dict(ctx,estate=e)
  z=contrast(d,c);z=(z-np.mean(z,axis=0))/np.maximum(z.std(axis=0),1e-8);z=np.clip(z,-3,3)
  off=np.zeros(len(d))
  for tr,va in GroupKFold(5).split(d,groups=groups(d)):
   off[va]=base(d.iloc[tr],d.iloc[va],c)
  ok=d.label.isin(['Healthy','Unhealthy']).to_numpy();zs.append(z[ok]);ys.append((d.label.to_numpy()[ok]=='Unhealthy').astype(float));bs.append(off[ok])
 z=np.vstack(zs);y=np.concatenate(ys);b=np.concatenate(bs)
 w=np.where(y==1,.5/y.mean(),.5/(1-y.mean()));w/=w.sum()
 def objective(a):
  s=b+z@a;loss=np.sum(w*(np.logaddexp(0,s)-y*s))+.05*np.sum(a*a)
  return loss,z.T@(w*(expit(s)-y))+.1*a
 coef=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B').x
 SOURCE_CACHE[key]=coef;return coef

def fit_predict(labeled,unlabeled,ctx):
 s=base(labeled,unlabeled,ctx);z=contrast(labeled,ctx);zt=contrast(unlabeled,ctx)
 zt=np.clip((zt-z.mean(axis=0))/np.maximum(z.std(axis=0),1e-8),-3,3)
 a=transfer_coef(ctx)
 return np.asarray(s+np.minimum((zt@a),0),dtype=np.float64)
