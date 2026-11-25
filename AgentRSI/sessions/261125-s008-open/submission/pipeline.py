"""Quality-controlled NDVI baseline plus source-estate spectral stress offsets."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.special import expit
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
CACHE={};STRENGTH=1.;MECHANISM='local';DEGREE='incumbent';OPTICAL_BLEND=.5

def table(d,ctx,source):
 p=str(Path(ctx['datasets_dir'])/source/'features.parquet')
 if p not in CACHE:CACHE[p]=pd.read_parquet(p)
 return d[['estate','Long','Lat']].merge(CACHE[p],on=['estate','Long','Lat'],how='left',sort=False,validate='many_to_one')
def blocks(d):
 x=(d.Long.to_numpy()-d.Long.min())*111320*np.cos(np.radians(d.Lat.mean()));y=(d.Lat.to_numpy()-d.Lat.min())*110540
 return np.floor(x/80).astype('int64')*100000+np.floor(y/80).astype('int64')
def prep(d,ctx):
 a=table(d,ctx,'gee_quality');r=d.copy();r['NDVI']=(1-OPTICAL_BLEND)*r.NDVI.to_numpy()+OPTICAL_BLEND*a.qNDVI_p75.to_numpy();return r

def base(d,t):
 y=(d.label=='Unhealthy').to_numpy(int);sc=StandardScaler();x=sc.fit_transform(d[['NDVI','Long','Lat']]);z=sc.transform(t[['NDVI','Long','Lat']])
 degree=(1 if d.estate.iloc[0]=='AirHitam' else 2) if DEGREE=='incumbent' else DEGREE
 if degree==2:po=PolynomialFeatures(2,include_bias=False);x=po.fit_transform(x);z=po.transform(z)
 m=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000);m.fit(x,y);return m.decision_function(z)
def feature(d,ctx):
 if MECHANISM=='local':return table(d,ctx,'gee_canopy')[['NDRE1_contrast35']].to_numpy()
 if MECHANISM=='quality':return table(d,ctx,'gee_quality')[['qNDRE']].to_numpy()
 if MECHANISM=='both':return np.c_[table(d,ctx,'gee_canopy')[['NDRE1_contrast35']],table(d,ctx,'gee_quality')[['qNDRE']]]
 if MECHANISM=='none':return np.zeros((len(d),1))
 raise ValueError(MECHANISM)

def fit_predict(labeled,unlabeled,ctx):
 with threadpool_limits(limits=1):return run(labeled,unlabeled,ctx)
def run(labeled,unlabeled,ctx):
 pool={ctx['estate']:labeled,**ctx['other_labeled']};parts=[];offset=[];ys=[];weights=[];targetx=None;targetscore=None
 for e,d in pool.items():
  d=d[d.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy(int);z=feature(d,ctx);sc=StandardScaler();z=np.clip(sc.fit_transform(z),-3.,3.);d=prep(d,ctx)
  if e==ctx['estate']:
   targetx=np.clip(sc.transform(feature(unlabeled,ctx)),-3.,3.);targetscore=base(d,prep(unlabeled,ctx));continue
  o=np.zeros(len(d));g=blocks(d)
  for tr,va in GroupKFold(min(4,len(np.unique(g)))).split(d,groups=g):o[va]=base(d.iloc[tr],d.iloc[va])
  parts.append(z);offset.append(o);ys.append(y);weights.append(np.where(y==1,len(d)/(2*y.sum()),len(d)/(2*(len(d)-y.sum()))))
 x=np.concatenate(parts);y=np.concatenate(ys);o=np.concatenate(offset);w=np.concatenate(weights);q=x.shape[1];estatecol=np.concatenate([np.full(len(p),i) for i,p in enumerate(parts)]);X=np.c_[x,np.eye(len(parts))[estatecol]]
 def fun(beta):
  s=o+X@beta;reg=np.r_[np.full(q,25.),np.full(len(parts),1.)];return np.dot(w,np.logaddexp(0,s)-y*s)+.5*np.dot(reg,beta**2),X.T@(w*(expit(s)-y))+reg*beta
 r=minimize(fun,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B');return targetscore+STRENGTH*np.minimum(targetx@r.x[:q],0.)
