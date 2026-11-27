"""Balanced ridge residual fitting with block-honest optical offsets."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import optical

def offset_fit(x,y,offset,lam=.1):
 w=np.where(y==1,.5/max(y.mean(),1e-6),.5/max(1-y.mean(),1e-6));w/=w.sum()
 def obj(a):
  s=offset+x@a
  return np.sum(w*(np.logaddexp(0,s)-y*s))+lam*np.sum(a*a)/2,x.T@(w*(expit(s)-y))+lam*a
 return minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B').x

def oof_base(d,ctx):
 out=np.zeros(len(d));g=optical.groups(d)
 for tr,va in GroupKFold(5).split(d,groups=g):out[va]=optical.fit_predict(d.iloc[tr],d.iloc[va],ctx)
 return out
