"""Compact source-prior ridge research engine; all transforms fitted on training rows."""
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
KEYS=['estate','Long','Lat']
OPT=['qNDVI','qNDRE','qNDMI','qNIRv','log_green_red','NDVI_contrast35','NDRE1_contrast35','NDMI_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
PHYS=['qNDVI','qNDRE','NDVI_contrast35','NDRE1_contrast35','ndvi_dip','ndre_contrast']
HEIGHT=['height_mean6','height_sd6','height_contrast6']
MIX=['deblur_'+n for n in ['NDVI','NDRE','NDMI','NBR','NIRv','GRVI','NDVI_contrast','NDRE_contrast','NDMI_contrast']]
DEFAULT=dict(C=.15,clip=None,poly='full',poly_weight=.3,opt_weight=.5,context_weight=1.,pool='all',pool_strength=1.,ae_weight=.5,height_weight=.35,mix_weight=.3,sar_weight=.3,guard=None)
@lru_cache(maxsize=12)
def table(root,name):return pd.read_parquet(Path(root)/name/'features.parquet').set_index(KEYS)
def read(d,root,cols,name='triage_v12'):
 t=table(str(root),name)
 if isinstance(cols,str):cols=[c for c in t if c.startswith(cols)]
 a=t.reindex(pd.MultiIndex.from_frame(d[KEYS]))[cols].to_numpy(float)
 if not np.isfinite(a).all():raise ValueError('Missing feature key')
 return a

def scaled(a,b,clip=None):
 s=StandardScaler().fit(a);x=s.transform(a);z=s.transform(b)
 if clip is not None:x=np.clip(x,-clip,clip);z=np.clip(z,-clip,clip)
 return x,z

def pcs(a,b,standardize=True,clip=None):
 if standardize:a,b=scaled(a,b)
 p=PCA(n_components=min(8,len(a)-1,a.shape[1]),svd_solver='full').fit(a)
 return scaled(p.transform(a),p.transform(b),clip)

def design(train,target,root,cfg):
 c={**DEFAULT,**cfg};lat=float(train.Lat.mean());lon=float(train.Long.mean())
 def base(d):return np.c_[d.NDVI.to_numpy(float),(d.Long-lon)*111320*np.cos(np.deg2rad(lat)),(d.Lat-lat)*110540]
 a,b=scaled(base(train),base(target),c['clip']);x=a;z=b
 if c['poly']!='none':
  p=PolynomialFeatures(2,include_bias=False);x=p.fit_transform(a);z=p.transform(b)
  x[:,3:]*=c['poly_weight'];z[:,3:]*=c['poly_weight']
  if c['poly']=='spatial':x=x[:,[0,1,2,6,7,8]];z=z[:,[0,1,2,6,7,8]]
  elif c['poly']=='squareless':x=x[:,[0,1,2,4,5,6,7,8]];z=z[:,[0,1,2,4,5,6,7,8]]
 names=['raw_NDVI','east','north']+({'full':['NDVI2','NDVI_east','NDVI_north','east2','east_north','north2'],'spatial':['east2','east_north','north2'],'squareless':['NDVI_east','NDVI_north','east2','east_north','north2'],'none':[]}[c['poly']])
 oa,ob=scaled(read(train,root,OPT),read(target,root,OPT),c['clip'])
 if c.get('opt_kind')=='physical':keep=[OPT.index(n) for n in PHYS];oa=oa[:,keep];ob=ob[:,keep];opts=PHYS
 else:opts=OPT
 oi=np.arange(x.shape[1],x.shape[1]+len(opts));x=np.c_[x,c['opt_weight']*oa];z=np.c_[z,c['opt_weight']*ob];names+=opts
 for kind,cols,w,standardize in [('ae',[f'c2025_A{i:02d}' for i in range(64)],c['ae_weight'],True),('height',HEIGHT,c['height_weight'],True),('mix',MIX,c['mix_weight'],True),('sar','oct103_',c['sar_weight'],False)]:
  if kind=='sar':a,b=pcs(read(train,root,cols,'dino_sar'),read(target,root,cols,'dino_sar'),False,c['clip'])
  elif kind=='ae':a,b=pcs(read(train,root,cols),read(target,root,cols),True,c['clip'])
  elif kind=='height':
   a=read(train,root,cols);b=read(target,root,cols);mean=a.mean(0);den=np.sqrt(a.var(0)+np.array([3,1,1])**2);a=(a-mean)/den;b=(b-mean)/den
   if c['clip'] is not None:a=np.clip(a,-c['clip'],c['clip']);b=np.clip(b,-c['clip'],c['clip'])
  else:
   a,b=scaled(read(train,root,cols),read(target,root,cols),c['clip'])
   if c.get('drop_mix_grvi',False):keep=[i for i,n in enumerate(cols) if n!='deblur_GRVI'];a=a[:,keep];b=b[:,keep];cols=[cols[i] for i in keep]
  x=np.c_[x,c['context_weight']*w*a];z=np.c_[z,c['context_weight']*w*b];names+=([kind+str(i) for i in range(a.shape[1])] if kind in ['ae','sar'] else cols)
 x=np.c_[x,np.full(len(train),10.)];z=np.c_[z,np.full(len(target),10.)];names+=['intercept']
 return x,z,oi,names

def weights(y):return np.where(y==1,len(y)/(2*max(1,y.sum())),len(y)/(2*max(1,(1-y).sum())))

def fit(train,target,ctx,cfg):
 c={**DEFAULT,**cfg};y=(train.label=='Unhealthy').to_numpy(int);root=ctx['datasets_dir']
 x,z,oi,names=design(train,target,root,c);pool=np.array([i for i,n in enumerate(names) if n in (OPT if c['pool']=='all' else PHYS if c['pool']=='physical' else [])],int)
 coeffs=[]
 for e,d in sorted(ctx['other_labeled'].items()):
  if len(pool)==0:break
  d=d[d.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);ys=(d.label=='Unhealthy').to_numpy(int)
  if len(np.unique(ys))<2:continue
  a,_,_,_=design(d,d,root,c)
  m=LogisticRegression(C=c['C'],fit_intercept=False,tol=1e-7,max_iter=2000).fit(a,ys,sample_weight=weights(ys));coeffs.append(m.coef_[0,pool])
 centre=np.mean(coeffs,axis=0) if coeffs else np.zeros(len(pool));prec=c['pool_strength']/c['C'];w=weights(y)
 penalty=np.full(x.shape[1],1/c['C'])
 if c.get('nuisance_penalty',1)!=1:
  for i,n in enumerate(names):
   if n in ['log_green_red','ndvi_sd20','ndvi_edge','deblur_GRVI']:penalty[i]*=c['nuisance_penalty']
 def objective(beta):
  s=x@beta;diff=beta[pool]-centre
  val=np.sum(w*(np.logaddexp(0,s)-y*s))+.5*np.sum(penalty*beta*beta)-.5*np.sum(penalty[pool]*beta[pool]**2)+.5*prec*np.sum(diff*diff)
  grad=x.T@(w*(expit(s)-y))+penalty*beta;grad[pool]+=prec*diff-penalty[pool]*beta[pool]
  return val,grad
 bounds=None
 if c.get('core_sign',False):
  bounds=[(None,0) if n in ['raw_NDVI','qNDVI','qNDRE'] else (None,None) for n in names]
 res=minimize(objective,np.zeros(x.shape[1]),jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-6})
 return x@res.x,z@res.x,dict(beta=res.x,names=names,success=res.success)

def core(train,target,ctx,kind='quality',degree='spatial',C=.1):
 lat=train.Lat.mean();lon=train.Long.mean();root=ctx['datasets_dir'];y=(train.label=='Unhealthy').to_numpy(int)
 def arr(d):
  nd=d.NDVI.to_numpy(float)
  if kind=='quality':nd=.5*nd+.5*read(d,root,['qNDVI_p75'],'gee_quality')[:,0]
  return np.c_[nd,(d.Long-lon)*111320*np.cos(np.deg2rad(lat)),(d.Lat-lat)*110540]
 a,b=scaled(arr(train),arr(target));x=a;z=b
 if degree in ['spatial','full']:
  p=PolynomialFeatures(2,include_bias=False);x=p.fit_transform(a);z=p.transform(b)
  if degree=='spatial':x=x[:,[0,1,2,6,7,8]];z=z[:,[0,1,2,6,7,8]]
 m=LogisticRegression(C=C,class_weight='balanced',max_iter=2000,tol=1e-8).fit(x,y)
 return m.decision_function(x),m.decision_function(z)

def calibrate(ref,base_ref,base_scores,kind='quantile'):
 if kind=='z':return (base_scores-base_ref.mean())/max(base_ref.std(),1e-8)*ref.std()+ref.mean()
 if kind=='robust':return (base_scores-np.median(base_ref))/max(np.subtract(*np.percentile(base_ref,[75,25])),1e-8)*np.subtract(*np.percentile(ref,[75,25]))+np.median(ref)
 s=np.sort(base_ref);q=(np.searchsorted(s,base_scores,side='left')+np.searchsorted(s,base_scores,side='right'))/(2*len(s));return np.quantile(ref,q)

def predict(labeled,target,ctx,cfg):
 c={**DEFAULT,**cfg};d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True)
 if len(target)==0:return np.empty(0,dtype=np.float64)
 if d.label.nunique()<2:return np.full(len(target),(d.label=='Unhealthy').mean(),dtype=np.float64)
 f,s,diag=fit(d,target,ctx,c)
 if c['guard']:
  br,bs=core(d,target,ctx,c.get('core_kind','quality'),c.get('core_degree','spatial'),c.get('core_C',.1))
  z=(target.NDVI.to_numpy()-d.NDVI.mean())/max(d.NDVI.std(ddof=0),1e-8)
  guard=c['guard'];threshold=c.get('threshold',2.5);width=c.get('width',.35)
  base=calibrate(f,br,bs,c.get('calibration','quantile'))
  if guard=='hard':s=np.where(z< -threshold,bs,s)
  elif guard=='budgetfloor':
   w=np.clip((-z-threshold+width)/(2*width),0,1)
   q=c.get('budget_quantile',.75)
   cap=np.quantile(f,q)
   desired=np.minimum(base,cap+c.get('tail_slope',.1)*np.std(f)*(bs-np.median(br))/max(np.std(br),1e-8))
   s=s+w*np.maximum(desired-s,0)
  elif guard=='diagnosticfloor':
   from scipy.stats import spearmanr
   # Label-free safety trigger on training score-vigor concordance.
   corr=float(spearmanr(f,-d.NDVI.to_numpy()).statistic)
   if corr<c.get('correlation_threshold',0.):
    w=np.clip((-z-threshold+width)/(2*width),0,1)
    s=s+w*np.maximum(base-s,0)
  elif guard=='rankresidual':
   bscore=calibrate(f,br,bs,c.get('calibration','quantile'))
   bound=c.get('bound',1.)*np.std(f)
   s=bscore+bound*np.tanh((s-bscore)/max(bound,1e-8))
  elif guard=='floor':
   w=np.clip((-z-threshold+width)/(2*width),0,1)
   s=s+w*np.maximum(base-s,0)
  elif guard=='blend':
   w=np.clip((-z-threshold+width)/(2*width),0,1);s=(1-w)*s+w*base
  elif guard=='max':s=np.maximum(s,base)
  elif guard=='residual':
   # Explicitly cap deviation from protected core in the low-NDVI tail.
   w=np.clip((-z-threshold+width)/(2*width),0,1);bound=c.get('bound',1.)*np.std(f)
   safe=base+np.clip(s-base,-bound,bound);s=(1-w)*s+w*safe
  elif guard=='globalblend':s=(1-c.get('blend',.5))*s+c.get('blend',.5)*base
 return np.asarray(s,dtype=np.float64)

def fit_predict(labeled,unlabeled,ctx,cfg=None):
 with threadpool_limits(limits=max(1,int(ctx['cpus']))):return predict(labeled,unlabeled,ctx,cfg or {})
