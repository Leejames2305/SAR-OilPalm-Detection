"""Controlled cross-lineage synthesis. No estate-specific recipe branches."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

KEY=['estate','Long','Lat']
OPT=['qNDVI','qNDRE','qNDMI','qNIRv','log_green_red','NDVI_contrast35','NDRE1_contrast35','NDMI_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
HEIGHT=['height_mean6','height_sd6','height_contrast6']
MIX=['deblur_'+v for v in ['NDVI','NDRE','NDMI','NBR','NIRv','GRVI','NDVI_contrast','NDRE_contrast','NDMI_contrast']]
BASE=['NDVI','east','north','NDVI2','NDVI_east','NDVI_north','east2','east_north','north2']
SIGNS={v:-1 for v in ['NDVI','qNDVI','qNDRE','qNDMI','qNIRv','deblur_NDVI','deblur_NDRE','deblur_NDMI','deblur_NBR','deblur_NIRv','vigor','water','mix_vigor','mix_water']}
SIGNS['ndvi_dip']=1

@lru_cache(maxsize=8)
def table(root,name='triage_v12'):
 return pd.read_parquet(Path(root)/name/'features.parquet').set_index(KEY)

def read(d,root,cols,name='triage_v12'):
 t=table(str(root),name)
 if isinstance(cols,str):cols=[c for c in t.columns if c.startswith(cols)]
 a=t.reindex(pd.MultiIndex.from_frame(d[KEY]))[cols].to_numpy(float)
 if not np.isfinite(a).all():raise ValueError('Missing prepared features')
 return a

def blocks(d,root,size=80):
 # Dataset contains complete authorized covariate-only L+U keys, stable union origin.
 keys=table(str(root)).index.to_frame(index=False)
 keys=keys[keys.estate==str(d.estate.iloc[0])]
 east=(d.Long.to_numpy()-keys.Long.min())*111320*np.cos(np.deg2rad(keys.Lat.mean()))
 north=(d.Lat.to_numpy()-keys.Lat.min())*110540
 b=np.floor(np.c_[east,north]/80).astype(np.int64)
 if size!=80:b=np.floor((b+.5)*80/size).astype(np.int64)
 return b[:,0]*100000+b[:,1]

def scaled(a,b):
 sc=StandardScaler().fit(a)
 return sc.transform(a),sc.transform(b)

def pcs(a,b,standard=True):
 if standard:a,b=scaled(a,b)
 p=PCA(n_components=min(8,len(a)-1,a.shape[1]),svd_solver='full').fit(a)
 return scaled(p.transform(a),p.transform(b))

def optical(d,t,root,cfg):
 a,b=scaled(read(d,root,OPT),read(t,root,OPT));names=list(OPT)
 if cfg.get('grouped'):
  groups=[([0,1,3],'vigor'),([2],'water'),([4],'log_green_red'),([5,6],'vigor_contrast'),([7],'NDMI_contrast35'),([8],'ndvi_dip'),([9,10],'roughness'),([11],'ndre_contrast')]
  a=np.column_stack([a[:,ix].mean(1) for ix,_ in groups]);b=np.column_stack([b[:,ix].mean(1) for ix,_ in groups]);names=[n for _,n in groups]
  a,b=scaled(a,b)
 return a,b,names

def design(d,t,root,cfg):
 lat,lon=float(d.Lat.mean()),float(d.Long.mean())
 def base(v):
  nd=v.NDVI.to_numpy(float)
  if cfg.get('quality_blend'):nd=.5*(nd+read(v,root,['qNDVI'])[:,0])
  return np.c_[nd,(v.Long.to_numpy()-lon)*111320*np.cos(np.deg2rad(lat)),(v.Lat.to_numpy()-lat)*110540]
 a,b=scaled(base(d),base(t));poly=PolynomialFeatures(2,include_bias=False)
 x,z=poly.fit_transform(a),poly.transform(b);x[:,3:]*=cfg.get('curvature',.3);z[:,3:]*=cfg.get('curvature',.3);names=list(BASE)
 oa,ob,on=optical(d,t,root,cfg)
 x,z=np.c_[x,.5*oa],np.c_[z,.5*ob];names+=on
 if cfg.get('no_coords'):
  ix=[1,2,4,5,6,7,8];x[:,ix]=0;z[:,ix]=0
 if cfg.get('additive'):
  x[:,[3,4,5]]=0;z[:,[3,4,5]]=0
 core_a,core_b=np.c_[a,oa],np.c_[b,ob]
 contexts_a=[];contexts_b=[];cn=[]
 if cfg.get('alpha',True):
  cols=[f'c2025_A{i:02d}' for i in range(64)]
  aa,bb=pcs(read(d,root,cols),read(t,root,cols));contexts_a.append(.5*aa);contexts_b.append(.5*bb);cn += [f'ae_pc{i}' for i in range(aa.shape[1])]
 if cfg.get('height',True):
  aa,bb=read(d,root,HEIGHT),read(t,root,HEIGHT);mean=aa.mean(0);den=np.sqrt(aa.var(0)+np.array([3.,1.,1.])**2)
  contexts_a.append(.35*(aa-mean)/den);contexts_b.append(.35*(bb-mean)/den);cn+=HEIGHT
 if cfg.get('mixture',True):
  aa,bb=scaled(read(d,root,MIX),read(t,root,MIX));mn=list(MIX)
  if cfg.get('grouped'):
   groups=[([0,1,4],'mix_vigor'),([2,3],'mix_water'),([5],'deblur_GRVI'),([6,7],'mix_vigor_contrast'),([8],'deblur_NDMI_contrast')]
   aa=np.column_stack([aa[:,ix].mean(1) for ix,_ in groups]);bb=np.column_stack([bb[:,ix].mean(1) for ix,_ in groups]);aa,bb=scaled(aa,bb);mn=[n for _,n in groups]
  contexts_a.append(.3*aa);contexts_b.append(.3*bb);cn+=mn
 for patch in cfg.get('patches',['oct103_']):
  aa,bb=pcs(read(d,root,patch,'dino_sar'),read(t,root,patch,'dino_sar'),False)
  contexts_a.append(.3*aa);contexts_b.append(.3*bb);cn += [patch+f'pc{i}' for i in range(aa.shape[1])]
 if contexts_a:
  ca,cb=np.c_[tuple(contexts_a)],np.c_[tuple(contexts_b)]
  if cfg.get('orthogonal'):
   if cfg['orthogonal']=='spectral':core_a,core_b=np.c_[a[:,0],oa],np.c_[b[:,0],ob]
   reg=cfg.get('projection_ridge',.1)
   projection=np.linalg.solve(core_a.T@core_a+reg*len(d)*np.eye(core_a.shape[1]),core_a.T@ca)
   ca,cb=ca-core_a@projection,cb-core_b@projection
  strength=cfg.get('context_scale',1.)
  x,z=np.c_[x,strength*ca],np.c_[z,strength*cb];names+=cn
 return x,z,names

def weights(y):
 return np.where(y==1,len(y)/(2*max(1,y.sum())),len(y)/(2*max(1,(1-y).sum())))

def fit_model(x,y,w,cfg,names,penalty=None):
 C=cfg.get('C',.15)
 if not cfg.get('signs') and penalty is None:
  m=LogisticRegression(C=C,fit_intercept=False,tol=1e-7,max_iter=2000).fit(x,y,sample_weight=w)
  return m.coef_[0]
 # Same summed weighted logistic loss/ridge objective as sklearn, plus explicit controls.
 bounds=[(None,None)]*x.shape[1]
 if cfg.get('signs'):
  for i,n in enumerate(names):
   sign=SIGNS.get(n)
   if sign:bounds[i]=((None,0) if sign<0 else (0,None))
 def objective(beta):
  s=x@beta
  value=np.sum(w*(np.logaddexp(0,s)-y*s))+.5*np.dot(beta,beta)/C
  gradient=x.T@(w*(expit(s)-y))+beta/C
  if penalty is not None:
   projection=penalty@beta
   value+=.5*np.dot(projection,projection)
   gradient+=penalty.T@projection
  return value,gradient
 result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':2000,'ftol':1e-12,'gtol':1e-6})
 return result.x

def joint(d,t,ctx,cfg,return_details=False):
 root=ctx['datasets_dir'];frames=[d]
 if cfg.get('pool',True):frames += [v[v.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True) for _,v in sorted(ctx['other_labeled'].items())]
 local=[];shared=[];outcomes=[];ws=[];allnames=[];penalties=[]
 for j,v in enumerate(frames):
  u=t if j==0 else v
  a,b,names=design(v,u,root,cfg);sa,sb,snames=optical(v,u,root,cfg)
  y=(v.label=='Unhealthy').to_numpy(int);w=weights(y)*len(d)/len(v)
  group=blocks(v,root)
  if j==0 and cfg.get('domain') and len(t)>0:
   da,db=(optical(v,t,root,cfg)[:2] if cfg['domain']=='optical' else (a,b))
   da,db=scaled(da,db)
   dm=LogisticRegression(C=.01,class_weight='balanced',max_iter=1000,tol=1e-7)
   dm.fit(np.vstack([da,db]),np.r_[np.zeros(len(da)),np.ones(len(db))])
   ratio=np.exp(np.clip(dm.decision_function(da),-np.log(2),np.log(2)))
   w*=ratio
   w[y==1]*=(len(d)/2)/w[y==1].sum();w[y==0]*=(len(d)/2)/w[y==0].sum()
  if cfg.get('block_balance'):
   _,inv,count=np.unique(group,return_inverse=True,return_counts=True)
   w*=1/np.sqrt(count[inv]);w*=len(d)/w.sum()
   # retain equal total class mass in each estate after spatial weighting
   w[y==1]*=(len(d)/2)/w[y==1].sum();w[y==0]*=(len(d)/2)/w[y==0].sum()
  local.append(a);shared.append(sa);outcomes.append(y);ws.append(w)
  allnames.append(names)
  if j==0:tz,ts=b,sb
 lengths=[len(a) for a in local];widths=[a.shape[1] for a in local]
 k=shared[0].shape[1] if cfg.get('pool',True) else 0;n=len(local);p=k+sum(widths)+n
 x=np.zeros((sum(lengths),p));z=np.zeros((len(t),p));encoded_names=list(snames) if k else []
 if k:
  scale=cfg.get('shared_scale',.3);x[:,:k]=scale*np.vstack(shared);z[:,:k]=scale*ts
 row,col=0,k
 for j,(a,length,width) in enumerate(zip(local,lengths,widths)):
  x[row:row+length,col:col+width]=a;x[row:row+length,k+sum(widths)+j]=10
  if j==0:z[:,col:col+width]=tz;z[:,k+sum(widths)+j]=10
  encoded_names+=allnames[j]
  row+=length;col+=width
 encoded_names += ['intercept']*n
 penalty=None
 if cfg.get('anchor'):
  # Penalize between-block healthy-background logit variability, separately per estate.
  # Fixed background-control experiment, not disease labels or prior neighbours at inference.
  ps=[];row=0
  for v,yy,length in zip(frames,outcomes,lengths):
   a=x[row:row+length];group=blocks(v,root,160)
   h=a[yy==0];centre=h.mean(0)
   for g in np.unique(group):
    ix=(group==g)&(yy==0)
    if ix.sum()>=4:ps.append((a[ix].mean(0)-centre)*np.sqrt(cfg['anchor']*len(d)/len(np.unique(group))))
   row+=length
  if ps:penalty=np.vstack(ps)
 beta=fit_model(x,np.concatenate(outcomes),np.concatenate(ws),cfg,encoded_names,penalty)
 scores=z@beta
 if return_details:return scores,dict(x=x,z=z,beta=beta,names=encoded_names,local_names=allnames[0],k=k,widths=widths,lengths=lengths)
 return scores

def predict(labeled,unlabeled,ctx,cfg):
 if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
 d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy(int)
 if len(np.unique(y))<2:return np.full(len(unlabeled),y.mean() if len(y) else 0.,dtype=np.float64)
 with threadpool_limits(limits=max(1,int(ctx['cpus']))):
  return np.asarray(joint(d,unlabeled,ctx,cfg),dtype=np.float64)


def adaptive(labeled,unlabeled,ctx,cfg):
 d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True)
 y=(d.label=='Unhealthy').to_numpy(int)
 if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
 if len(np.unique(y))<2:return np.full(len(unlabeled),float(y.mean()) if len(y) else 0.,dtype=np.float64)
 from sklearn.metrics import average_precision_score
 options=cfg.get('options',[{'C':.05},{'C':.15},{'C':.45}])
 group=blocks(d,ctx['datasets_dir']);ng=len(np.unique(group))
 if ng<3:return predict(d,unlabeled,ctx,options[0])
 scores=[];folds=list(GroupKFold(min(4,ng)).split(d,groups=group))
 with threadpool_limits(limits=max(1,int(ctx['cpus']))):
  for option in options:
   s=np.zeros(len(d))
   for tr,va in folds:
    s[va]=predict(d.iloc[tr],d.iloc[va].drop(columns='label'),ctx,option)
   scores.append(average_precision_score(y,s))
  best=int(np.argmax(scores))
  # Prefer the simplest candidate within a fixed .005 AP indifference margin.
  selected=next((i for i,value in enumerate(scores) if value>=scores[best]-.005),best)
  return predict(d,unlabeled,ctx,options[selected])


def source_prior(labeled,unlabeled,ctx,cfg):
 """Modular source response transfer, with target-local contextual coefficients."""
 if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
 d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True)
 y=(d.label=='Unhealthy').to_numpy(int)
 if len(np.unique(y))<2:return np.full(len(unlabeled),y.mean() if len(y) else 0.,dtype=np.float64)
 root=ctx['datasets_dir'];base=dict(cfg,pool=False)
 with threadpool_limits(limits=max(1,int(ctx['cpus']))):
  source_betas=[]
  for _,v in sorted(ctx['other_labeled'].items()):
   v=v[v.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True)
   yy=(v.label=='Unhealthy').to_numpy(int)
   if len(np.unique(yy))<2:continue
   a,_,names=design(v,v,root,base);a=np.c_[a,np.full(len(v),10.)]
   beta=fit_model(a,yy,weights(yy),base,names+['intercept'])
   source_betas.append(beta[9:9+len(OPT)])
  if not source_betas:return predict(d,unlabeled,ctx,base)
  coeff=np.vstack(source_betas);mu=coeff.mean(0)
  precision=np.full(len(OPT),1/cfg.get('C',.15))
  if cfg.get('heterogeneity'):
   # Fixed unit logit-coefficient disagreement floor: low source agreement weakens borrowing.
   precision/=1+np.var(coeff,axis=0,ddof=0)/(.5**2)
  a,b,names=design(d,unlabeled,root,base);a=np.c_[a,np.full(len(d),10.)];b=np.c_[b,np.full(len(unlabeled),10.)]
  w=weights(y);C=cfg.get('C',.15);ix=np.arange(9,9+len(OPT))
  def objective(beta):
   s=a@beta;diff=beta[ix]-mu
   value=np.sum(w*(np.logaddexp(0,s)-y*s))+.5*(beta@beta)/C-.5*(beta[ix]@beta[ix])/C+.5*np.sum(precision*diff*diff)
   grad=a.T@(w*(expit(s)-y))+beta/C;grad[ix]+=precision*diff-beta[ix]/C
   return value,grad
  res=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-6})
  return np.asarray(b@res.x,dtype=np.float64)
