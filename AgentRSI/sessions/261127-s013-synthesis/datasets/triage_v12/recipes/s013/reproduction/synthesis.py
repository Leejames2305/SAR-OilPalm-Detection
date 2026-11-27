"""Controlled cross-lineage mechanisms. Parameters are research-only configurations."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from threadpoolctl import threadpool_limits

KEY=['estate','Long','Lat']
OPT=['qNDVI','qNDRE','qNDMI','qNIRv','log_green_red','NDVI_contrast35','NDRE1_contrast35','NDMI_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
SHARED=['qNDRE','qNDMI','NDRE1_contrast35','ndvi_dip']
HEIGHT=['height_mean6','height_sd6','height_contrast6']
MIX=['deblur_'+v for v in ['NDVI','NDRE','NDMI','NBR','NIRv','GRVI','NDVI_contrast','NDRE_contrast','NDMI_contrast']]

@lru_cache(maxsize=12)
def table(root,name):
    return pd.read_parquet(Path(root)/name/'features.parquet').set_index(KEY)

def read(d,root,cols,name='triage_v12'):
    t=table(str(root),name)
    if isinstance(cols,str):cols=[c for c in t.columns if c.startswith(cols)]
    v=t.reindex(pd.MultiIndex.from_frame(d[KEY]))[cols].to_numpy(float)
    if not np.isfinite(v).all():raise ValueError('Missing prepared covariates')
    return v

def scaled(x,t):
    if x.shape[1]==0:return x,t
    s=StandardScaler().fit(x);return s.transform(x),s.transform(t)

def pca(x,t,standardize=True):
    if standardize:x,t=scaled(x,t)
    p=PCA(n_components=min(8,len(x)-1,x.shape[1]),svd_solver='full').fit(x)
    return scaled(p.transform(x),p.transform(t))

def base(d,t,root,cfg):
    lat,lon=float(d.Lat.mean()),float(d.Long.mean())
    def f(v):
        nd=v.NDVI.to_numpy(float)
        if cfg.get('quality_blend'):nd=.5*nd+.5*read(v,root,['qNDVI'])[:,0]
        return np.c_[nd,(v.Long.to_numpy()-lon)*111320*np.cos(np.deg2rad(lat)),(v.Lat.to_numpy()-lat)*110540]
    x,z=scaled(f(d),f(t));p=PolynomialFeatures(2,include_bias=False)
    x,z=p.fit_transform(x),p.transform(z)
    x[:,3:]*=cfg.get('curvature',.3);z[:,3:]*=cfg.get('curvature',.3)
    if cfg.get('no_coords'):x[:,[1,2,4,5,6,7,8]]=0;z[:,[1,2,4,5,6,7,8]]=0
    return x,z

def physical(d,root,cfg):
    if cfg.get('optical','compact')=='compact':
        cols=cfg.get('optcols',OPT);return read(d,root,cols),cols
    q=table(str(root),'gee_quality').reindex(pd.MultiIndex.from_frame(d[KEY]))
    c=read(d,root,['NDVI_contrast35','NDRE1_contrast35'],'gee_canopy')
    tex=read(d,root,['ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast'],'optical_jitter')
    return np.c_[np.log(np.maximum(q.B4.to_numpy()+q.B8.to_numpy(),1)),q.B3.to_numpy()/(q.B4.to_numpy()+q.B3.to_numpy()+1),c,tex],['log_brightness','green_fraction','NDVI_contrast35','NDRE1_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']

def patches(d,t,root,cfg):
    xs=[];zs=[];names=[]
    for group in cfg.get('patches',[]):
        name,prefix=('dino_sar','oct103_') if group=='sar' else ('dino_optical','nrg160_')
        x,z=pca(read(d,root,prefix,name),read(t,root,prefix,name),False)
        if cfg.get('clip_patch'):x=np.clip(x,-3,3);z=np.clip(z,-3,3)
        xs.append(x);zs.append(z);names += [group+f'_pc{i}' for i in range(x.shape[1])]
    return (np.column_stack(xs),np.column_stack(zs),names) if xs else (np.empty((len(d),0)),np.empty((len(t),0)),[])

def residual_features(d,t,root,cfg):
    x,z,names=patches(d,t,root,dict(cfg,clip_patch=True))
    if cfg.get('extra_offset'):
        a,b=read(d,root,HEIGHT),read(t,root,HEIGHT);mean=a.mean(0);den=np.sqrt(a.var(0)+np.array([3.,1.,1.])**2)
        ha,hb=(a-mean)/den,(b-mean)/den
        ma,mb=scaled(read(d,root,MIX),read(t,root,MIX))
        x=np.c_[x,ha,np.clip(ma,-3,3)];z=np.c_[z,hb,np.clip(mb,-3,3)]
    return x,z

def design(d,t,root,cfg):
    x,z=base(d,t,root,cfg);names=['NDVI','east','north','NDVI2','NDVI_east','NDVI_north','east2','east_north','north2']
    a,onames=physical(d,root,cfg);b,_=physical(t,root,cfg)
    a,b=scaled(a,b);v=cfg.get('opt_scale',.5);x=np.c_[x,v*a];z=np.c_[z,v*b];names+=onames
    if cfg.get('alpha',True):
        cols=[f'c2025_A{i:02d}' for i in range(64)]
        a,b=pca(read(d,root,cols),read(t,root,cols),cfg.get('alpha_std',True));v=cfg.get('alpha_scale',.5)
        x=np.c_[x,v*a];z=np.c_[z,v*b];names += [f'ae_pc{i}' for i in range(a.shape[1])]
    if cfg.get('height',True):
        a,b=read(d,root,HEIGHT),read(t,root,HEIGHT);mean=a.mean(0);den=np.sqrt(a.var(0)+np.array([3.,1.,1.])**2)
        x=np.c_[x,.35*(a-mean)/den];z=np.c_[z,.35*(b-mean)/den];names+=HEIGHT
    if cfg.get('mixture',True):
        cols=cfg.get('mixcols',MIX);a,b=scaled(read(d,root,cols),read(t,root,cols));x=np.c_[x,.3*a];z=np.c_[z,.3*b];names+=cols
    a,b,pnames=patches(d,t,root,cfg)
    if len(pnames):
        v=cfg.get('patch_scale',.3);x=np.c_[x,v*a];z=np.c_[z,v*b];names+=pnames
    return x,z,names

def weights(y):
    return np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))

def logistic(x,y,w,C=.15,intercept=True):
    return LogisticRegression(C=C,max_iter=2000,tol=1e-7,fit_intercept=intercept).fit(x,y,sample_weight=w)

def block(d):
    east=(d.Long.to_numpy()-d.Long.min())*111320*np.cos(np.deg2rad(d.Lat.mean()))
    north=(d.Lat.to_numpy()-d.Lat.min())*110540
    return np.floor(east/80).astype(np.int64)*100000+np.floor(north/80).astype(np.int64)

def offset_fit(x,y,o,lam=.5):
    w=weights(y)/len(y)
    def objective(beta):
        s=o+x@beta
        return np.sum(w*(np.logaddexp(0,s)-y*s))+.5*lam*(beta@beta),x.T@(w*(expit(s)-y))+lam*beta
    return minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B').x

def joint(d,t,ctx,cfg):
    root=ctx['datasets_dir'];y=(d.label=='Unhealthy').to_numpy(int)
    x,z,_=design(d,t,root,cfg)
    if not cfg.get('pool'):
        if cfg.get('bag',0):
            _,inv=np.unique(block(d),return_inverse=True);n=inv.max()+1;rng=np.random.default_rng(20261127)
            scores=[]
            for k in range(cfg['bag']):
                count=np.bincount(rng.integers(n,size=n),minlength=n)[inv]
                pos=float((count*y).sum());neg=float((count*(1-y)).sum())
                if min(pos,neg)==0:continue
                w=count*np.where(y==1,len(y)/(2*pos),len(y)/(2*neg))
                m=logistic(x,y,w,cfg.get('C',.15));scores.append(expit(m.decision_function(z)))
            return np.mean(scores,axis=0)
        m=logistic(x,y,weights(y),cfg.get('C',.15));return m.decision_function(z)
    frames=[d]+[v[v.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True) for _,v in sorted(ctx['other_labeled'].items())]
    local=[];shared=[];ys=[];ws=[]
    cols=cfg.get('shared_cols',SHARED)
    patch_shared=None
    if cfg.get('shared_patch'):
        # One aligned basis for all estates, with estate-specific location/scale.
        # Independent per-estate PCA bases cannot carry a common coefficient.
        pcov=[];target_patch=None
        for j,v in enumerate(frames):
            a,b=scaled(read(v,root,'oct103_','dino_sar'),read(t if j==0 else v,root,'oct103_','dino_sar'))
            pcov.append(a)
            if j==0:target_patch=b
        p=PCA(n_components=8,svd_solver='full').fit(np.vstack(pcov))
        patch_shared=[]
        for j,a in enumerate(pcov):
            a,b=scaled(p.transform(a),p.transform(target_patch) if j==0 else p.transform(a))
            patch_shared.append(a)
            if j==0:target_shared_patch=b
    for j,v in enumerate(frames):
        u=t if j==0 else v
        a,b,_=design(v,u,root,cfg)
        sa,sb=scaled(read(v,root,cols),read(u,root,cols))
        if patch_shared is not None:
            sa=np.c_[sa,patch_shared[j]]
            sb=np.c_[sb,target_shared_patch if j==0 else patch_shared[j]]
        if cfg.get('pool_self') and j>0:continue
        yv=(v.label=='Unhealthy').to_numpy(int)
        if cfg.get('permute') and j>0:yv=np.random.default_rng(450+j).permutation(yv)
        local.append(a);shared.append(sa);ys.append(yv);ws.append(weights(yv)*len(d)/len(v))
        if j==0:tz=b;ts=sb
    widths=[a.shape[1] for a in local];lengths=[len(a) for a in local];k=shared[0].shape[1];n=len(local)
    a=np.zeros((sum(lengths),k+sum(widths)+n));b=np.zeros((len(t),a.shape[1]))
    shared_scale=cfg.get('shared_scale',.3);a[:,:k]=shared_scale*np.vstack(shared);b[:,:k]=shared_scale*ts
    row=0;col=k
    for j,(width,length) in enumerate(zip(widths,lengths)):
        a[row:row+length,col:col+width]=local[j];a[row:row+length,k+sum(widths)+j]=10
        if j==0:b[:,col:col+width]=tz;b[:,k+sum(widths)+j]=10
        row+=length;col+=width
    m=logistic(a,np.concatenate(ys),np.concatenate(ws),cfg.get('C',.15),False)
    return m.decision_function(b)

def predict(labeled,unlabeled,ctx,cfg):
    if len(unlabeled)==0:return np.empty(0,dtype=np.float64)
    d=labeled[labeled.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy(int)
    if len(np.unique(y))<2:return np.full(len(unlabeled),y.mean() if len(y) else 0.,dtype=np.float64)
    with threadpool_limits(limits=max(1,int(ctx['cpus']))):
        if cfg.get('offset'):
            core=dict(cfg,patches=[],pool=False,bag=0,offset=False)
            if cfg.get('extra_offset'):core.update(height=False,mixture=False)
            s=joint(d,unlabeled,ctx,core);o=np.zeros(len(d))
            for tr,va in GroupKFold(min(5,len(np.unique(block(d))))).split(d,groups=block(d)):
                o[va]=joint(d.iloc[tr],d.iloc[va],ctx,core)
            x,z=residual_features(d,unlabeled,ctx['datasets_dir'],cfg)
            coef=offset_fit(x,y,o,cfg.get('offset_lam',.5))
            correction=z@coef
            return np.asarray(s+cfg.get('offset_scale',.3)*np.clip(correction,-1,1),dtype=np.float64)
        return np.asarray(joint(d,unlabeled,ctx,cfg),dtype=np.float64)
