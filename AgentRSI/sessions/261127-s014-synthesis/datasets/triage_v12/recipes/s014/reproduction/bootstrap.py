"""Paired resampling of original80m blocks; fixed OOF scores, selection-conditional."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research');import evaluate
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');rng=np.random.default_rng(261127014);B=4000
models=['incumbent','coords','s009','s012','s013','source_prior'];arrays={m:[] for m in models};rows=[]
for estate,d in L.groupby('estate'):
 d=d.reset_index(drop=True);ok=d.label.isin(['Healthy','Unhealthy']).to_numpy();y=(d.loc[ok,'label']=='Unhealthy').to_numpy(float)
 groups=evaluate.blocks(d)[ok];_,inv=np.unique(groups,return_inverse=True);ng=inv.max()+1
 counts=rng.multinomial(ng,np.full(ng,1/ng),size=B);w=counts[:,inv];positive=w@y;valid=positive>0
 for model in models:
  s=np.load(ROOT/f'research/results/{model}_official_{estate}.npz')['s'][ok];order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ww=w[:,order]
  ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(ww*yy,axis=1)[:,ends];total=np.cumsum(ww,axis=1)[:,ends];increments=np.diff(np.c_[np.zeros(B),tp],axis=1)
  ap=np.sum(np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)*increments,axis=1)/np.maximum(positive,1)
  ap[~valid]=np.nan;arrays[model].append(ap)
 base={m:arrays[m][-1] for m in models}
 for ref in models[:-1]:
  delta=base['source_prior']-base[ref];lo,hi=np.nanquantile(delta,[.025,.975]);rows.append(dict(model='source_prior',reference=ref,estate=estate,bootstrap_mean=float(np.nanmean(delta)),lower=float(lo),upper=float(hi),fraction_positive=float(np.nanmean(delta>0)),valid=int(valid.sum()),resamples=B))
for ref in models[:-1]:
 delta=np.nanmean(np.stack(arrays['source_prior'])-np.stack(arrays[ref]),axis=0);lo,hi=np.nanquantile(delta,[.025,.975]);rows.append(dict(model='source_prior',reference=ref,estate='composite',bootstrap_mean=float(np.nanmean(delta)),lower=float(lo),upper=float(hi),fraction_positive=float(np.mean(delta>0)),valid=int(np.isfinite(delta).sum()),resamples=B))
out=pd.DataFrame(rows);out.to_csv(ROOT/'report/bootstrap.csv',index=False);print(out.round(6).to_string(index=False))
(ROOT/'report/bootstrap_method.json').write_text(json.dumps(dict(resamples=B,seed=261127014,method='Resample original80m groups with replacement separately perestate; same block multiplicities for all models; step AP with weighted ties; fixed OOF predictions; unweighted mean of estate AP differences',limits=['No refitting','Conditions on selected fits and adaptive campaign','Adjacent blocks may be dependent','Same labels as model development; no prospective inference']),indent=2))
