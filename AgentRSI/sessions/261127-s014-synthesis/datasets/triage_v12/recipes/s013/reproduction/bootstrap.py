"""Paired block resampling conditional on fixed official OOF predictions."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research')
from evaluate import blocks
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv')
models=['incumbent','coords','s009','s010','s011','s012','s012_bag','pooled_sar']
estates=sorted(L.estate.unique());samples={};B=4000
for ei,estate in enumerate(estates):
    d=L[L.estate==estate].reset_index(drop=True);ok=d.label.isin(['Healthy','Unhealthy']).to_numpy();y=(d.label.to_numpy()[ok]=='Unhealthy').astype(float)
    _,inv=np.unique(blocks(d)[ok],return_inverse=True);ng=inv.max()+1
    rng=np.random.default_rng(261127+ei)
    count=np.array([np.bincount(rng.integers(ng,size=ng),minlength=ng) for _ in range(B)])
    w=count[:,inv]
    for model in models:
        s=np.load(ROOT/f'research/results/{model}_official_{estate}.npz')['s'][ok]
        order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
        allap=[]
        for begin in range(0,B,200):
            ww=w[begin:begin+200,order];pos=np.add.reduceat(ww*yy,starts,axis=1);total=np.add.reduceat(ww,starts,axis=1)
            cp=pos.cumsum(1);ct=total.cumsum(1);precision=np.divide(cp,ct,out=np.zeros_like(cp,dtype=float),where=ct>0)
            ap=np.sum(precision*pos,axis=1)/np.maximum(pos.sum(1),1)
            allap.extend(ap)
        samples[model,estate]=np.asarray(allap)
rows=[]
for reference in models[:-1]:
    deltas=[]
    for estate in estates:
        delta=samples['pooled_sar',estate]-samples[reference,estate];deltas.append(delta)
        rows.append(dict(reference=reference,estate=estate,mean=float(delta.mean()),lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975)),fraction_positive=float((delta>0).mean())))
    delta=np.mean(deltas,axis=0);rows.append(dict(reference=reference,estate='composite',mean=float(delta.mean()),lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975)),fraction_positive=float((delta>0).mean())))
pd.DataFrame(rows).to_csv(ROOT/'report/bootstrap.csv',index=False)
(ROOT/'report/bootstrap.json').write_text(json.dumps({'B':B,'unit':'original 80m block','fixed_OOF_no_refit':True,'selection_conditional':True,'caveat':'Adjacent blocks may remain correlated. Repeats/selection do not create independent estates or prospective observations.','results':rows},indent=2))
print(pd.DataFrame(rows).query('estate=="composite"').round(6).to_string(index=False))
