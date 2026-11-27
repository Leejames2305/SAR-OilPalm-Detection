from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'/app/research')
from evaluate import metrics,budget
ROOT=Path('/app');OUT=ROOT/'research/results';L=pd.read_csv(ROOT/'data/L_labelled.csv')
data={e:d.reset_index(drop=True) for e,d in L.groupby('estate',sort=True)}
rows=[];cached={}
for p in sorted(OUT.glob('*.npz')):
    name,protocol,estate=p.stem.rsplit('_',2);z=np.load(p);s=z['s'];cached[name,protocol,estate]=s
    rows.append(dict(model=name,protocol=protocol,estate=estate,**metrics(data[estate],s),seconds=float(z['seconds'])))
df=pd.DataFrame(rows)
refs=['incumbent','coords','s009','s010','s011','s012','s012_bag']
for ref in refs:
    r=df[df.model==ref][['protocol','estate','lift']].rename(columns={'lift':'reference'})
    df=df.merge(r,on=['protocol','estate'],how='left');df['delta_'+ref]=df.lift-df.reference;df=df.drop(columns='reference')
df.to_csv(ROOT/'report/EXPERIMENTS.csv',index=False)
summary=df.groupby(['model','protocol']).agg({c:(lambda v: v.mean() if len(v)==3 and v.notna().sum()==3 else np.nan) for c in df.select_dtypes('number').columns}).reset_index()
counts=df.groupby(['model','protocol']).estate.nunique().rename('estate_count').reset_index()
summary=summary.merge(counts,on=['model','protocol']);summary.to_csv(ROOT/'report/EXPERIMENT_SUMMARY.csv',index=False)
official=df[df.protocol=='official'].pivot(index='model',columns='estate',values='lift');official['composite']=official.mean(axis=1)
for ref in refs:
    if ref in official.index:official['delta_'+ref]=official.composite-official.loc[ref,'composite']
official.to_csv(ROOT/'report/OFFICIAL_EXPERIMENTS.csv')
print(official.round(6).to_string())
repeats=df[df.protocol.str.startswith('repeat')].groupby(['model','estate']).agg(lift=('lift','mean'),minlift=('lift','min'),incwins=('delta_incumbent',lambda x:(x>0).sum()),n=('lift','size')).reset_index()
repeats.to_csv(ROOT/'report/repeat_summary.csv',index=False)
pairs=[]
for estate,d in data.items():
    ok=d.label.isin(['Healthy','Unhealthy']).to_numpy()
    names=[m for m in official.index if (m,'official',estate) in cached]
    for a,b in itertools.combinations(names,2):
        x=cached[a,'official',estate][ok];y=cached[b,'official',estate][ok]
        # Exact top-fraction disagreement, with fractional boundary inclusion.
        def selected(v):
            n=len(v)*.25;threshold=np.sort(v)[::-1][int(np.ceil(n))-1]
            out=(v>threshold).astype(float);tie=v==threshold;out[tie]=(n-out.sum())/tie.sum();return out
        ax,bx=selected(x),selected(y)
        pairs.append(dict(estate=estate,first=a,second=b,spearman=spearmanr(x,y).statistic,top25_jaccard=np.minimum(ax,bx).sum()/np.maximum(ax,bx).sum(),top25_disagreement=np.abs(ax-bx).sum()/len(x)))
pd.DataFrame(pairs).to_csv(ROOT/'report/prediction_disagreement.csv',index=False)
