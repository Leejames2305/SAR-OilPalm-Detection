from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research')
from evaluate import metrics
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');D={e:d.reset_index(drop=True) for e,d in L.groupby('estate')}
rows=[]
for p in (ROOT/'research/results').glob('*.npz'):
 model,protocol,estate=p.stem.rsplit('_',2)
 z=np.load(p);rows.append(dict(model=model,protocol=protocol,estate=estate,**metrics(D[estate],z['s']),seconds=float(z['seconds'])))
df=pd.DataFrame(rows)
for ref in ['incumbent','coords','s009','s012','s013']:
 refdf=df[df.model==ref][['protocol','estate','lift']].rename(columns={'lift':'reference'})
 df=df.merge(refdf,on=['protocol','estate'],how='left');df['delta_'+ref]=df.lift-df.reference;df=df.drop(columns='reference')
df.to_csv(ROOT/'report/EXPERIMENTS.csv',index=False)
pivot=df.pivot(index=['model','protocol'],columns='estate',values='lift');pivot=pivot.dropna();pivot['composite']=pivot.mean(axis=1)
for ref in ['incumbent','coords','s009','s012','s013']:
 b=pivot.loc[ref,'composite'] if ref in pivot.index.get_level_values(0) else pd.Series(dtype=float)
 pivot['delta_'+ref]=[v-b.get(p,np.nan) for (m,p),v in pivot.composite.items()]
pivot.to_csv(ROOT/'report/EXPERIMENT_SUMMARY.csv')
print(pivot.xs('official',level='protocol').round(5).to_string())
print('\nRepeated means:')
print(df[df.protocol.str.startswith('repeat')].pivot_table(index='model',columns='estate',values='lift').round(5).to_string())
