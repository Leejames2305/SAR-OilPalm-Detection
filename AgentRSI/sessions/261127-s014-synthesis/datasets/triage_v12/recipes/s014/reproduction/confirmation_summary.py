from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path('/app');x=pd.read_csv(ROOT/'report/EXPERIMENTS.csv');f=x[x.protocol.isin(['repeat'+str(i) for i in range(701,721)])]
assert len(f)==6*3*20
models=[]
for model,d in f.groupby('model'):
 by=d.pivot(index='protocol',columns='estate',values='lift');row={'model':model,'assignments':len(by),'composite':by.mean(axis=1).mean(),**{e:by[e].mean() for e in by.columns}}
 for ref in ['incumbent','coords','s009','s012','s013']:
  b=f[f.model==ref].pivot(index='protocol',columns='estate',values='lift');delta=by-b
  row['delta_'+ref]=delta.mean(axis=1).mean();row['composite_wins_'+ref]=int((delta.mean(axis=1)>0).sum())
 models.append(row)
rows=[]
for e,d in f[f.model=='source_prior'].groupby('estate'):
 row=dict(estate=e,mean_lift=d.lift.mean(),assignments=len(d))
 for ref in ['incumbent','coords','s009','s012','s013']:
  values=d['delta_'+ref];row['mean_delta_'+ref]=values.mean();row['worst_delta_'+ref]=values.min();row['wins_'+ref]=int((values>0).sum())
 rows.append(row)
pd.DataFrame(models).to_csv(ROOT/'report/frozen_model_comparison.csv',index=False);pd.DataFrame(rows).to_csv(ROOT/'report/frozen_confirmation.csv',index=False);f.to_csv(ROOT/'report/frozen_confirmation_long.csv',index=False)
print(pd.DataFrame(models).round(6).to_string(index=False));print('\nFinal perestate:');print(pd.DataFrame(rows).round(6).to_string(index=False))
