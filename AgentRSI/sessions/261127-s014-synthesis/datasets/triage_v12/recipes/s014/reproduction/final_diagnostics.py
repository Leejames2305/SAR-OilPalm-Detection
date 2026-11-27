from pathlib import Path
import sys,tempfile,json
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0,'/app/research');import engine,evaluate
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv');data={e:v.reset_index(drop=True) for e,v in L.groupby('estate')}
rows=[];rankings=[];weights=[];sourceweights=[];distance=[];support=[];disagreements=[]
def membership(s,f):
 size=len(s)*f;boundary=np.sort(s)[::-1][min(int(np.ceil(size))-1,len(s)-1)];a=s>boundary;b=s==boundary;return a.astype(float)+b*max(0,size-a.sum())/max(1,b.sum())
for estate,d in data.items():
 t=U[U.estate==estate].reset_index(drop=True);binary=d.label.isin(['Healthy','Unhealthy']).to_numpy();y=(d.label=='Unhealthy').to_numpy(int);s=np.load(ROOT/f'research/results/source_prior_official_{estate}.npz')['s']
 for include in [False,True]:
  keep=np.ones(len(d),bool) if include else binary
  for f in [.05,.10,.15,.20,.25,.30,.40,.50]:
   alloc=membership(s[keep],f);lab=d.label.to_numpy()[keep];found=float(alloc[lab=='Unhealthy'].sum());healthy=float(alloc[lab=='Healthy'].sum());middle=float(alloc[lab=='Middle'].sum());n=float(alloc.sum())
   rows.append(dict(estate=estate,include_middle=include,budget=f,inspections=n,found=found,healthy=healthy,middle=middle,recall=found/y.sum(),precision=found/n,enrichment=(found/y.sum())/f,other_per_found=(healthy+middle)/max(found,1e-8)))
 with tempfile.TemporaryDirectory() as tmp:
  ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=Path(tmp),cpus=1)
  us,info=engine.source_prior(d,t,ctx,{'details':True})
  allnames=info['names'];beta=info['beta']
  for i,name in enumerate(allnames):
   prior=info['prior'][i-9] if 9<=i<21 else 0.
   weights.append(dict(estate=estate,feature=name,coefficient=beta[i],prior_center=prior,direct_standardized_optical_effect=.5*beta[i] if 9<=i<21 else np.nan,score_contribution_sd=float(np.std(info['x'][:,i]*beta[i]))))
  for j,src in enumerate(sorted(ctx['other_labeled'])):
   for feature,value in zip(engine.OPT,info['source_coefficients'][j]):sourceweights.append(dict(target_estate=estate,source_estate=src,feature=feature,source_coef=value))
  out=t[engine.KEY].copy();out['risk_score']=us;out['rank']=out.risk_score.rank(method='first',ascending=False).astype(int);out['inspection_top25']=out['rank']<=int(np.ceil(.25*len(t)));rankings.append(out)
  train=d[binary];rawa=engine.read(train,ctx['datasets_dir'],engine.OPT);rawb=engine.read(t,ctx['datasets_dir'],engine.OPT)
  for feature,a,b in [('NDVI',train.NDVI.to_numpy(),t.NDVI.to_numpy())]+[(n,rawa[:,i],rawb[:,i]) for i,n in enumerate(engine.OPT)]:
   outside=(b<a.min())|(b>a.max());support.append(dict(estate=estate,feature=feature,L_min=a.min(),L_max=a.max(),U_min=b.min(),U_max=b.max(),U_outside=int(outside.sum()),U_total=len(t),outside_score_median=float(np.median(us[outside])) if outside.any() else np.nan))
 coords=evaluate.xy(d);dist=np.full(len(d),np.nan)
 for fold,tr,va in evaluate.splits(d,'official'):
  positives=tr[(d.iloc[tr].label=='Unhealthy').to_numpy()]
  dist[va]=cKDTree(coords[positives]).query(coords[va])[0] if len(positives) else np.inf
 allocation=np.zeros(len(d));allocation[binary]=membership(s[binary],.25)
 for lower,upper in [(0,25),(25,80),(80,150),(150,np.inf)]:
  mask=(dist>=lower)&(dist<upper)&(y==1);n=mask.sum();found=float(allocation[mask].sum());distance.append(dict(estate=estate,distance_low=lower,distance_high=upper,positives=int(n),found=found,recall25=found/n if n else np.nan))
 for model in ['incumbent','coords','s009','s012','s013']:
  sp=np.load(ROOT/f'research/results/{model}_official_{estate}.npz')['s'][binary];a=membership(s[binary],.25);b=membership(sp,.25)
  disagreements.append(dict(estate=estate,first='source_prior',second=model,spearman=spearmanr(s[binary],sp).statistic,top25_disagreement=float(np.abs(a-b).mean()),top25_jaccard=float(np.minimum(a,b).sum()/np.maximum(a,b).sum())))
for name,records in [('inspection_budget_diagnostics',rows),('final_coefficients',weights),('source_prior_coefficients',sourceweights),('error_by_positive_distance',distance),('covariate_support',support),('final_prediction_disagreement',disagreements)]:pd.DataFrame(records).to_csv(ROOT/f'report/{name}.csv',index=False)
combined=pd.concat(rankings,ignore_index=True);combined=U[engine.KEY].merge(combined,on=engine.KEY,how='left',validate='one_to_one',sort=False);assert len(combined)==len(U) and combined.risk_score.notna().all();combined.to_csv(ROOT/'report/unlabelled_ranking.csv',index=False);combined.to_parquet(ROOT/'report/unlabelled_ranking.parquet',index=False)
print(pd.DataFrame(rows).query('budget==.25').round(5).to_string(index=False));print('Aligned rankings:',len(combined))
