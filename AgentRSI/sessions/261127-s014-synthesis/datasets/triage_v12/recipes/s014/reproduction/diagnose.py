from pathlib import Path
import itertools,json,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'/app/research')
import engine as sy,evaluate
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');D={e:d.reset_index(drop=True) for e,d in L.groupby('estate')}
effects=[];pairs=[];coeffs=[];dis=[];splitaudit=[]
for e,d in D.items():
 binary=d.label.isin(['Healthy','Unhealthy']);v=d[binary].reset_index(drop=True);y=(v.label=='Unhealthy').to_numpy(int)
 raw=sy.table('/app/datasets').reindex(pd.MultiIndex.from_frame(v[sy.KEY])).reset_index(drop=True);raw['NDVI']=v.NDVI.to_numpy()
 for name in raw.columns:
  a,b=raw[name].to_numpy(float)[y==1],raw[name].to_numpy(float)[y==0]
  sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
  effects.append(dict(estate=e,feature=name,d=(a.mean()-b.mean())/max(sd,1e-12)))
 relevant=['NDVI']+sy.OPT+sy.HEIGHT+sy.MIX;c=raw[relevant].corr()
 for a,b in itertools.combinations(relevant,2):
  if abs(c.loc[a,b])>.7:pairs.append(dict(estate=e,first=a,second=b,r=c.loc[a,b]))
 ctx=dict(estate=e,datasets_dir=ROOT/'datasets',other_labeled={f:t for f,t in D.items() if f!=e},cpus=1,scratch_dir=ROOT/'research')
 for model,cfg in [('s012',{'pool':False,'patches':[]}),('s013',{}),('grouped',{'grouped':True}),('signs',{'signs':True}),('orthogonal',{'orthogonal':'all'}),('anchor_background',{'anchor':.1})]:
  _,info=sy.joint(v,v,ctx,cfg,True);k=info['k'];beta=info['beta'];names=info['local_names']
  for i,name in enumerate(names):
   local=beta[k+i];scale=(.5 if name in sy.OPT else .3 if name in sy.MIX else 1)
   shared=beta[sy.OPT.index(name)] if k and name in sy.OPT else 0.
   total=local*scale+shared*.3
   coeffs.append(dict(estate=e,model=model,feature=name,encoded_coef=local,shared_encoded_coef=shared,total_standardized_coef=total,contribution_sd=float(np.std(info['x'][:len(v),k+i]*local))))
 models=['incumbent','coords','s009','s012','s013']
 for a,b in itertools.combinations(models,2):
  sa=np.load(ROOT/f'research/results/{a}_official_{e}.npz')['s'][binary];sb=np.load(ROOT/f'research/results/{b}_official_{e}.npz')['s'][binary]
  def allocation(s):
   size=.25*len(s);boundary=np.sort(s)[::-1][min(int(np.ceil(size))-1,len(s)-1)];above=s>boundary;tie=s==boundary
   return above.astype(float)+tie*max(0,size-above.sum())/max(1,tie.sum())
  ma,mb=allocation(sa),allocation(sb)
  dis.append(dict(estate=e,first=a,second=b,spearman=spearmanr(sa,sb).statistic,top25_disagreement=float(np.abs(ma-mb).mean()),top25_jaccard=float(np.minimum(ma,mb).sum()/np.maximum(ma,mb).sum())))
 original=evaluate.blocks(d)
 for protocol in ['official','block120','block160','buffer40','buffer80','blockquad','blockquad80','blockquad160']:
  for fold,tr,va in evaluate.splits(d,protocol):
   splitaudit.append(dict(estate=e,protocol=protocol,fold=fold,ntrain=len(tr),ntest=len(va),train_positive=int((d.iloc[tr].label=='Unhealthy').sum()),test_positive=int((d.iloc[va].label=='Unhealthy').sum()),shared_original_blocks=len(np.intersect1d(original[tr],original[va]))))
pd.DataFrame(effects).to_csv(ROOT/'report/feature_effects.csv',index=False)
ef=pd.DataFrame(effects).pivot(index='feature',columns='estate',values='d');ef['same_sign']=np.sign(ef).nunique(axis=1)==1;ef.to_csv(ROOT/'report/feature_sign_consistency.csv')
for name,rows in [('feature_correlations',pairs),('model_coefficients',coeffs),('prediction_disagreement',dis),('spatial_split_audit',splitaudit)]:pd.DataFrame(rows).to_csv(ROOT/f'report/{name}.csv',index=False)
print(pd.DataFrame(dis).query('first=="s012" and second=="s013"').to_string(index=False))
print(pd.DataFrame(coeffs).query('feature in ["qNDVI","qNDRE","qNDMI","ndvi_sd20","ndvi_edge"] and model in ["s012","s013"]').pivot(index=['model','feature'],columns='estate',values='total_standardized_coef').round(4).to_string())
