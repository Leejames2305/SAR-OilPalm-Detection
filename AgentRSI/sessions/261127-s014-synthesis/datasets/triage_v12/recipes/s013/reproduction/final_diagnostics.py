from pathlib import Path
import sys,importlib.util,json
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
sys.path.insert(0,'/app/research')
from evaluate import splits,budget,metrics
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv')
spec=importlib.util.spec_from_file_location('submission_final',ROOT/'submission/pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data={e:d.reset_index(drop=True) for e,d in L.groupby('estate')};rankings=[];budgets=[];errors=[];support=[];weights=[]
for estate,d in data.items():
    t=U[U.estate==estate].copy();ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=ROOT/'research',cpus=1)
    s=m.fit_predict(d,t,ctx);t=t[['estate','Long','Lat']].copy();t['risk_score']=s;t['within_estate_rank']=rankdata(-s,method='min').astype(int);t['illustrative_top25']=t.within_estate_rank<=int(np.ceil(len(t)*.25));rankings.append(t)
    oof=np.load(ROOT/f'research/results/pooled_sar_official_{estate}.npz')['s'];ok=d.label.isin(['Healthy','Unhealthy']).to_numpy();y=(d.label=='Unhealthy').to_numpy(int)
    for include_middle in [False,True]:
        mask=np.ones(len(d),dtype=bool) if include_middle else ok
        score=oof[mask];outcome=y[mask];labels=d.label.to_numpy()[mask]
        for f in [.05,.10,.15,.20,.25,.30,.40,.50]:
            n=len(score)*f;threshold=np.sort(score)[::-1][int(np.ceil(n))-1];selected=(score>threshold).astype(float);tie=score==threshold;selected[tie]=(n-selected.sum())/tie.sum()
            found=float((selected*outcome).sum());healthy=float(selected[labels=='Healthy'].sum());middle=float(selected[labels=='Middle'].sum())
            budgets.append(dict(estate=estate,include_middle=include_middle,budget=f,inspections=n,found=found,healthy=healthy,middle=middle,recall=found/outcome.sum(),precision=found/n,other_per_found=(n-found)/max(found,1e-9)))
    posdist=np.empty(len(d));alldist=np.empty(len(d));lat=d.Lat.mean();xy=np.c_[d.Long*111320*np.cos(np.deg2rad(lat)),d.Lat*110540]
    for k,tr,va in splits(d,'official'):
        binary=tr[ok[tr]];positive=tr[y[tr]==1]
        alldist[va]=cKDTree(xy[binary]).query(xy[va])[0];posdist[va]=cKDTree(xy[positive]).query(xy[va])[0]
    scores=oof[ok];threshold=np.sort(scores)[::-1][int(np.ceil(len(scores)*.25))-1];selected=oof>=threshold
    for lo,hi in [(0,25),(25,80),(80,150),(150,np.inf)]:
        take=(y==1)&(posdist>=lo)&(posdist<hi);errors.append(dict(estate=estate,lower_distance=lo,upper_distance=hi,n_positive=int(take.sum()),found_top25=int(selected[take].sum()),recall=float(selected[take].mean()) if take.any() else None))
    binary=d[ok];cols=m._OPTICAL+m._HEIGHT+m._MIXTURE
    a=m._read(binary,ROOT/'datasets',cols);b=m._read(U[U.estate==estate],ROOT/'datasets',cols)
    for j,c in enumerate(cols):support.append(dict(estate=estate,feature=c,training_min=float(a[:,j].min()),training_max=float(a[:,j].max()),U_outside_range=int(((b[:,j]<a[:,j].min())|(b[:,j]>a[:,j].max())).sum())))
    records=[];original=m.LogisticRegression
    class Capture(original):
        def fit(self,x,y,**kw):
            result=super().fit(x,y,**kw);records.append((self,x));return result
    m.LogisticRegression=Capture;m.fit_predict(d,U[U.estate==estate],ctx);m.LogisticRegression=original
    fitted,x=records[-1];coef=fitted.coef_[0];k=12;width=49;base=9
    for j,c in enumerate(m._OPTICAL):
        # Raw standardized feature coefficient: shared .3 plus local .5.
        shared=.3*coef[j];local=.5*coef[k+base+j]
        weights.append(dict(estate=estate,feature=c,shared=shared,local=local,total=shared+local))
rank=pd.concat(rankings).reindex(U.index)
assert len(rank)==4473 and np.array_equal(rank[['estate','Long','Lat']].to_numpy(),U[['estate','Long','Lat']].to_numpy())
rank.to_csv(ROOT/'report/unlabelled_ranking.csv',index=False);rank.to_parquet(ROOT/'report/unlabelled_ranking.parquet',index=False)
pd.DataFrame(budgets).to_csv(ROOT/'report/inspection_budget_diagnostics.csv',index=False)
pd.DataFrame(errors).to_csv(ROOT/'report/error_by_positive_distance.csv',index=False)
pd.DataFrame(support).to_csv(ROOT/'report/covariate_support.csv',index=False)
pd.DataFrame(weights).to_csv(ROOT/'report/final_optical_coefficients.csv',index=False)
print('PASS 4473 exact-order rankings; inspection budgets, spatial errors and conditional coefficients saved')
