"""Descriptive, full-L conditional weights and collinearity; not confirmation."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/app/research')
import evaluate,synthesis as sy
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv')
data={e:d.reset_index(drop=True) for e,d in L.groupby('estate',sort=True)}
effects=[];pairs=[];coeffs=[];design_summary=[]
for estate,frame in data.items():
    d=frame[frame.label.isin(['Healthy','Unhealthy'])].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy(int)
    target=U[U.estate==estate].reset_index(drop=True);ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=ROOT/'research',cpus=1)
    raw=sy.table(str(ROOT/'datasets'),'triage_v12').reindex(pd.MultiIndex.from_frame(d[sy.KEY])).reset_index(drop=True)
    raw['NDVI']=d.NDVI.to_numpy()
    for name in raw.columns:
        x=raw[name].to_numpy(float);a=x[y==1];b=x[y==0];pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(x)-2))
        effects.append(dict(estate=estate,feature=name,d=(a.mean()-b.mean())/max(pooled,1e-12),sd=x.std()))
    relevant=['NDVI']+sy.OPT+sy.HEIGHT+sy.MIX
    cor=raw[relevant].corr()
    for i,a in enumerate(relevant):
        for b in relevant[i+1:]:
            if abs(cor.loc[a,b])>.7:pairs.append(dict(estate=estate,first=a,second=b,correlation=cor.loc[a,b]))
    for modelname in ['s011','s012','s010']:
        module,_=evaluate.load(modelname);records=[]
        class Capture(LogisticRegression):
            def fit(self,x,y,**kwargs):
                result=super().fit(x,y,**kwargs);records.append((self,x.copy(),np.asarray(y).copy()));return result
        original=module.LogisticRegression;module.LogisticRegression=Capture
        scores=module.fit_predict(frame,target,ctx);module.LogisticRegression=original
        fitted,x,yy=records[-1];beta=fitted.coef_[0]
        if modelname in ['s011','s012']:
            names=['NDVI','east','north','NDVI2','NDVI_east','NDVI_north','east2','east_north','north2']+sy.OPT+[f'ae_pc{i}' for i in range(8)]
            if modelname=='s012':names+=sy.HEIGHT+sy.MIX
            for col,name in enumerate(names):
                coeffs.append(dict(estate=estate,model=modelname,feature=name,coefficient=beta[col],score_contribution_sd=float((x[:,col]*beta[col]).std())))
        else:
            # Shared physical block and target independent optical block both use .3.
            names=['log_brightness','green_fraction','NDVI_contrast35','NDRE1_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
            basen=3 if estate=='AirHitam' else 9
            for col,name in enumerate(names):
                shared=beta[col];local=beta[8+basen+col]
                coeffs.append(dict(estate=estate,model=modelname,feature=name,coefficient=shared+local,shared_coefficient=shared,local_coefficient=local,score_contribution_sd=float((x[:len(d),col]*(shared+local)).std())))
        singular=np.linalg.svd(x,compute_uv=False);design_summary.append(dict(model=modelname,estate=estate,n=x.shape[0],p=x.shape[1],condition=float(singular[0]/max(singular[-1],1e-14)),coef_norm=float(np.linalg.norm(beta))))
pd.DataFrame(effects).to_csv(ROOT/'report/feature_effects.csv',index=False)
ef=pd.DataFrame(effects).pivot(index='feature',columns='estate',values='d');ef['same_sign']=(np.sign(ef).nunique(axis=1)==1);ef.to_csv(ROOT/'report/feature_sign_consistency.csv')
pd.DataFrame(pairs).to_csv(ROOT/'report/feature_correlations.csv',index=False)
pd.DataFrame(coeffs).to_csv(ROOT/'report/model_coefficients.csv',index=False)
pd.DataFrame(design_summary).to_csv(ROOT/'report/design_condition.csv',index=False)
print(pd.DataFrame(coeffs).pivot(index=['model','feature'],columns='estate',values='coefficient').round(4).to_string())
