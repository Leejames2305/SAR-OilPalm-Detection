from pathlib import Path
import importlib.util,sys,tempfile,json
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research');import evaluate
ROOT=Path('/app');spec=importlib.util.spec_from_file_location('selected',ROOT/'submission/pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv');data={e:d.reset_index(drop=True) for e,d in L.groupby('estate')};records=[];original=m.minimize;current={}
def capture(*args,**kw):
 result=original(*args,**kw);records.append(dict(**current,success=bool(result.success),message=str(result.message),iterations=int(result.nit),gradient_max=float(np.max(np.abs(result.jac)))));assert result.success,records[-1];return result
m.minimize=capture
for estate,d in data.items():
 with tempfile.TemporaryDirectory() as scratch:
  ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=Path(scratch),cpus=1)
  current=dict(estate=estate,case='full_U');m.fit_predict(d,U[U.estate==estate],ctx)
  for protocol in ['official','repeat717','blockquad80','blockquad160']:
   for fold,tr,va in evaluate.splits(d,protocol):
    current=dict(estate=estate,case=protocol+'_'+str(fold));m.fit_predict(d.iloc[tr],d.iloc[va].drop(columns='label'),ctx)
(ROOT/'report/optimizer_audit.json').write_text(json.dumps(records,indent=2));print('Target convex optimizations converged:',len(records),'/',len(records))
