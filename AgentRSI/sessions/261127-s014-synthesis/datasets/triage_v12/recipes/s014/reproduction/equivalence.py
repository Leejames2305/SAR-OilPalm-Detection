"""Clean package versus immutable V2 checkpoint. Scientific recipe stays frozen."""
from pathlib import Path
import importlib.util,sys,tempfile,json
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research');import evaluate
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv');data={e:v.reset_index(drop=True) for e,v in L.groupby('estate')}
sys.path.insert(0,str(ROOT/'research/checkpoint_v2'))
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
original=load(ROOT/'research/checkpoint_v2/pipeline.py','v2');clean=load(ROOT/'research/clean_pipeline.py','clean');rows=[]
for estate,d in data.items():
 with tempfile.TemporaryDirectory() as temp:
  ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=Path(temp),cpus=1)
  cases=[('full_U',d,U[U.estate==estate].reset_index(drop=True))]
  for protocol in ['official','repeat717','blockquad80','blockquad160']:
   for fold,tr,va in evaluate.splits(d,protocol):cases.append((protocol+'_'+str(fold),d.iloc[tr].reset_index(drop=True),d.iloc[va].drop(columns='label').reset_index(drop=True)))
  for name,a,b in cases:
   s=original.fit_predict(a,b,ctx);t=clean.fit_predict(a,b,ctx);delta=float(np.max(np.abs(s-t))) if len(t) else 0.
   np.testing.assert_allclose(s,t,rtol=1e-8,atol=1e-8)
   rows.append(dict(estate=estate,case=name,max_absolute_delta=delta,exact=bool(np.array_equal(s,t))))
result=dict(fits=len(rows),max_absolute_delta=max(r['max_absolute_delta'] for r in rows),all_exact=all(r['exact'] for r in rows),cases=rows)
(ROOT/'report/package_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
