"""Prove clean submission predictions match the immutable V8 package."""
from pathlib import Path
import sys,importlib.util,json,tempfile
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research')
from evaluate import splits
ROOT=Path('/app')
def module(name,path):
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
verified=module('verified_v8',ROOT/'research/candidates/pooled_sar/portable/pipeline.py')
final=module('final_clean',ROOT/'submission/pipeline.py')
L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv');data={e:d.reset_index(drop=True) for e,d in L.groupby('estate')};rows=[]
for estate,d in data.items():
    ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=ROOT/'research',cpus=1)
    cases=[('full',d,U[U.estate==estate])]
    for protocol in ['official','blockquad80','repeat401']:
        for k,tr,va in splits(d,protocol):cases.append((f'{protocol}_{k}',d.iloc[tr].reset_index(drop=True),d.iloc[va].drop(columns='label').reset_index(drop=True)))
    for name,a,b in cases:
        s=verified.fit_predict(a,b,ctx);t=final.fit_predict(a,b,ctx);error=float(np.max(abs(s-t)));assert error<1e-10,(estate,name,error)
        rows.append(dict(estate=estate,case=name,max_difference=error))
(ROOT/'report/package_equivalence_final.json').write_text(json.dumps(rows,indent=2))
print('PASS',len(rows),'fit comparisons, max',max(r['max_difference'] for r in rows))
