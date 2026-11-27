from pathlib import Path
import ast,hashlib,importlib.metadata,json
import numpy as np,pandas as pd
ROOT=Path('/app');KEY=['estate','Long','Lat'];allrows=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv')[KEY],pd.read_csv(ROOT/'data/unlabelled.csv')[KEY]],ignore_index=True)
keys=pd.MultiIndex.from_frame(allrows);forbidden={'label','class','id','fid','objectid','tree_id','treeid','row_id','block','part'};rows=[]
assert not keys.duplicated().any() and len(keys)==7220
for p in sorted((ROOT/'datasets').glob('*/features.parquet')):
    d=pd.read_parquet(p);k=pd.MultiIndex.from_frame(d[KEY]);v=d.drop(columns=KEY).to_numpy(float)
    assert len(d)==7220 and not k.duplicated().any() and len(keys.difference(k))==0 and len(k.difference(keys))==0
    assert not (forbidden & set(d.columns.str.lower())) and np.isfinite(v).all()
    rows.append(dict(dataset=p.parent.name,shape=list(d.shape),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),missing=0,duplicate_keys=0,nonfinite=0))
symlinks=[str(p) for base in ['datasets','submission'] for p in (ROOT/base).rglob('*') if p.is_symlink()];assert not symlinks
size=sum(p.stat().st_size for p in (ROOT/'datasets').rglob('*') if p.is_file());assert size<128*1024**2
imports=set()
for p in (ROOT/'submission').glob('*.py'):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,ast.Import):imports.update(a.name.split('.')[0] for a in n.names)
        if isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
assert imports<= {'functools','pathlib','numpy','pandas','sklearn','threadpoolctl'}
assert (ROOT/'submission/requirements.txt').read_text().strip()=='threadpoolctl==3.8.0'
result=dict(datasets=rows,count=len(rows),total_dataset_bytes=size,symlinks=symlinks,submission_imports=sorted(imports),non_base_pins={'threadpoolctl':'3.8.0'},pipeline_sha256=hashlib.sha256((ROOT/'submission/pipeline.py').read_bytes()).hexdigest())
(ROOT/'report/dataset_audit.json').write_text(json.dumps(result,indent=2))
environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','pyyaml','threadpoolctl']};(ROOT/'report/environment.json').write_text(json.dumps(environment,indent=2))
print('PASS',len(rows),'tables',size/1024**2,'MiB; imports and pins valid')
