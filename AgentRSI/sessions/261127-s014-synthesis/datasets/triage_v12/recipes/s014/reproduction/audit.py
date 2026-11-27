from pathlib import Path
import hashlib,importlib.metadata as md,json,ast,sys
import numpy as np,pandas as pd
ROOT=Path('/app');keys=['estate','Long','Lat'];L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv');allkeys=pd.concat([L[keys],U[keys]],ignore_index=True)
expected=pd.MultiIndex.from_frame(allkeys);forbidden={'label','class','id','fid','objectid','tree_id','treeid','row_id','block','part'}
records=[]
for p in sorted((ROOT/'datasets').glob('*/features.*')):
 if p.suffix not in ['.csv','.parquet']:continue
 d=pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p);actual=pd.MultiIndex.from_frame(d[keys]);cols=[c for c in d if c not in keys]
 row=dict(name=p.parent.name,shape=d.shape,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),duplicates=int(actual.duplicated().sum()),missing=int(len(expected.difference(actual))),extra=int(len(actual.difference(expected))),nan=int(d.isna().sum().sum()),nonfinite=int((~np.isfinite(d[cols].to_numpy(float))).sum()),forbidden=sorted(forbidden.intersection(map(str.lower,d.columns))),manifest=(p.parent/'MANIFEST.md').exists())
 row['valid']=(row['duplicates']==row['missing']==row['extra']==row['nan']==row['nonfinite']==0 and not row['forbidden'] and row['manifest']);assert row['valid'],row;records.append(row)
symlinks=[str(p) for root in ['datasets','submission'] for p in (ROOT/root).rglob('*') if p.is_symlink()];assert not symlinks
size=sum(p.stat().st_size for p in (ROOT/'datasets').rglob('*') if p.is_file());assert size<128*1024**2
versions={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','threadpoolctl']}
(ROOT/'report/dataset_audit.json').write_text(json.dumps(dict(tables=records,bytes=size,symlinks=symlinks),indent=2));(ROOT/'report/environment.json').write_text(json.dumps(dict(python=sys.version,versions=versions),indent=2))
print('All',len(records),'exact-key tables valid;',round(size/1024**2,2),'MiB; no symlinks')
