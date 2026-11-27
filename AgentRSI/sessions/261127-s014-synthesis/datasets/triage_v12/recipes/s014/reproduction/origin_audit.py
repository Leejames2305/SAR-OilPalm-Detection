from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0,'/app/research');import evaluate as ev
ROOT=Path('/app');L=pd.read_csv(ROOT/'data/L_labelled.csv');rows=[]
for estate,d in L.groupby('estate'):
 d=d.reset_index(drop=True);old=np.floor(np.c_[(d.Long-d.Long.min())*111320*np.cos(np.deg2rad(d.Lat.mean())),(d.Lat-d.Lat.min())*110540]/80).astype(np.int64);old=old[:,0]*100000+old[:,1];new=ev.blocks(d)
 leakage=np.zeros(len(d),bool);shared=set()
 for tr,va in GroupKFold(5).split(d,groups=old):
  both=np.intersect1d(new[tr],new[va]);shared.update(both.tolist());leakage[va]=np.isin(new[va],both)
 rows.append(dict(estate=estate,legacy_Lonly_blocks=len(np.unique(old)),correct_union_blocks=len(np.unique(new)),key_changes=int((old!=new).sum()),original_blocks_split_across_legacy_folds=len(shared),legacy_validation_rows_sharing_original_training_block=int(leakage.sum()),legacy_validation_share=float(leakage.mean())))
pd.DataFrame(rows).to_csv(ROOT/'report/validation_origin_audit.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
