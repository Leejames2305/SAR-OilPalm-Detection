"""Reproduce the exact covariate-only runtime subset from carried source tables.

Run: python /app/datasets/compact_triage/consolidate.py
Source extraction scripts/manifests remain under the four named source folders.
Labels are never read; use only source-table exact keys, preserving row order.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
KEYS=['estate','Long','Lat']

def main():
    q=pd.read_parquet(ROOT/'gee_quality/features.parquet')
    keys=q[KEYS].copy()
    idx=pd.MultiIndex.from_frame(keys)
    def get(name,columns):
        x=pd.read_parquet(ROOT/name/'features.parquet').set_index(KEYS).reindex(idx)
        return x[columns].to_numpy(float)
    optical=['qNDVI','qNDRE','qNDMI','qNIRv','log_green_red','NDVI_contrast35','NDRE1_contrast35','NDMI_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
    qv=get('gee_quality',['qNDVI','qNDRE','qNDMI','qNIRv','B3','B4'])
    local=get('gee_canopy',['NDVI_contrast35','NDRE1_contrast35','NDMI_contrast35'])
    texture=get('optical_jitter',['ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast'])
    alpha=[f'c2025_A{i:02d}' for i in range(64)]
    values=np.c_[qv[:,:4],np.log(np.maximum(qv[:,4],1)/np.maximum(qv[:,5],1)),local,texture,get('alphaearth',alpha)]
    out=pd.concat([keys,pd.DataFrame(values,columns=optical+alpha)],axis=1)
    assert out.shape==(7220,79) and not out.duplicated(KEYS).any() and np.isfinite(values).all()
    target=ROOT/'compact_triage';target.mkdir(exist_ok=True)
    out.to_parquet(target/'features.parquet',index=False)
    sources={name:hashlib.sha256((ROOT/name/'features.parquet').read_bytes()).hexdigest() for name in ['gee_quality','gee_canopy','optical_jitter','alphaearth']}
    (target/'source_checksums.json').write_text(json.dumps(sources,indent=2))
    print(out.shape,'bytes',(target/'features.parquet').stat().st_size)

if __name__=='__main__':main()
