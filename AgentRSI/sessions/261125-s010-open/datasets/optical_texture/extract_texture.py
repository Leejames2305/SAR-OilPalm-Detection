"""Canopy-neighborhood optical features from label-free chip arrays."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import map_coordinates
ROOT=Path('/app/research/optical_chips');OUT=Path('/app/datasets/optical_texture');OUT.mkdir(exist_ok=True)
x=np.load(ROOT/'chips.npz')['chips'];blue,green,red,nir,re1,nira,swir=x.transpose(1,0,2,3)
nd=(nir-red)/(nir+red+1e-8);nr=(nira-re1)/(nira+re1+1e-8);nm=(nir-swir)/(nir+swir+1e-8)
# Windows aligned to 10m pixel centres; include central pixel index16.
yy,xx=np.mgrid[:32,:32];rr=np.sqrt((xx-16)**2+(yy-16)**2)
def mean(a,r):return np.mean(a[:,rr<=r],axis=1)
def med(a,r):return np.median(a[:,rr<=r],axis=1)
def sd(a,r):return np.std(a[:,rr<=r],axis=1)
def ring(a,lo,hi):return np.mean(a[:,(rr>lo)&(rr<=hi)],axis=1)
d=pd.read_parquet(ROOT/'keys.parquet')
data=dict(ndvi_center=nd[:,16,16],ndvi_med10=med(nd,1),ndvi_bg40=med(nd,4),ndvi_contrast=med(nd,1)-med(nd,4),ndvi_sd20=sd(nd,2),ndvi_sd40=sd(nd,4),ndvi_dip=ring(nd,1,3)-nd[:,16,16],ndvi_edge=np.sqrt(np.diff(nd,axis=1)[:,:,:-1]**2+np.diff(nd,axis=2)[:,:-1,:]**2)[:,14:18,14:18].mean((1,2)),nir_sd20=sd(nir,2),red_sd20=sd(red,2),ndre_contrast=med(nr,1)-med(nr,4),ndmi_contrast=med(nm,1)-med(nm,4))
for k,v in data.items():d[k]=v
assert not d.isna().any().any() and len(d)==7220
d.to_parquet(OUT/'features.parquet',index=False)
print(d.shape)
