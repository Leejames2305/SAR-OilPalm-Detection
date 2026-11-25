"""Build co-registered, label-free multi-date SAR crops. Run from /app."""
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
from scipy.ndimage import map_coordinates
ROOT=Path('/app');R=ROOT/'rasters'
D=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv').drop(columns='label'),pd.read_csv(ROOT/'data/unlabelled.csv')],ignore_index=True)
K=D[['estate','Long','Lat']].copy();n=len(D);out=np.zeros((n,22,15,15),np.float32);valid=np.zeros((n,2,15,15),np.float32)
yy,xx=np.meshgrid(np.arange(-7,8)*6.43,np.arange(-7,8)*6.43,indexing='ij')
for e in D.estate.unique():
 ix=np.flatnonzero(D.estate==e);g=D.iloc[ix]
 for j,date in enumerate(['260610','261014']):
  paths=[R/f'ALOS2-Subset_{e}_{date}_Cal_Spk_TC.tif',R/f'ALOS2-Subset_{e}_{date}_Cal_mat_HAlpha_Spk_TC.tif',R/f'ALOS2-Subset_{e}_{date}_Cal_mat_Yamaguchi_Spk_TC.tif']
  off=j*11
  with rasterio.open(paths[0]) as s:
   a=s.read();support=np.isfinite(a).all(0)&(a>0).all(0);crs=s.crs
  for z,f in enumerate(paths):
   with rasterio.open(f) as s:
    a=s.read().astype(float);px,py=transform('EPSG:4326',s.crs,g.Long.to_list(),g.Lat.to_list());inv=~s.transform
    X=np.asarray(px)[:,None,None]+xx;Y=np.asarray(py)[:,None,None]-yy
    col=inv.a*X+inv.b*Y+inv.c-.5;row=inv.d*X+inv.e*Y+inv.f-.5
    coords=np.array([row,col]);m=support&np.isfinite(a).all(0)
    if z<2:m &= (a>0).all(0)
    if z==0:valid[ix,j]=map_coordinates(m.astype(float),coords,order=0,mode='constant',cval=0)
    for b in range(len(a)):
     v=a[b];v=np.where(m,v,np.nan)
     if z==0:v=10*np.log10(v)
     if z==1 and b==2:v=v/90
     fill=np.nanmedian(v);v=np.nan_to_num(v,nan=fill,posinf=fill,neginf=fill)
     out[ix,off]=map_coordinates(v,coords,order=1,mode='nearest');off+=1
 print(e,'done',flush=True)
# Robust global channel scale, label-free and independent of split labels.
med=np.median(out,axis=(0,2,3));q1=np.percentile(out,25,axis=(0,2,3));q3=np.percentile(out,75,axis=(0,2,3));scale=np.maximum((q3-q1)/1.349,.02)
out=np.clip((out-med[None,:,None,None])/scale[None,:,None,None],-6,6)
np.savez_compressed('/run/agentrsi/colab-io/in/sar_patches.npz',patches=out.astype(np.float16),valid=valid.astype(np.float16),median=med,scale=scale)
K.to_parquet(Path(__file__).parent/'keys.parquet',index=False)
print('shape',out.shape,'finite',np.isfinite(out).all(),flush=True)
