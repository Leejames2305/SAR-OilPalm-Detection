"""Audit grid edge/centre convention and test fixed registration integration."""
from pathlib import Path
import rasterio,numpy as np,pandas as pd
from rasterio.warp import transform
from scipy.ndimage import map_coordinates
ROOT=Path('/app/research/optical_chips')
D=pd.read_parquet(ROOT/'keys.parquet')
arrays={};indices={}
for e,d in D.groupby('estate',sort=False):
 with rasterio.open(ROOT/f'{e}.tif') as src:
  arrays[e]=src.read()/10000
  xs,ys=transform('EPSG:4326',src.crs,d.Long.tolist(),d.Lat.tolist())
  # The original sampleRectangle coordinates are pixel centres (.5), so the
  # saved raster origin was displaced +5m east/-5m south. Correct index here.
  cols=(np.array(xs)-(src.transform.c-5))/10;rows=((src.transform.f+5)-np.array(ys))/10
  indices[e]=(d.index.to_numpy(),rows,cols)

yy,xx=np.mgrid[-16:16,-16:16];rr=np.sqrt(xx**2+yy**2)
def calc(chips):
 blue,green,red,nir,re1,nira,swir=chips.transpose(1,0,2,3)
 nd=(nir-red)/(nir+red+1e-8);nr=(nira-re1)/(nira+re1+1e-8);nm=(nir-swir)/(nir+swir+1e-8)
 def med(a,r):return np.median(a[:,rr<=r],axis=1)
 def sd(a,r):return np.std(a[:,rr<=r],axis=1)
 ring=(rr>1)&(rr<=3)
 return dict(ndvi_dip=nd[:,ring].mean(1)-nd[:,16,16],ndvi_sd20=sd(nd,2),ndvi_edge=np.sqrt(np.diff(nd,axis=1)[:,:,:-1]**2+np.diff(nd,axis=2)[:,:-1,:]**2)[:,14:18,14:18].mean((1,2)),ndre_contrast=med(nr,1)-med(nr,4))

def make(offsets,name):
 out={}
 for dx,dy in offsets:
  chips=np.empty((len(D),7,32,32),np.float32)
  for e,(idx,rows,cols) in indices.items():
   a=arrays[e]
   for j,row,col in zip(idx,rows,cols):
    r=int(np.floor(row+dy));c=int(np.floor(col+dx));chips[j]=a[:,r-16:r+16,c-16:c+16]
  for k,v in calc(chips).items():out[k]=out.get(k,0)+v/len(offsets)
 d=D.copy()
 for k,v in out.items():d[k]=v
 dest=Path('/app/datasets')/name;dest.mkdir(exist_ok=True);d.to_parquet(dest/'features.parquet',index=False)
make([(0,0)],'optical_aligned')
make([(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)],'optical_jitter')
print('DONE')
