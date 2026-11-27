"""Download cloud-controlled mosaics, construct label-free per-tree image chips."""
import os,time,zipfile,io
from pathlib import Path
import requests,ee,numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
from rasterio.transform import Affine
from rasterio.windows import Window
ROOT=Path('/app/research/optical_chips');ROOT.mkdir(exist_ok=True)
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
D=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)
bands=['B2','B3','B4','B8','B5','B8A','B11']
def add(im):
 s=im.select('SCL');return im.select(bands).updateMask(s.eq(4).And(im.select('cs_cdf').gte(.65)))
chips=np.zeros((len(D),7,32,32),dtype=np.float32)
for estate,d in D.groupby('estate',sort=False):
 path=ROOT/f'{estate}.tif'
 if not path.exists():
  roi=ee.Geometry.Rectangle([d.Long.min()-.003,d.Lat.min()-.003,d.Long.max()+.003,d.Lat.max()+.003])
  cs=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-04-15','2026-08-15').linkCollection(cs,['cs_cdf']).map(add)
  im=col.median()
  def fb(im):s=im.select('SCL');return im.select(bands).updateMask(s.eq(4))
  fallback=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-03-15','2026-09-15').map(fb).median()
  im=im.unmask(fallback).toFloat()
  # This service account has compute access but no thumbnails.create permission.
  # Use the documented sampleRectangle computation endpoint instead.
  bx,by=transform('EPSG:4326','EPSG:32648',[d.Long.min()-.003,d.Long.max()+.003],[d.Lat.min()-.003,d.Lat.max()+.003])
  xmin=np.floor(min(bx)/10)*10;ymax=np.ceil(max(by)/10)*10
  xmax=np.ceil(max(bx)/10)*10;ymin=np.floor(min(by)/10)*10
  proj=ee.Projection('EPSG:32648',[10,0,float(xmin),0,-10,float(ymax)])
  region=ee.Geometry.Rectangle([float(xmin),float(ymin),float(xmax),float(ymax)],'EPSG:32648',False)
  im=im.reproject(proj).addBands(ee.Image.pixelCoordinates(proj).rename(['px','py']))
  props=im.sampleRectangle(region=region,defaultValue=0).getInfo()['properties']
  arr=np.array([props[b] for b in bands],dtype=np.float32);px=np.array(props['px']);py=np.array(props['py'])
  trans=Affine(10,0,float(xmin+px.min()*10),0,-10,float(ymax-py.min()*10))
  with rasterio.open(path,'w',driver='GTiff',width=arr.shape[2],height=arr.shape[1],count=7,dtype='float32',crs='EPSG:32648',transform=trans,compress='deflate') as dst:dst.write(arr)
 with rasterio.open(path) as src:
  data=src.read();xs,ys=transform('EPSG:4326',src.crs,d.Long.tolist(),d.Lat.tolist())
  for (idx,row),x,y in zip(d.iterrows(),xs,ys):
   rr,cc=src.index(x,y);chip=src.read(window=Window(cc-16,rr-16,32,32),boundless=True,fill_value=np.nan)
   for k in range(7):
    valid=np.isfinite(chip[k])&(chip[k]>0);med=np.median(chip[k][valid]) if valid.any() else np.nanmedian(data[k][data[k]>0]);chip[k][~valid]=med
   chips[idx]=chip/10000
 print(estate,'done',flush=True)
np.savez_compressed(ROOT/'chips.npz',chips=chips)
D[['estate','Long','Lat']].to_parquet(ROOT/'keys.parquet',index=False)
print(chips.shape,float(chips.min()),float(chips.max()),flush=True)
