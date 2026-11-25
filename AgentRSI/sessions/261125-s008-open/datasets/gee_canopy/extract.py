"""Canopy-centred current and prior-year optical covariates, no label access."""
import os,time,json
from pathlib import Path
import ee,numpy as np,pandas as pd
ROOT=Path('/app');OUT=Path(__file__).parent
cred=ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']);ee.Initialize(cred,project=os.environ['GCP_PROJECT'])
D=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv').drop(columns='label'),pd.read_csv(ROOT/'data/unlabelled.csv')],ignore_index=True)
def mask(im):
 s=im.select('SCL');return im.updateMask(s.gte(4).And(s.lte(7)))
def make(start,end,roi):
 im=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start,end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',80)).map(mask).select(['B3','B4','B5','B6','B8','B8A','B11','B12']).median().setDefaultProjection('EPSG:32648',None,10)
 def norm(a,b):return im.select(a).subtract(im.select(b)).divide(im.select(a).add(im.select(b)).add(1e-6))
 ix=norm('B8','B4').rename('NDVI').addBands(norm('B8A','B5').rename('NDRE1')).addBands(norm('B8','B11').rename('NDMI')).addBands(norm('B3','B4').rename('GRVI'))
 # Highest NDVI among a 3x3 native 10m grid supplies a canopy pixel proxy.
 nd=ix.select('NDVI');bands=im.addBands(ix)
 kernel=ee.Kernel.square(1,'pixels',False)
 arr=bands.neighborhoodToBands(kernel)
 ns=arr.bandNames();sub=nd.neighborhoodToBands(kernel);best=sub.toArray().arrayArgmax().arrayGet([0]).toInt()
 selected=[]
 for name in ['B3','B4','B8','NDVI','NDRE1','NDMI','GRVI']:
  a=arr.select(name+'_.*').toArray();selected.append(a.arrayGet(best).rename(name+'_crown'))
 result=ee.Image.cat(selected)
 for name in ['NDVI','NDRE1','NDMI','GRVI']:
  x=ix.select(name);med=x.focalMedian(10,'circle','meters');bg=x.focalMedian(35,'circle','meters');sd=x.reduceNeighborhood(ee.Reducer.stdDev(),ee.Kernel.circle(20,'meters'))
  result=result.addBands(x.rename(name+'_point')).addBands(med.rename(name+'_med10')).addBands(med.subtract(bg).rename(name+'_contrast35')).addBands(sd.rename(name+'_std20'))
 return result
rows=[]
for e,g in D.groupby('estate',sort=False):
 roi=ee.Geometry.Rectangle([g.Long.min()-.001,g.Lat.min()-.001,g.Long.max()+.001,g.Lat.max()+.001])
 im=make('2026-05-15','2026-07-15',roi)
 prev=make('2025-05-15','2025-07-15',roi);im=im.addBands(prev.rename(prev.bandNames().map(lambda n:ee.String('prev_').cat(n))))
 print(e,'bands',im.bandNames().getInfo(),flush=True)
 for i in range(0,len(g),250):
  sub=g.iloc[i:i+250];fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'seq':int(j)}) for j,r in sub.iterrows()]
  got=im.reduceRegions(collection=ee.FeatureCollection(fs),reducer=ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']
  for f in got:
   p=f['properties'];j=p.pop('seq');r=D.loc[j];rows.append(dict(estate=r.estate,Long=r.Long,Lat=r.Lat,**p))
  print(e,i,'ok',flush=True)
 out=D[['estate','Long','Lat']].merge(pd.DataFrame(rows),on=['estate','Long','Lat'],how='left',sort=False)
 out.to_parquet(OUT/'features.parquet',index=False)
 print('interim',out.shape,'nan',out.isna().sum().sum(),flush=True)
print('finished',out.shape,'NaNs',out.isna().sum().sum(),flush=True)
