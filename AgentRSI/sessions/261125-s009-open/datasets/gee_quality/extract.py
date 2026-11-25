"""Cloud-score filtered optical condition measurements at native 10m support."""
import os
from pathlib import Path
import ee,pandas as pd
OUT=Path(__file__).parent
cred=ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']);ee.Initialize(cred,project=os.environ['GCP_PROJECT'])
D=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)
def add(im):
 # strict vegetation SCL; CloudScore+ filters semitransparent clouds and shadows.
 mask=im.select('SCL').eq(4).And(im.select('cs_cdf').gte(.65));im=im.updateMask(mask)
 def norm(a,b):return im.select(a).subtract(im.select(b)).divide(im.select(a).add(im.select(b)).add(1e-6))
 return im.select(['B2','B3','B4','B8','B5','B6','B8A','B11']).addBands(norm('B8','B4').rename('qNDVI')).addBands(norm('B8A','B5').rename('qNDRE')).addBands(norm('B8','B11').rename('qNDMI')).addBands(im.select('B8').multiply(norm('B8','B4')).divide(10000).rename('qNIRv'))
rows=[]
for e,g in D.groupby('estate',sort=False):
 roi=ee.Geometry.Rectangle([g.Long.min()-.002,g.Lat.min()-.002,g.Long.max()+.002,g.Lat.max()+.002]);cs=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-04-15','2026-08-15').linkCollection(cs,['cs_cdf']).map(add)
 names=['qNDVI','qNDRE','qNDMI','qNIRv'];med=col.select(names).median();high=col.select(names).reduce(ee.Reducer.percentile([75]));count=col.select('qNDVI').count().rename('quality_count');reflect=col.select(['B2','B3','B4','B8','B5','B6','B8A','B11']).median()
 im=med.addBands(high).addBands(count).addBands(reflect)
 # Fallback to SCL4-7 from wider window only where strict quality has no observations.
 def fb(im):s=im.select('SCL');return im.updateMask(s.gte(4).And(s.lte(7))).addBands(ee.Image.constant(1).rename('cs_cdf'))
 fallback=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-03-15','2026-09-15').map(fb).map(add)
 fm=fallback.select(names).median().addBands(fallback.select(names).reduce(ee.Reducer.percentile([75]))).addBands(ee.Image.constant(0).rename('quality_count')).addBands(fallback.select(['B2','B3','B4','B8','B5','B6','B8A','B11']).median())
 im=im.unmask(fm)
 print(e,'start',flush=True)
 for i in range(0,len(g),350):
  sub=g.iloc[i:i+350];fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'seq':int(j)}) for j,r in sub.iterrows()]
  got=im.reduceRegions(collection=ee.FeatureCollection(fs),reducer=ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']
  for f in got:
   p=f['properties'];j=p.pop('seq');r=D.loc[j];rows.append(dict(estate=r.estate,Long=r.Long,Lat=r.Lat,**p))
  print(e,i,'ok',flush=True)
out=D[['estate','Long','Lat']].merge(pd.DataFrame(rows),on=['estate','Long','Lat'],how='left',sort=False);out.to_parquet(OUT/'features.parquet',index=False);print(out.shape,'NaNs',out.isna().sum().sum(),flush=True)
