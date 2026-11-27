"""Label-free annual foundation embeddings and local background contrasts."""
import os,time
from pathlib import Path
import ee,numpy as np,pandas as pd
OUT=Path('/app/datasets/alphaearth');OUT.mkdir(exist_ok=True)
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
D=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)
NAMES=[f'A{i:02}' for i in range(64)]
allrows=[]
for estate,d in D.groupby('estate',sort=False):
 roi=ee.Geometry.Rectangle([d.Long.min()-.003,d.Lat.min()-.003,d.Long.max()+.003,d.Lat.max()+.003])
 col=ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(roi)
 out=[]
 for year in [2023,2024,2025]:
  im=col.filterDate(f'{year}-01-01',f'{year+1}-01-01').mosaic().select(NAMES).setDefaultProjection('EPSG:32648',None,10)
  bg=im.focalMean(40,'circle','meters')
  out.extend([im.rename([f'y{year}_{n}' for n in NAMES]),im.subtract(bg).rename([f'c{year}_{n}' for n in NAMES])])
 image=ee.Image.cat(out)
 for start in range(0,len(d),200):
  sub=d.iloc[start:start+200];fc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'seq':int(j)}) for j,r in sub.iterrows()])
  for attempt in range(4):
   try:
    got=image.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features'];break
   except Exception:
    if attempt==3:raise
    time.sleep(2)
  for f in got:
   p=f['properties'];idx=p.pop('seq');r=D.loc[idx];allrows.append(dict(estate=r.estate,Long=r.Long,Lat=r.Lat,**p))
  print(estate,start,'ok',flush=True)
 keys=D[['estate','Long','Lat']]
 result=keys.merge(pd.DataFrame(allrows),on=['estate','Long','Lat'],how='left',validate='one_to_one')
 result.to_parquet(OUT/'features.parquet',index=False)
result=keys.merge(pd.DataFrame(allrows),on=['estate','Long','Lat'],how='left',validate='one_to_one')
assert len(result)==7220 and not result.isna().any().any()
result.to_parquet(OUT/'features.parquet',index=False)
print('DONE',result.shape,flush=True)
