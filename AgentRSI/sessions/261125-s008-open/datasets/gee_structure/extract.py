"""Extract independent 2020 canopy structure for confounding/interaction tests."""
import os
from pathlib import Path
import ee,pandas as pd
OUT=Path(__file__).parent
cred=ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']);ee.Initialize(cred,project=os.environ['GCP_PROJECT'])
D=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)
h=ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('height2020');sd=ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').rename('height_uncertainty')
im=h.addBands(sd).addBands(h.subtract(h.focalMedian(50,'circle','meters')).rename('height_contrast50')).addBands(h.reduceNeighborhood(ee.Reducer.stdDev(),ee.Kernel.circle(30,'meters')).rename('height_std30'))
rows=[]
for i in range(0,len(D),350):
 sub=D.iloc[i:i+350];fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'seq':int(j)}) for j,r in sub.iterrows()]
 got=im.reduceRegions(collection=ee.FeatureCollection(fs),reducer=ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']
 for f in got:
  p=f['properties'];j=p.pop('seq');r=D.loc[j];rows.append(dict(estate=r.estate,Long=r.Long,Lat=r.Lat,**p))
 print(i,'ok',flush=True)
out=D[['estate','Long','Lat']].merge(pd.DataFrame(rows),on=['estate','Long','Lat'],how='left',sort=False);out.to_parquet(OUT/'features.parquet',index=False);print(out.shape,'NaNs',out.isna().sum().sum(),flush=True)
