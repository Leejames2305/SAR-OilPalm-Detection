"""Extract SRTM/MERIT topographic and drainage proxies at every tree point."""
from pathlib import Path
import os, pandas as pd, ee
cred=ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']); ee.Initialize(cred,project=os.environ['GCP_PROJECT'])
root=Path('/app'); L=pd.read_csv(root/'data/L_labelled.csv'); U=pd.read_csv(root/'data/unlabelled.csv'); D=pd.concat([L.drop(columns=['label']),U],ignore_index=True).drop_duplicates(['estate','Long','Lat'])
srtm=ee.Image('USGS/SRTMGL1_003').select('elevation'); sl=ee.Terrain.slope(srtm); asp=ee.Terrain.aspect(srtm)
mer=ee.Image('MERIT/Hydro/v1_0_1');
im=(srtm.rename('elevation').addBands(sl.rename('slope')).addBands(asp.rename('aspect'))
    .addBands(mer.select('hnd').rename('hand')).addBands(mer.select('upa').add(1).log().rename('log_upa'))
    .addBands(srtm.subtract(srtm.focal_median(radius=150,units='meters')).rename('rel_elev150'))
    .addBands(srtm.subtract(srtm.focal_median(radius=500,units='meters')).rename('rel_elev500')))
def pull(df,chunk=500):
 out=[]
 for i in range(0,len(df),chunk):
  sub=df.iloc[i:i+chunk]; fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'row':int(j)}) for j,r in sub.iterrows()]; fc=ee.FeatureCollection(fs)
  got=im.reduceRegions(fc,ee.Reducer.mean(),30,tileScale=4).getInfo()['features']
  out.extend([f.get('properties',{}) for f in got]); print(i,len(got),flush=True)
 return out
arr=[]
for e,g in D.groupby('estate',sort=False):
 print(e,len(g),flush=True); p=pull(g.reset_index(drop=True));
 for rr,row in zip(p,g.itertuples()):
  rr['estate']=e; rr['Long']=float(row.Long); rr['Lat']=float(row.Lat)
 arr.extend(p)
out=pd.DataFrame(arr)
out=D[['estate','Long','Lat']].merge(out,on=['estate','Long','Lat'],how='left',sort=False)
out.to_parquet(Path(__file__).parent/'features.parquet',index=False)
print(out.shape,out.isna().sum().sum())
