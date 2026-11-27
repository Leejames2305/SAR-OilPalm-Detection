"""Independent late-season optical observations, paired canopy context."""
import os,shutil
from pathlib import Path
import numpy as np,pandas as pd,ee
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
root=Path('/app/datasets/autumn');root.mkdir(exist_ok=True)
d=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']]
out=[];cs=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');bands=['B2','B3','B4','B8','B5','B8A','B11']
for estate,a in d.groupby('estate',sort=False):
    roi=ee.Geometry.Rectangle([float(a.Long.min()-.004),float(a.Lat.min()-.004),float(a.Long.max()+.004),float(a.Lat.max()+.004)])
    col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-09-01','2026-11-16').linkCollection(cs,['cs_cdf'])
    def mask(im):return im.select(bands).updateMask(im.select('SCL').eq(4).And(im.select('cs_cdf').gte(.65)))
    im=col.map(mask).median().unmask(col.map(lambda im:im.select(bands).updateMask(im.select('SCL').eq(4))).median()).setDefaultProjection('EPSG:32648',None,10)
    count=col.map(mask).select('B8').count().rename('observations');q=im.addBands(count)
    def nd(a,b,name):return im.expression('(a-b)/(a+b)',{'a':im.select(a),'b':im.select(b)}).rename(name)
    for v in [nd('B8','B4','ndvi'),nd('B8A','B5','ndre'),nd('B8','B11','ndmi')]:
        name=v.bandNames().getInfo()[0];med=v.focalMedian(10,'circle','meters');bg=v.focalMedian(40,'circle','meters')
        q=q.addBands(v).addBands(med.rename(name+'_med10')).addBands(med.subtract(bg).rename(name+'_contrast40')).addBands(v.reduceNeighborhood(ee.Reducer.stdDev(),ee.Kernel.circle(20,'meters')).rename(name+'_sd20'))
    pts=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'joinseq':int(i)}) for i,r in a.iterrows()])
    vals=pd.DataFrame([f['properties'] for f in q.reduceRegions(pts,ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']]).set_index('joinseq').reindex(a.index)
    out.append(pd.concat([a,vals],axis=1));print(estate,vals.shape,'missing',vals.isna().sum().sum(),flush=True)
res=pd.concat(out).reindex(d.index);assert res.isna().sum().sum()==0;res.to_parquet(root/'features.parquet',index=False)
(root/'MANIFEST.md').write_text('''# Autumn optical condition and canopy-neighborhood covariates

Source COPERNICUS/S2_SR_HARMONIZED linked CloudScore+ cs_cdf>=.65, SCL4 vegetation,2026-09-01 through2026-11-15 inclusive. Median B2/B3/B4/B8/B5/B8A/B11 DN; fallback same-window SCL4 median. No extra harmonization offset. EPSG32648 default10m, red-edge/SWIR native20m not sharpened. Per-index NDVI(B8-B4)/(B8+B4),NDRE(B8A-B5)/(B8A+B5),NDMI(B8-B11)/(B8+B11), point/10m circular focal median/median10-minus-median40/SD20. Observations=count strict-valid B8. reduceRegions(mean),10m. No labels or target outcomes used, exact keys all7220 trees. NaNs zero, duplicates zero. Acquisition is later than June and may encode seasonal changes and persistence; label timing was not supplied so no early-detection/causal claim. Reproduction extract_autumn.py.
''');shutil.copy2(__file__,root/'extract_autumn.py');print(res.shape,flush=True)
