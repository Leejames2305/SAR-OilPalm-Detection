"""October-aligned Sentinel-2 physiology and local canopy contrast, all estates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,json,time,shutil
import numpy as np
import pandas as pd
import ee

ROOT=Path('/app');OUT=ROOT/'datasets/gee_s2_autumn';OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'research/s2_autumn_cache';CACHE.mkdir(parents=True,exist_ok=True)
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
d=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv').drop(columns='label'),pd.read_csv(ROOT/'data/unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']]
BANDS=['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
def mask(im):
    scl=im.select('SCL'); return im.updateMask(scl.gte(4).And(scl.lte(7)))

def image(g):
    reg=ee.Geometry.Rectangle([g.Long.min()-.002,g.Lat.min()-.002,g.Long.max()+.002,g.Lat.max()+.002])
    col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(reg).filterDate('2026-09-14','2026-11-14').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',80)).map(mask)
    count=col.size().getInfo(); print('image_count',g.estate.iloc[0],count,flush=True)
    if count==0:raise RuntimeError('No October-aligned scenes are available')
    b=col.select(BANDS).median().setDefaultProjection(ee.Projection('EPSG:32648').atScale(10))
    def norm(a,c):return b.select(a).subtract(b.select(c)).divide(b.select(a).add(b.select(c)).add(1e-6))
    phys=norm('B8','B4').rename('NDVI').addBands(norm('B8A','B5').rename('NDRE1')).addBands(norm('B8A','B6').rename('NDRE2')).addBands(norm('B8','B11').rename('NDMI')).addBands(norm('B8','B12').rename('NBR'))
    out=b.addBands(phys).addBands(col.select('B4').count().rename('clear_count'))
    p=phys.select(['NDVI','NDRE1','NDMI'])
    # Define neighbourhoods explicitly in metres on a 10 m UTM analysis grid.
    med=p.focalMedian(radius=10,units='meters',kernelType='square')
    mx=p.focalMax(radius=10,units='meters',kernelType='square')
    bg=p.focalMedian(radius=30,units='meters',kernelType='square')
    out=out.addBands(med.rename(['NDVI_med10','NDRE1_med10','NDMI_med10']))
    out=out.addBands(mx.rename(['NDVI_max10','NDRE1_max10','NDMI_max10']))
    out=out.addBands(p.subtract(bg).rename(['NDVI_contrast30','NDRE1_contrast30','NDMI_contrast30']))
    return out

def pull(args):
    estate,im,start,g=args; cache=CACHE/f'{estate}_{start}.json'
    if cache.exists():return json.loads(cache.read_text())
    fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'sample_index':int(j)}) for j,r in g.iterrows()]
    for attempt in range(4):
        try:
            got=im.reduceRegions(ee.FeatureCollection(fs),ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']
            props=[z['properties'] for z in got];cache.write_text(json.dumps(props));print(estate,start,len(props),flush=True);return props
        except Exception as err:
            if attempt==3:raise
            print('retry',estate,start,type(err).__name__,flush=True);time.sleep(2*(attempt+1))

rows=[]
for estate,g in d.groupby('estate',sort=False):
    im=image(g)
    jobs=[(estate,im,start,g.iloc[start:start+400]) for start in range(0,len(g),400)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for props in pool.map(pull,jobs):rows.extend(props)
rows=sorted(rows,key=lambda x:x['sample_index'])
assert [r['sample_index'] for r in rows]==list(range(len(d)))
f=pd.DataFrame(rows).drop(columns='sample_index');f=pd.concat([d.reset_index(drop=True),f.reset_index(drop=True)],axis=1)
assert len(f)==7220 and not f[['estate','Long','Lat']].duplicated().any()
f.to_parquet(OUT/'features.parquet',index=False)
(OUT/'audit.json').write_text(json.dumps({'shape':list(f.shape),'nan_total':int(f.isna().sum().sum()),'nan_by_feature':f.isna().sum().to_dict(),'clear_count_by_estate':f.groupby('estate').clear_count.describe().to_dict()},indent=2))
if Path(__file__).resolve() != (OUT/'extract.py').resolve():
    shutil.copy(__file__,OUT/'extract.py')
(OUT/'MANIFEST.md').write_text('''# October-aligned Sentinel-2 canopy physiology

Source: Google Earth Engine COPERNICUS/S2_SR_HARMONIZED, 2026-09-14 inclusive to 2026-11-14 exclusive, covering all three estates. CLOUDY_PIXEL_PERCENTAGE < 80 and SCL 4 through 7 retained, matching supplied data conventions. Harmonized collection is used directly with no additional 1000 DN offset. Median SR bands B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12 remain in raw DN units. NDVI=(B8-B4)/(B8+B4); NDRE1=(B8A-B5)/(B8A+B5); NDRE2=(B8A-B6)/(B8A+B6); NDMI=(B8-B11)/(B8+B11); NBR=(B8-B12)/(B8+B12). Bandwise median composite, then indices, not median of scene indices.

Analysis grid EPSG:32648 at 10 m; native 20 m red-edge/SWIR bands retain their original information limit. Nearest-neighbour resampling is Earth Engine's default; no spatial super-resolution is claimed. NDVI/NDRE1/NDMI also have square radius-10 m median and maximum, plus centre minus radius-30 m median. Grid projection is explicit before focal filtering. Clear observation count is exported for quality auditing.

All 7,220 exact keys from public L/unlabelled CSV union are preserved in their original order. No supervised extraction or imputation is performed. Missing values and per-estate quality summaries are in audit.json; training-only imputation is required if needed. Extraction uses a service-account path from the environment but never prints or writes credentials. Run extract.py in the development environment with earthengine-api, pandas and pyarrow; cached API response chunks are optional and contain only covariates and temporary sequence numbers, absent from the final table.
''')
print(f.shape,'NaNs',int(f.isna().sum().sum()),flush=True)
