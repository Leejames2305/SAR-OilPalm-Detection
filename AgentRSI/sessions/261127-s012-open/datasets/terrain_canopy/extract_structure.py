"""Independent terrain and historical canopy structure; no label use."""
import os,json,time
from pathlib import Path
import numpy as np,pandas as pd,ee
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
root=Path('/app/datasets/terrain_canopy');root.mkdir(exist_ok=True)
d=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']]
height=ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1');print('canopy bands',height.bandNames().getInfo(),flush=True)
out=[]
for estate,a in d.groupby('estate',sort=False):
    roi=ee.Geometry.Rectangle([float(a.Long.min()-.01),float(a.Lat.min()-.01),float(a.Long.max()+.01),float(a.Lat.max()+.01)])
    dem=ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(roi).select('DEM').mosaic().reproject('EPSG:32648',None,30)
    terrain=ee.Terrain.products(dem)
    aspect=terrain.select('aspect').multiply(np.pi/180)
    im=dem.rename('elevation').addBands(terrain.select('slope'))
    im=im.addBands(aspect.cos().rename('aspect_north')).addBands(aspect.sin().rename('aspect_east'))
    for r in [60,120]:
        im=im.addBands(dem.subtract(dem.focalMean(r,'circle','meters')).rename(f'tpi{r}'))
        im=im.addBands(dem.reduceNeighborhood(ee.Reducer.stdDev(),ee.Kernel.circle(r,'meters')).rename(f'relief{r}'))
    h=height.select(0).reproject('EPSG:32648',None,10)
    im=im.addBands(h.rename('height2020'))
    im=im.addBands(h.subtract(h.focalMean(40,'circle','meters')).rename('height_contrast40'))
    im=im.addBands(h.reduceNeighborhood(ee.Reducer.stdDev(),ee.Kernel.circle(40,'meters')).rename('height_sd40'))
    pts=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]),{'joinseq':int(i)}) for i,r in a.iterrows()])
    data=im.reduceRegions(pts,ee.Reducer.mean(),scale=10,crs='EPSG:32648',tileScale=4).getInfo()['features']
    vals=pd.DataFrame([f['properties'] for f in data]).set_index('joinseq').reindex(a.index)
    for k in vals:a=a.assign(**{k:vals[k].to_numpy()})
    out.append(a);print(estate,'done',a.shape,flush=True)
res=pd.concat(out).reindex(d.index);res.to_parquet(root/'features.parquet',index=False)
print('result',res.shape,'missing',res.isna().sum().to_dict(),flush=True)
(root/'MANIFEST.md').write_text('''# Independent terrain and historical canopy structure

Source: Google Earth Engine COPERNICUS/DEM/GLO30 (30m digital surface model) and users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1 (10m 2020 canopy-height estimate). The elevation mosaic is projected to EPSG:32648 at30m before terrain derivatives. Slope in degrees, aspect as sine/cosine, topographic position elevation minus circular focal mean60/120m, relief as circular standard deviation60/120m. Terrain is a surface model and includes vegetation effects; it is not bare-earth terrain or a measured disease indicator. Height first source band in metres, raw/contrast to40m circular mean/SD40m. No labels used. reduceRegions(mean) at10m, EPSG32648; terrain remains native30m information. No offset/harmonisation. 2020 height predates current screening and can encode stand age/management.

Exact source keys for all7220 trees; zero duplicate keys. Extraction script extract_structure.py; missing summary in extraction log and report audit.
''')
import shutil;shutil.copy2(__file__,root/'extract_structure.py')
