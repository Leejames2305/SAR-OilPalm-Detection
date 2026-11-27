"""Cloud-controlled native-grid optical mosaics, exact georeferencing audited.
SCL4-7 preserves exposed soil/water while removing cloud/shadow. No labels used.
"""
import os,json
from pathlib import Path
import ee,numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
from affine import Affine

ROOT=Path('/app/research/native_optical');ROOT.mkdir(exist_ok=True)
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
keys=pd.concat([pd.read_csv('/app/data/L_labelled.csv',usecols=['estate','Long','Lat']),pd.read_csv('/app/data/unlabelled.csv',usecols=['estate','Long','Lat'])],ignore_index=True)
bands=['B2','B3','B4','B8','B5','B8A','B11','B12']
for estate,d in keys.groupby('estate',sort=False):
    path=ROOT/f'{estate}.tif'
    if path.exists():continue
    xs,ys=transform('EPSG:4326','EPSG:32648',d.Long.tolist(),d.Lat.tolist())
    xmin=np.floor((min(xs)-120)/10)*10;ymax=np.ceil((max(ys)+120)/10)*10
    xmax=np.ceil((max(xs)+120)/10)*10;ymin=np.floor((min(ys)-120)/10)*10
    roi=ee.Geometry.Rectangle([float(xmin),float(ymin),float(xmax),float(ymax)],'EPSG:32648',False)
    proj=ee.Projection('EPSG:32648',[10,0,float(xmin),0,-10,float(ymax)])
    cs=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2026-04-15','2026-08-15').linkCollection(cs,['cs_cdf'])
    def strict(im):
        scl=im.select('SCL');return im.select(bands).updateMask(scl.gte(4).And(scl.lte(7)).And(im.select('cs_cdf').gte(.65)))
    def fallback(im):
        scl=im.select('SCL');return im.select(bands).updateMask(scl.gte(4).And(scl.lte(7)))
    im=col.map(strict).median().unmask(col.map(fallback).median()).reproject(proj).toFloat()
    im=im.addBands(ee.Image.pixelCoordinates(proj).rename(['px','py']))
    prop=im.sampleRectangle(region=roi,defaultValue=-9999).getInfo()['properties']
    arr=np.array([prop[b] for b in bands],np.float32)/10000
    px=np.array(prop['px']);py=np.array(prop['py'])
    # EE pixelCoordinates returns center coordinates (.5,.5) for pixel0.
    # Convert those centers to GeoTIFF upper-left CORNER coordinates.
    # Boundary probe archived in research/georeference_audit.json.
    trans=Affine(10,0,float(xmin+px.min()*10-5),0,-10,float(ymax-py.min()*10+5))
    with rasterio.open(path,'w',driver='GTiff',width=arr.shape[2],height=arr.shape[1],count=len(bands),dtype='float32',crs='EPSG:32648',transform=trans,compress='deflate') as dst:
        dst.write(arr);dst.descriptions=tuple(bands)
    (ROOT/f'{estate}_metadata.json').write_text(json.dumps({'shape':arr.shape,'affine':list(trans),'projection':proj.getInfo(),'bands':bands,'missing':int((arr<0).sum())},indent=2))
    print(estate,arr.shape,'negative',(arr<0).sum(),flush=True)
