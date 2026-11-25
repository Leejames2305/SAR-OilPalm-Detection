"""Extract leak-free Sentinel-2 stress covariates for all supplied tree coordinates."""
from pathlib import Path
import os, time
import pandas as pd, numpy as np
import ee

PROJECT=os.environ.get('GCP_PROJECT')
cred=ee.ServiceAccountCredentials(None, os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
ee.Initialize(cred, project=PROJECT)
ROOT=Path('/app')
L=pd.read_csv(ROOT/'data/L_labelled.csv'); U=pd.read_csv(ROOT/'data/unlabelled.csv')
D=pd.concat([L.drop(columns=['label']),U],ignore_index=True).drop_duplicates(['estate','Long','Lat'])
# SCL classes 4 vegetation, 5 bare, 6 water, 7 unclassified: retain 4-7 as in source spec.
def mask(img):
    scl=img.select('SCL')
    return img.updateMask(scl.gte(4).And(scl.lte(7)))
def composite(start,end,region):
    c=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start,end).filterBounds(region)
       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',80)).map(mask))
    # reflectance is scaled DN in this collection; ratios are scale invariant.
    im=c.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']).median()
    b=im
    eps=ee.Image.constant(1e-6)
    out=b.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
    def norm(a,z): return b.select(a).subtract(b.select(z)).divide(b.select(a).add(b.select(z)).add(eps))
    out=out.addBands(norm('B8A','B5').rename('NDRE1'))
    out=out.addBands(norm('B8A','B6').rename('NDRE2'))
    out=out.addBands(norm('B8','B11').rename('NDMI'))
    out=out.addBands(norm('B8','B12').rename('NBR'))
    out=out.addBands(b.select('B8A').divide(b.select('B5').add(eps)).subtract(1).rename('CI_rededge'))
    out=out.addBands(b.select('B11').divide(b.select('B8').add(eps)).rename('MSI'))
    out=out.addBands(b.select('B8').subtract(b.select('B4')).multiply(2.5).divide(b.select('B8').add(b.select('B4').multiply(6)).subtract(b.select('B2').multiply(7.5)).add(10000).add(eps)).rename('EVI'))
    return out

def region_for(df):
    return ee.Geometry.Rectangle([float(df.Long.min())-.002,float(df.Lat.min())-.002,float(df.Long.max())+.002,float(df.Lat.max())+.002])

def pull(df, im, chunk=300):
    feats=[]
    for i in range(0,len(df),chunk):
        sub=df.iloc[i:i+chunk]
        fs=[ee.Feature(ee.Geometry.Point([float(r.Long),float(r.Lat)]), {'row':int(j)}) for j,r in sub.iterrows()]
        fc=ee.FeatureCollection(fs).map(lambda f:f.set('Long',f.geometry().coordinates().get(0)).set('Lat',f.geometry().coordinates().get(1)))
        x=im.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale=20,tileScale=4)
        got=x.getInfo()['features']
        for f in got:
            p=f.get('properties',{}); feats.append((p.get('row'),p))
        print('chunk',i,len(got),flush=True)
    feats.sort(key=lambda z:z[0]); return [p for _,p in feats]

allout=[]
for estate,g in D.groupby('estate',sort=False):
    print('estate',estate,len(g),flush=True)
    reg=region_for(g)
    im26=composite('2026-05-15','2026-07-15',reg)
    # A prior-year matched-season composite supplies an independent seasonal delta.
    im25=composite('2025-05-15','2025-07-15',reg)
    p26=pull(g.reset_index(drop=True),im26); p25=pull(g.reset_index(drop=True),im25)
    for a,b in zip(p26,p25):
        q={k:v for k,v in a.items() if k not in ('row','Long','Lat')}
        for k,v in b.items():
            if k not in ('row','Long','Lat'): q['prev_'+k]=v
        q['estate']=estate; q['Long']=a.get('Long'); q['Lat']=a.get('Lat'); allout.append(q)
out=pd.DataFrame(allout)
# restore exact source coordinates and row order; no forbidden names.
out= D[['estate','Long','Lat']].merge(out,on=['estate','Long','Lat'],how='left',sort=False)
# derive robust temporal changes and within-feature finite values.
for c in [x for x in out.columns if x.startswith('B') or x in ['NDRE1','NDRE2','NDMI','NBR','CI_rededge','MSI','EVI']]:
    pc='prev_'+c
    if pc in out: out['delta_'+c]=out[c]-out[pc]
out.to_parquet(Path(__file__).parent/'features.parquet',index=False)
print(out.shape, 'nan',int(out.isna().sum().sum()), 'bytes', (Path(__file__).parent/'features.parquet').stat().st_size)
