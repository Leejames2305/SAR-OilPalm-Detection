"""Public Meta/WRI fine-scale historical canopy covariates, no labels used."""
from pathlib import Path
import os,json,time
import ee,numpy as np,pandas as pd
ROOT=Path('/app/datasets/highres_canopy');ROOT.mkdir(exist_ok=True)
ee.Initialize(ee.ServiceAccountCredentials(None,os.environ['GOOGLE_APPLICATION_CREDENTIALS']),project=os.environ['GCP_PROJECT'])
d=pd.concat([pd.read_csv('/app/data/L_labelled.csv',usecols=['estate','Long','Lat']),
             pd.read_csv('/app/data/unlabelled.csv',usecols=['estate','Long','Lat'])],ignore_index=True)
pieces=[];audit={}
for estate,a in d.groupby('estate',sort=False):
    cache=ROOT/f'{estate}.parquet'
    if cache.exists():pieces.append(pd.read_parquet(cache));continue
    roi=ee.Geometry.Rectangle([a.Long.min()-.002,a.Lat.min()-.002,a.Long.max()+.002,a.Lat.max()+.002])
    col=ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').filterBounds(roi)
    # Preserve documented native ~1.194m Mercator grid; focal kernels in metres.
    proj=ee.Image(col.first()).projection()
    h=col.mosaic().setDefaultProjection(proj).rename('height').toFloat()
    images=[h]
    for r in [3,6,12,24]:
        ker=ee.Kernel.circle(r,'meters')
        images.extend([h.reduceNeighborhood(ee.Reducer.mean(),ker).rename(f'height_mean{r}'),
                       h.reduceNeighborhood(ee.Reducer.stdDev(),ker).rename(f'height_sd{r}'),
                       h.reduceNeighborhood(ee.Reducer.percentile([90]),ker).rename(f'height_p90_{r}')])
    for cut in [2,5,10]:
        for r in [6,12,24]:
            images.append(h.gt(cut).reduceNeighborhood(ee.Reducer.mean(),ee.Kernel.circle(r,'meters')).rename(f'cover{cut}_{r}'))
    im=ee.Image.cat(images)
    for r in [6,12]:
        images.append(im.select(f'height_mean{r}').subtract(im.select('height_mean24')).rename(f'height_contrast{r}'))
    im=ee.Image.cat(images)
    cols=im.bandNames().getInfo();rows=[]
    for start in range(0,len(a),160):
        chunk=a.iloc[start:start+160]
        # Temporary query offset only, removed before feature-table serialization.
        fc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(v.Long),float(v.Lat)]),{'query_offset':int(ix)}) for ix,v in chunk.iterrows()])
        vals=im.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale=1.1943285669558747,crs=proj,tileScale=4).getInfo()['features']
        for f in vals:
            p=f['properties'];ix=int(p.pop('query_offset'));rows.append({'query_offset':ix,**{c:p.get(c,np.nan) for c in cols}})
        print(estate,start+len(chunk),'/',len(a),flush=True)
    tab=pd.DataFrame(rows).set_index('query_offset').reindex(a.index)
    result=pd.concat([a,tab],axis=1)
    audit[estate]={'nan':int(tab.isna().sum().sum()),'min':float(tab.min().min()),'max':float(tab.max().max()),'projection':proj.getInfo()}
    assert np.isfinite(tab.to_numpy(float)).all()
    result.to_parquet(cache,index=False);pieces.append(result)
out=pd.concat(pieces).sort_index()
assert len(out)==7220 and not out.duplicated(['estate','Long','Lat']).any()
out.to_parquet(ROOT/'features.parquet',index=False)
(ROOT/'extraction.json').write_text(json.dumps(audit,indent=2))
print('DONE',out.shape,flush=True)
