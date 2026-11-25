"""Reproducible correctly masked per-tree L-band power/texture covariates."""
from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform
from rasterio.transform import rowcol

ROOT=Path('/app'); OUT=ROOT/'datasets/alos_physics';OUT.mkdir(parents=True,exist_ok=True)
d=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv').drop(columns='label'),pd.read_csv(ROOT/'data/unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']]
assert len(d)==7220 and not d.duplicated().any()
out={c:d[c].to_numpy() for c in d}
audit=[]
for estate in sorted(d.estate.unique()):
    ix=np.flatnonzero(d.estate.eq(estate)); frame=d.iloc[ix]
    for date in ('260610','261014'):
        spath=ROOT/'rasters'/f'ALOS2-Subset_{estate}_{date}_Cal_Spk_TC.tif'
        with rasterio.open(spath) as s:
            back=s.read().astype(float); support=np.isfinite(back).all(axis=0)&(back>0).all(axis=0)
            sp_meta=(s.transform,s.crs,support)
        for family,suffix,bands in [('power','',('HH','HV','VH','VV')),('ha','mat_HAlpha_',('H','A','alpha')),('yam','mat_Yamaguchi_',('Pd','Pv','Ps','Pc'))]:
            path=ROOT/'rasters'/f'ALOS2-Subset_{estate}_{date}_Cal_{suffix}Spk_TC.tif'
            with rasterio.open(path) as src:
                arr=src.read().astype(float)
                xx,yy=transform('EPSG:4326',src.crs,frame.Long.to_numpy(),frame.Lat.to_numpy())
                rr,cc=rowcol(src.transform,xx,yy)
                if family=='yam':
                    roi=np.zeros(src.shape,dtype=np.uint8)
                    reproject(sp_meta[2].astype(np.uint8),roi,src_transform=sp_meta[0],src_crs=sp_meta[1],dst_transform=src.transform,dst_crs=src.crs,resampling=Resampling.nearest)
                    vm=np.isfinite(arr).all(axis=0)&(roi>0)
                    # Convert dB per pixel before averaging and normalizing power.
                    lin=np.power(10,arr/10)
                    fractions=lin/np.maximum(lin.sum(axis=0,keepdims=True),1e-30)
                    stacks=[(bands[k]+'_db',arr[k]) for k in range(4)]+[(bands[k]+'_frac',fractions[k]) for k in range(4)]
                elif family=='power':
                    vm=np.isfinite(arr).all(axis=0)&(arr>0).all(axis=0)
                    stacks=[(bands[k]+'_db',10*np.log10(np.maximum(arr[k],1e-30))) for k in range(4)]
                    stacks += [('span_db',10*np.log10(np.maximum(arr[0]+arr[3]+arr[1]+arr[2],1e-30)))]
                else:
                    vm=np.isfinite(arr).all(axis=0)&(arr>0).all(axis=0)
                    stacks=list(zip(bands,arr))
                audit.append(dict(estate=estate,date=date,family=family,supported_pixels=int(vm.sum()),total_pixels=int(vm.size)))
                for bname,band in stacks:
                    for r in (1,2,4):
                        result=np.full((len(ix),4),np.nan)
                        for j,(row,col) in enumerate(zip(rr,cc)):
                            r0=max(0,row-r);r1=min(src.height,row+r+1);c0=max(0,col-r);c1=min(src.width,col+r+1)
                            values=band[r0:r1,c0:c1][vm[r0:r1,c0:c1]]
                            if len(values)>=5:
                                q=np.percentile(values,[10,50,90])
                                result[j]=[values.mean(),values.std(),q[2]-q[0],q[1]]
                        for k,stat in enumerate(('mean','std','range80','median')):
                            name=f'{family}_{bname}_{date}_r{r}_{stat}'
                            if name not in out:out[name]=np.full(len(d),np.nan)
                            out[name][ix]=result[:,k]
        print(estate,date,flush=True)
f=pd.DataFrame(out)
# Matched date changes and crown-neighborhood contrasts; no label is involved.
extras={}
for col in f:
    if '_260610_' in col:
        b=col.replace('_260610_','_261014_')
        if b in f:extras[col.replace('_260610_','_change_')]=f[b]-f[col]
    if '_r1_' in col and col.endswith('_mean'):
        b=col.replace('_r1_','_r4_')
        extras[col.replace('_r1_','_contrast_')]=f[col]-f[b]
f=pd.concat([f,pd.DataFrame(extras)],axis=1)
f.to_parquet(OUT/'features.parquet',index=False)
(OUT/'audit.json').write_text(json.dumps({'shape':list(f.shape),'nan_total':int(f.isna().sum().sum()),'nan_by_estate':{e:int(f.loc[d.estate.eq(e)].isna().sum().sum()) for e in d.estate.unique()},'raster_support':audit},indent=2))
if Path(__file__).resolve() != (OUT/'extract.py').resolve():
    shutil.copy(__file__,OUT/'extract.py')
(OUT/'MANIFEST.md').write_text('''# Physically scaled ALOS covariates

Source: the 18 supplied backscatter, HAlpha, and Yamaguchi GeoTIFFs for 2026-06-10 and 2026-10-14. Exact WGS84 coordinates from the union of the two public CSVs are reprojected to each raster CRS before sampling. All 7,220 unique `(estate, Long, Lat)` keys are preserved; labels and row identifiers are absent.

Backscatter is finite positive linear power transformed to 10*log10(power) per pixel. HAlpha retains original units and requires finite positive support. Yamaguchi accepts negative dB and is cross-masked by finite positive Cal_Spk_TC support reprojected with nearest-neighbor resampling. Pd/Pv/Ps/Pc fractions are computed per pixel as 10**(db/10) divided by their sum, before averaging. No invalid zero power is logged as a valid observation.

Radius 1, 2, and 4 square windows (3, 5, 9 pixels at 6.43 m) supply mean, standard deviation, 90th minus 10th percentile, and median, requiring at least five valid pixels. Matched-date changes are October minus June. Crown-neighborhood contrast is radius-1 mean minus radius-4 mean, without changing the physical sign. Features contain no labels and no fitted supervised transformations. `audit.json` records shape, NaN counts, and masking support; any missing values must be imputed using training data only. Reproduce with `python /app/datasets/alos_physics/extract.py` in the development environment (numpy, pandas, rasterio).
''')
print(f.shape,'NaNs',f.isna().sum().sum(),flush=True)
