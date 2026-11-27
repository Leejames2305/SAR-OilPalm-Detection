"""Label-free, native-CRS two-date SAR descriptors and masked-AE inputs.

Run from any directory. Only exact coordinate keys are read from input tables.
Yamaguchi input is dB: finite negatives are retained, support from power rasters.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform,reproject,Resampling
from scipy.ndimage import uniform_filter

ROOT = Path('/app/datasets/sar_structure')
ROOT.mkdir(exist_ok=True)
IO = Path('/run/agentrsi/colab-io/in')
IO.mkdir(parents=True, exist_ok=True)
keys = pd.concat([pd.read_csv('/app/data/L_labelled.csv', usecols=['estate','Long','Lat']),
                  pd.read_csv('/app/data/unlabelled.csv', usecols=['estate','Long','Lat'])], ignore_index=True)
names = ['HH','HV','VH','VV','H','A','alpha','Pd','Pv','Ps','Pc']
patches = np.zeros((len(keys),22,12,12), np.float32)
features = {}
support_summary = {}
for estate, d in keys.groupby('estate', sort=False):
    all_dates = []
    for j, date in enumerate(['260610','261014']):
        pref = f'/app/rasters/ALOS2-Subset_{estate}_{date}'
        with rasterio.open(pref+'_Cal_Spk_TC.tif') as src:
            p = src.read().astype(float)
            xs,ys = transform('EPSG:4326',src.crs,d.Long.tolist(),d.Lat.tolist())
            rc = np.array([src.index(x,y) for x,y in zip(xs,ys)])
            support = np.all(np.isfinite(p)&(p>0),axis=0)
            # Authoritative power geometry; HAlpha and Yamaguchi verified below.
            geom = (src.crs,src.transform,src.shape)
        with rasterio.open(pref+'_Cal_mat_HAlpha_Spk_TC.tif') as src:
            assert geom == (src.crs,src.transform,src.shape)
            h = src.read().astype(float)
        with rasterio.open(pref+'_Cal_mat_Yamaguchi_Spk_TC.tif') as src:
            assert geom == (src.crs,src.transform,src.shape)
            yam = src.read().astype(float)
        p[~np.isfinite(p)|(p<=0)] = np.nan
        h[~np.isfinite(h)|(h<=0)] = np.nan
        yam[:,~support] = np.nan
        yam[~np.isfinite(yam)] = np.nan
        if j == 0:
            june_geom = geom
        else:
            # October grids differ by a pixel in width/origin. Reproject physical
            # powers to the June grid before temporal subtraction/stacking.
            def align(values):
                dest=np.full((len(values),*june_geom[2]),np.nan)
                for k in range(len(values)):
                    reproject(values[k],dest[k],src_transform=geom[1],src_crs=geom[0],
                              dst_transform=june_geom[1],dst_crs=june_geom[0],
                              src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
                return dest
            p=align(p);h=align(h);yam=10*np.log10(align(10**(yam/10)))
            support=np.all(np.isfinite(p)&(p>0),axis=0)
            rr,cc=rasterio.transform.rowcol(june_geom[1],xs,ys)
            rc=np.column_stack([rr,cc])
        z = np.concatenate([10*np.log10(p),h,yam])
        # Fixed broad physical clipping; retain negative decomposition powers.
        z[:4] = np.clip(z[:4],-55,15)
        z[7:] = np.clip(z[7:],-65,15)
        all_dates.append(z)
        estate_support=[]
        for n,(r,c) in zip(d.index,rc):
            estate_support.append(support[r-6:r+6,c-6:c+6].mean())
            chip = z[:,r-6:r+6,c-6:c+6].copy()
            assert chip.shape == (11,12,12)
            for k in range(11):
                bad=~np.isfinite(chip[k])
                if bad.any():
                    chip[k,bad]=np.nanmedian(z[k])
            patches[n,j*11:(j+1)*11]=chip
        support_summary[estate+'_'+date] = {'min':float(min(estate_support)), 'mean':float(np.mean(estate_support))}
        # Compact pixel statistics across <=71m windows; no labels or neighbors.
        for w in [3,7,11]:
            vals=[]
            for r,c in rc:
                cut=z[:,r-w//2:r+w//2+1,c-w//2:c+w//2+1]
                vals.append(np.stack([np.nanmedian(cut,axis=(1,2)),np.nanstd(cut,axis=(1,2)),
                            np.nanpercentile(cut,10,axis=(1,2)),np.nanpercentile(cut,90,axis=(1,2))],axis=1).ravel())
            vals=np.array(vals)
            for k,name in enumerate(names):
                for t,stat in enumerate(['median','sd','q10','q90']):
                    col=f'{date}_{name}_{stat}{w}'
                    if col not in features: features[col]=np.full(len(keys),np.nan)
                    features[col][d.index]=vals[:,k*4+t]
    # Same-pixel calibrated changes, not subtraction of nonlinear display RGB.
    delta=all_dates[1]-all_dates[0]
    for w in [3,7]:
        vals=[]
        for r,c in rc:
            cut=delta[:,r-w//2:r+w//2+1,c-w//2:c+w//2+1]
            vals.append(np.stack([np.nanmedian(cut,axis=(1,2)),np.nanstd(cut,axis=(1,2))],axis=1).ravel())
        vals=np.array(vals)
        for k,name in enumerate(names):
            for t,stat in enumerate(['median','sd']):
                col=f'delta_{name}_{stat}{w}'
                if col not in features: features[col]=np.full(len(keys),np.nan)
                features[col][d.index]=vals[:,k*2+t]
    print(estate,'extracted',flush=True)
out=pd.concat([keys,pd.DataFrame(features)],axis=1)
assert np.isfinite(out.iloc[:,3:].to_numpy()).all()
out.to_parquet(ROOT/'features.parquet',index=False)
keys.to_parquet('/app/research/sar_keys.parquet',index=False)
# Label-free global physical normalization, parameters saved with GPU checkpoint.
mean=patches.mean(axis=(0,2,3),keepdims=True)
sd=patches.std(axis=(0,2,3),keepdims=True).clip(.0001)
patches=np.clip((patches-mean)/sd,-6,6)
np.savez_compressed(IO/'native_sar.npz',patches=patches,mean=mean,sd=sd)
(ROOT/'extraction.json').write_text(json.dumps({'support':support_summary,'shape':out.shape,'nan':int(out.isna().sum().sum()),'patch_shape':patches.shape},indent=2))
(ROOT/'MANIFEST.md').write_text('''# Native two-date SAR spatial descriptors

ALOS PALSAR-2 June10 and October14 2026, EPSG:32648, 6.43m pixels.
Exact WGS84 keys reprojected before trunk-containing pixel indexing. All raster
geometries checked identical per estate/date. Power bands HH/HV/VH/VV linear
positive finite ->10log10 dB. HAlpha H/A/alpha finite positive. Yamaguchi
Pd/Pv/Ps/Pc already dB, all finite negatives retained; cross-mask power ROI.
Broad physical clipping power[-55,15], decomposition[-65,15]dB.
3/7/11-pixel windows (~19/45/71m) median/SD/10th/90th percentile.
October-minus-June same-pixel median and SD for3/7 windows. No labels used.
All7220 exact unique keys, zeroNaNs. extract_sar.py reproduces features and
22-channel77m patch input for separate label-free self-supervised experiment.
Overlap across adjacent blocks is possible: require coarser/buffered checks.
''')
print(out.shape,support_summary,flush=True)
