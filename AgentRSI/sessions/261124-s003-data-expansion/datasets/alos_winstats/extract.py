from pathlib import Path
import glob, re
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform as reproj
from rasterio.transform import rowcol
ROOT=Path('/app'); OUT=ROOT/'datasets'/'alos_winstats'; OUT.mkdir(parents=True,exist_ok=True)
allf=pd.concat([pd.read_csv(ROOT/'data'/'L_labelled.csv').drop(columns=['label']),pd.read_csv(ROOT/'data'/'unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']].drop_duplicates().reset_index(drop=True)
out=allf.copy()
def family(path):
 n=Path(path).name; m=re.match(r'ALOS2-Subset_(.+?)_(260610|261014)_Cal_(.*)Spk_TC\.tif$',n)
 if not m:return None
 est,date,s=m.groups(); kind={'':'spk','mat_':'t3','mat_HAlpha_':'ha','mat_Yamaguchi_':'yam'}.get(s)
 return (est,date,kind) if kind else None
for path in sorted(glob.glob(str(ROOT/'rasters'/'*.tif'))):
 info=family(path)
 if not info: continue
 est,date,kind=info; sub=out.estate.eq(est); idx=np.flatnonzero(sub.to_numpy())
 if not len(idx): continue
 with rasterio.open(path) as src:
  xs0=out.loc[sub,'Long'].to_numpy(float); ys0=out.loc[sub,'Lat'].to_numpy(float)
  xs,ys=reproj('EPSG:4326',src.crs,xs0,ys0); rr,cc=rowcol(src.transform,xs,ys); arr=src.read()
  if kind in ('spk','ha'): valid=np.isfinite(arr)&(arr>0)
  elif kind=='t3': valid=np.isfinite(arr)&(arr[0:1]>0)
  else: valid=np.isfinite(arr)
  prefix=f'rs_{date}_{kind}'
  for b in range(arr.shape[0]):
   vals=arr[b]
   for rad in (1,2):
    stats=[]
    for r,c in zip(rr,cc):
     r0=max(0,r-rad);r1=min(vals.shape[0],r+rad+1);c0=max(0,c-rad);c1=min(vals.shape[1],c+rad+1)
     z=vals[r0:r1,c0:c1].astype(float); vm=valid[b,r0:r1,c0:c1]; z=z[vm]
     stats.append([np.mean(z),np.std(z),np.min(z),np.max(z),np.percentile(z,10),np.percentile(z,90),float(z.size)] if z.size>=3 else [np.nan]*7)
    stats=np.asarray(stats)
    for j,sn in enumerate(('mean','std','min','max','p10','p90','n')): out.loc[idx,f'{prefix}_b{b+1}_r{rad}_{sn}']=stats[:,j]
for fam in ('spk','ha','yam'):
 for b in range(1,10):
  a=f'rs_260610_{fam}_b{b}_r1_mean'; z=f'rs_261014_{fam}_b{b}_r1_mean'
  if a in out and z in out:
   out[f'rs_delta_{fam}_b{b}']=out[z]-out[a]; out[f'rs_ratio_{fam}_b{b}']=out[z]/(np.abs(out[a])+1e-6)
out.to_parquet(OUT/'features.parquet',index=False)
OUT.joinpath('MANIFEST.md').write_text('''# ALOS window statistics\n\nExtracted from `/app/rasters` with explicit filename mapping: `spk`=Cal_Spk_TC linear backscatter, `ha`=Cal_mat_HAlpha_Spk_TC, `yam`=Cal_mat_Yamaguchi_Spk_TC signed dB, and `t3`=AirHitam Cal_mat_Spk_TC coherency. Coordinates are reprojected from EPSG:4326 to each native raster CRS before radius-1/radius-2 sampling. Power/HAlpha use finite positive masks; T3 uses finite positive T11 support; Yamaguchi uses finite signed values. Each band stores mean, std, min, max, p10, p90, and valid count. Matched-date deltas and ratios are stored for spk, ha, and yam.\n''')
print(out.shape, out.isna().sum().sum())
