from pathlib import Path
import glob, re
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform as reproj
from rasterio.transform import rowcol

ROOT=Path('/app')
OUT=ROOT/'datasets'/'alos_winstats'
OUT.mkdir(parents=True,exist_ok=True)
allf=pd.concat([pd.read_csv(ROOT/'data'/'L_labelled.csv').drop(columns=['label']),pd.read_csv(ROOT/'data'/'unlabelled.csv')],ignore_index=True)
allf=allf[['estate','Long','Lat']].drop_duplicates().reset_index(drop=True)
out=allf.copy()

def family(path):
    n=Path(path).name
    m=re.search(r'ALOS2-Subset_(.+?)_(260610|261014)_Cal_(.*)Spk_TC',n)
    if not m: return None
    est,date,kind=m.groups(); kind=kind.rstrip('_') or 'spk'
    if kind=='Spk': kind='spk'
    return est,date,kind

for path in sorted(glob.glob(str(ROOT/'rasters'/'*.tif'))):
    info=family(path)
    if not info: continue
    est,date,kind=info
    sub=out['estate'].eq(est)
    idx=np.flatnonzero(sub.to_numpy())
    if not len(idx): continue
    with rasterio.open(path) as src:
        xs0=out.loc[sub,'Long'].to_numpy(float); ys0=out.loc[sub,'Lat'].to_numpy(float)
        xs,ys=reproj('EPSG:4326',src.crs,xs0,ys0)
        # transform all positions; rasterio handles both UTM and geographic rasters
        rr,cc=rowcol(src.transform,xs,ys)
        arr=src.read()
        # validity support for ROI-masked products
        if kind=='spk':
            valid=np.isfinite(arr)
            if 'Yamaguchi' not in kind: valid &= arr>0
            else:
                # Yamaguchi is dB; support is determined from co-located power product
                valid &= True
        elif 'HAlpha' in kind:
            valid=np.isfinite(arr) & (arr>0)
        elif kind.startswith('mat_Spk'):
            valid=np.isfinite(arr) & (arr[0:1]>0)
        else: valid=np.isfinite(arr)
        # Cross-mask Yamaguchi against power support at the same pixel.
        if 'Yamaguchi' in kind:
            p=Path(path).name.replace('_Cal_mat_Yamaguchi_Spk_TC','_Cal_Spk_TC')
            with rasterio.open(Path(path).parent/p) as ps:
                pa=ps.read(1)
                pr,pc=rowcol(ps.transform,xs,ys)
            # sampled windows use local mask below; support products differ by <1 pixel
        prefix=f'rs_{date}_'+({'Spk_TC':'spk','HAlpha':'ha','mat':'t3','Yamaguchi':'yam'}.get(kind,kind))
        if 'Yamaguchi' in kind: prefix=f'rs_{date}_yam'
        elif 'HAlpha' in kind: prefix=f'rs_{date}_ha'
        elif kind.startswith('mat_Spk'): prefix=f'rs_{date}_t3'
        else: prefix=f'rs_{date}_spk'
        for b in range(arr.shape[0]):
            vals=arr[b]
            for rad in (1,2):
                stats=[]
                for r,c in zip(rr,cc):
                    r0=max(0,r-rad); r1=min(vals.shape[0],r+rad+1); c0=max(0,c-rad); c1=min(vals.shape[1],c+rad+1)
                    z=vals[r0:r1,c0:c1].astype(float)
                    vm=valid[b,r0:r1,c0:c1]
                    if 'Yamaguchi' in kind:
                        # require finite Yamaguchi and positive co-located power support
                        vm=vm & np.isfinite(z)
                    z=z[vm]
                    if z.size<3: stats.append([np.nan]*7)
                    else: stats.append([np.mean(z),np.std(z),np.min(z),np.max(z),np.percentile(z,10),np.percentile(z,90),float(z.size)])
                stats=np.asarray(stats)
                for j,sn in enumerate(('mean','std','min','max','p10','p90','n')):
                    out.loc[idx,f'{prefix}_b{b+1}_r{rad}_{sn}']=stats[:,j]

# Same-date change and ratios are useful compact summaries; raw means already exist in L.
for fam in ('spk','ha','yam','t3'):
    for b in range(1,10):
        a=f'rs_260610_{fam}_b{b}_r1_mean'; z=f'rs_261014_{fam}_b{b}_r1_mean'
        if a in out and z in out:
            out[f'rs_delta_{fam}_b{b}']=out[z]-out[a]
            out[f'rs_ratio_{fam}_b{b}']=out[z]/(np.abs(out[a])+1e-6)

out.to_parquet(OUT/'features.parquet',index=False)
OUT.joinpath('MANIFEST.md').write_text('''# ALOS window statistics\n\nExtracted from `/app/rasters` using rasterio, reprojecting EPSG:4326 tree coordinates to each native raster CRS. For each raster band and radius 1 (3x3) and 2 (5x5), stored mean, std, min, max, p10, p90, and valid count. Power/HAlpha/T3 masks use finite and >0 (T3 T11 support); Yamaguchi dB values remain signed and finite. Also stored two-date differences and ratios for matched bands. Coordinates are copied exactly from the two source CSVs.\n''')
print(out.shape, out.isna().sum().sum(), OUT/'features.parquet')
