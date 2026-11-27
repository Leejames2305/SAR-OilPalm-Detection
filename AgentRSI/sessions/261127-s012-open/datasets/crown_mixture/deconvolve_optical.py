"""Label-free regularized crown-mixture inversion of native Sentinel-2 spectra.

This is a geometric observation-model experiment, not a claim of new spatial
resolution. Gaussian PSF and ridge prior are fixed before looking at labels.
"""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import cg
from scipy.ndimage import map_coordinates

ROOT=Path('/app/datasets/crown_mixture');ROOT.mkdir(exist_ok=True)
keys=pd.concat([pd.read_csv('/app/data/L_labelled.csv',usecols=['estate','Long','Lat']),pd.read_csv('/app/data/unlabelled.csv',usecols=['estate','Long','Lat'])],ignore_index=True)
bands=['B2','B3','B4','B8','B5','B8A','B11','B12'];parts=[];audit={}
def indices(v):
    b=dict(zip(bands,v.T));eps=1e-8
    return {'NDVI':(b['B8']-b['B4'])/(b['B8']+b['B4']+eps),
            'NDRE':(b['B8A']-b['B5'])/(b['B8A']+b['B5']+eps),
            'NDMI':(b['B8']-b['B11'])/(b['B8']+b['B11']+eps),
            'NBR':(b['B8']-b['B12'])/(b['B8']+b['B12']+eps),
            'NIRv':b['B8']*(b['B8']-b['B4'])/(b['B8']+b['B4']+eps),
            'GRVI':(b['B3']-b['B4'])/(b['B3']+b['B4']+eps)}
for estate,d in keys.groupby('estate',sort=False):
    with rasterio.open(f'/app/research/native_optical/{estate}.tif') as src:
        image=src.read().astype(float);xs,ys=transform('EPSG:4326',src.crs,d.Long.tolist(),d.Lat.tolist());xy=np.column_stack([xs,ys])
        rr,cc=np.indices(src.shape);px,py=rasterio.transform.xy(src.transform,rr.ravel(),cc.ravel());pix=np.column_stack([px,py])
        # Bilinear interpolated point spectrum is a stable prior f0.
        c=(np.asarray(xs)-src.transform.c)/10-.5;r=(src.transform.f-np.asarray(ys))/10-.5
        f0=np.column_stack([map_coordinates(b,[r,c],order=1,mode='nearest') for b in image])
    tree=cKDTree(xy)
    distance,nearest=tree.query(pix,k=8)
    keep=(distance[:,0]<8)&np.all(np.isfinite(image.reshape(8,-1).T),axis=1)
    target=image.reshape(8,-1).T[keep]
    assert np.isfinite(f0).all() and (target>0).all()
    # At most8 known crowns within15m contribute to each pixel; normalize
    # spectral mixture so a spatially constant canopy is reconstructed exactly.
    dist=distance[keep];ix=nearest[keep]
    weight=np.exp(-dist**2/(2*5.**2))*(dist<15)
    weight/=weight.sum(1,keepdims=True)
    W=coo_matrix((weight.ravel(),(np.repeat(np.arange(len(ix)),8),ix.ravel())),shape=(len(ix),len(d))).tocsr()
    result={}
    for prefix,v in [('point',f0)]:
        for name,value in indices(v).items():result[f'{prefix}_{name}']=value
    diag={}
    for ridge in [.25,1.]:
        # Ridge toward the observed bilinear spectrum, not toward a class score.
        op=W.T@W+ridge*eye(len(d),format='csr');rhs=W.T@(target-W@f0)
        delta=[]
        for k in range(8):
            z,info=cg(op,rhs[:,k],rtol=1e-8,maxiter=1000);assert info==0;delta.append(z)
        fit=f0+np.column_stack(delta);fit=np.clip(fit,.0001,1.)
        name='deblur' if ridge==.25 else 'stable'
        for col,value in indices(fit).items():result[f'{name}_{col}']=value
        diag[name]={'pixel_rmse':float(np.sqrt(np.mean((W@fit-target)**2))),
                    'point_change_rms':float(np.sqrt(np.mean((fit-f0)**2)))}
    # Local spectral deficits relative to catalogue neighbors at <=30m.
    _,neighbors=tree.query(xy,k=10)
    for prefix in ['point','deblur','stable']:
        for index in ['NDVI','NDRE','NDMI']:
            v=result[f'{prefix}_{index}']
            result[f'{prefix}_{index}_contrast']=v-np.median(v[neighbors[:,1:]],axis=1)
    part=pd.concat([d,pd.DataFrame(result,index=d.index)],axis=1);parts.append(part)
    audit[estate]={'pixels':len(target),'trees':len(d),'prior_pixel_rmse':float(np.sqrt(np.mean((W@f0-target)**2))),**diag}
    print(estate,audit[estate],flush=True)
out=pd.concat(parts).sort_index();assert np.isfinite(out.iloc[:,3:].to_numpy()).all()
out.to_parquet(ROOT/'features.parquet',index=False)
(ROOT/'inversion.json').write_text(json.dumps(audit,indent=2))
for f in ['extract_optical_native.py','deconvolve_optical.py']:shutil.copy(Path('/app/research')/f,ROOT/f)
shutil.copy('/app/research/georeference_audit.json',ROOT/'georeference_audit.json')
(ROOT/'MANIFEST.md').write_text('''# Native Sentinel-2 crown-mixture experiment

Source COPERNICUS/S2_SR_HARMONIZED Apr15-Aug15 2026, SCL4-7 and CloudScore+
cs_cdf>=.65; same-window SCL4-7 fallback. Median B2/B3/B4/B8/B5/B8A/B11/B12,
DN/10000, no extra offset. Native10m/20m information, reprojectEPSG32648 10m.
sampleRectangle pixel-center coordinates (.5,.5) converted to GeoTIFF corners;
boundary-coordinate audit included. Exact coordinate keys reprojected before
bilinear point sampling. No labels used. All7220unique keys/zeroNaNs.

Gaussian5m crown observation model from ALL supplied tree coordinates perestate,
up to8crowns within15m per10m pixel, normalized weights; pixels nearest crown<8m.
Solve (W.T W+lambda I)(f-f0)=W.T(p-Wf0), f0=bilinear pixel spectrum. Two fixed
lambda .25/1 variants. No learned disease scores or selected label-dependent
hyperparameters. Clip reconstructed spectra .0001..1. Compute NDVI,NDRE,NDMI,
NBR,NIRv,GRVI for point/deblur/stable spectra and local contrast to9nearesttrees.
20m bands remain20m information. This underdetermined inverse problem does NOT
create native per-crown resolution, may amplify geolocation/mosaic noise, and
assumes catalogue completeness/Gaussian mixing. Footprint overlaps: coarser and
buffered validation required. Reproduction scripts and inversion diagnostics
retained. Source mosaic rasters remain research assets, not runtime dependencies.
''')
print(out.shape)
