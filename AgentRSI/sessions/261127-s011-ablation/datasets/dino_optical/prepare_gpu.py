"""Create corrected-grid, label-free optical patches for frozen GPU encoder."""
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
root=Path('/app/research/optical_chips');d=pd.read_parquet(root/'keys.parquet')
chips=np.empty((len(d),4,32,32),np.float32)
for e,a in d.groupby('estate',sort=False):
    with rasterio.open(root/f'{e}.tif') as s:
        image=s.read([3,2,1,4])/10000
        xx,yy=transform('EPSG:4326',s.crs,a.Long.tolist(),a.Lat.tolist())
        # sampleRectangle coordinate grids give centres; recover raster corner.
        cc=np.floor((np.array(xx)-(s.transform.c-5))/10).astype(int)
        rr=np.floor(((s.transform.f+5)-np.array(yy))/10).astype(int)
        for i,r,c in zip(a.index,rr,cc):chips[i]=image[:,r-16:r+16,c-16:c+16]
assert np.isfinite(chips).all() and (chips>0).all()
dest=Path('/run/agentrsi/colab-io/in');dest.mkdir(parents=True,exist_ok=True)
np.savez_compressed(dest/'s010_optical.npz',chips=chips)
d.to_parquet(dest/'s010_keys.parquet',index=False)
print(chips.shape,flush=True)
