"""SAR patch representation: frozen encoder input, both dates and change.
Linear backscatter support is authoritative; no Yamaguchi scale assumptions.
"""
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.windows import Window
from rasterio.warp import transform
d=pd.concat([pd.read_csv('/app/data/L_labelled.csv').drop(columns='label'),pd.read_csv('/app/data/unlabelled.csv')],ignore_index=True)[['estate','Long','Lat']]
chips=np.zeros((len(d),2,3,32,32),dtype=np.uint8);support=np.zeros((len(d),2),np.float32)
for e,a in d.groupby('estate',sort=False):
 for j,date in enumerate(['260610','261014']):
    with rasterio.open(f'/app/rasters/ALOS2-Subset_{e}_{date}_Cal_Spk_TC.tif') as s:
        xx,yy=transform('EPSG:4326',s.crs,a.Long.tolist(),a.Lat.tolist());image=s.read([1,2,4]);med=np.array([np.median(v[np.isfinite(v)&(v>0)]) for v in image])
        for i,x,y in zip(a.index,xx,yy):
            r,c=s.index(x,y);v=s.read([1,2,4],window=Window(c-16,r-16,32,32),boundless=True,fill_value=np.nan)
            valid=np.isfinite(v)&(v>0);support[i,j]=valid.mean()
            for k in range(3):v[k,~valid[k]]=med[k]
            db=10*np.log10(v)
            # Fixed physical display ranges shared over dates/estates.
            lo=np.array([-25,-35,-25])[:,None,None];hi=np.array([0,-5,0])[:,None,None]
            chips[i,j]=np.round(np.clip((db-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    print(e,date,flush=True)
dest=Path('/run/agentrsi/colab-io/in');np.savez_compressed(dest/'s010_sar.npz',chips=chips,support=support)
d.to_parquet('/app/research/sar_keys.parquet',index=False)
print(chips.shape,'support',support.min(),support.mean(),flush=True)
