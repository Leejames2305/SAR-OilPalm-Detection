from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.warp import transform
root=Path('/app'); out=[]
allrows=pd.concat([pd.read_csv(root/'data/L_labelled.csv').drop(columns=['label']),pd.read_csv(root/'data/unlabelled.csv')],ignore_index=True)
# preserve exact keys
outdf=allrows[['estate','Long','Lat']].copy()
for estate in ['AirHitam','Palong','Serting']:
 idx=np.flatnonzero(allrows.estate.to_numpy()==estate)
 lon=allrows.iloc[idx].Long.to_numpy(float); lat=allrows.iloc[idx].Lat.to_numpy(float)
 for date in ['260610','261014']:
  for typ,bands,maskmode in [('spk',[1,2,3,4],'pos'),('ha',[1,2,3],'pos'),('yam',[1,2,3,4],'finite')]:
   pat={'spk':'Cal_Spk_TC.tif','ha':'Cal_mat_HAlpha_Spk_TC.tif','yam':'Cal_mat_Yamaguchi_Spk_TC.tif'}[typ]
   f=root/'rasters'/f'ALOS2-Subset_{estate}_{date}_{pat}'
   with rasterio.open(f) as src:
    if src.crs.to_string()=='EPSG:4326': xs,ys=lon,lat
    else: xs,ys=transform('EPSG:4326',src.crs,lon.tolist(),lat.tolist())
    rows=[]; cols=[]
    for xx,yy in zip(xs,ys):
     rr,cc=src.index(xx,yy); rows.append(rr); cols.append(cc)
    arr=src.read()
    for bi,b in enumerate(bands):
     vals=[]
     for rr,cc in zip(rows,cols):
      r0=max(0,rr-1);r1=min(src.height,rr+2);c0=max(0,cc-1);c1=min(src.width,cc+2)
      z=arr[bi,r0:r1,c0:c1].astype(float).ravel()
      if maskmode=='pos': z=z[np.isfinite(z)&(z>0)]
      else: z=z[np.isfinite(z)]
      if len(z)==0: vals.append([np.nan]*5)
      else: vals.append([np.mean(z),np.std(z),np.min(z),np.max(z),np.percentile(z,75)-np.percentile(z,25)])
     vals=np.asarray(vals)
     base={'spk':['HH','HV','VH','VV'],'ha':['H','A','alpha'],'yam':['Pd','Pv','Ps','Pc']}[typ][bi]
     for j,s in enumerate(['m','sd','mn','mx','iqr']): outdf.loc[idx,f'{base}_{date}_{s}']=vals[:,j]
outdf.to_parquet(root/'datasets/winstats/features.parquet',index=False)
print(outdf.shape,outdf.isna().sum().sum(),outdf.memory_usage(deep=True).sum())
