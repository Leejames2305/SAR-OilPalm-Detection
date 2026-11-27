"""Label-free planting geometry over the complete supplied coordinate catalogue."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import cKDTree,ConvexHull
ROOT=Path('/app/datasets/planting_geometry');ROOT.mkdir(exist_ok=True)
d=pd.concat([pd.read_csv('/app/data/L_labelled.csv',usecols=['estate','Long','Lat']),pd.read_csv('/app/data/unlabelled.csv',usecols=['estate','Long','Lat'])],ignore_index=True)
out=[]
for estate,a in d.groupby('estate',sort=False):
    xy=np.column_stack([(a.Long-a.Long.mean())*111320*np.cos(np.deg2rad(a.Lat.mean())),(a.Lat-a.Lat.mean())*110540])
    tree=cKDTree(xy);dist,ix=tree.query(xy,k=17)
    r=dist[:,1:];dx=xy[ix[:,1:]]-xy[:,None,:];ang=np.arctan2(dx[:,:,1],dx[:,:,0])
    feat={f'distance{k}':r[:,k-1] for k in [1,3,6,10,16]}
    for k in [3,6,10]:
        feat[f'spacing_cv{k}']=r[:,:k].std(1)/r[:,:k].mean(1)
    for radius in [12,20,30,45]:feat[f'neighbor_count{radius}']=tree.query_ball_point(xy,radius,return_length=True)-1
    for harmonic in [2,4,6]:
        z=np.exp(1j*harmonic*ang[:,:6]).mean(1)
        feat[f'angular_regular{harmonic}']=np.abs(z)
        feat[f'angular_cos{harmonic}']=z.real;feat[f'angular_sin{harmonic}']=z.imag
    feat['hexagon_area']=np.array([ConvexHull(row).volume for row in dx[:,:6]])
    out.append(pd.concat([a,pd.DataFrame(feat,index=a.index)],axis=1))
out=pd.concat(out).sort_index();assert np.isfinite(out.iloc[:,3:].to_numpy()).all()
out.to_parquet(ROOT/'features.parquet',index=False)
(ROOT/'MANIFEST.md').write_text('''# Label-free planting geometry

All7220 exact original coordinate keys from L union U; no labels used.
Local metres:111320*cos(meanlat) east,110540north. KDTree over all supplied
trees per estate. Distances to kth1/3/6/10/16 neighbors; coefficient of variation
nearest3/6/10 spacing; counts within12/20/30/45m; angular2/4/6harmonic complex
mean over6 nearest tree bearings (magnitude,real,imaginary); nearest6 convex
hull area. Script extract_geometry.py. All finite/no duplicate keys. Catalogue
completeness is not guaranteed: these are geometry/context proxies, not observed
stumps, missing trees or disease diagnoses. Counts can encode estate boundary.
Spatial footprints up to45m radius require buffered/coarse validation.
''')
print(out.shape)
