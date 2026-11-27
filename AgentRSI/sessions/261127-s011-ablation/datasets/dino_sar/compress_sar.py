from pathlib import Path
import shutil,json
import pandas as pd,numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
root=Path('/app/datasets/dino_sar');root.mkdir(exist_ok=True);raw=Path('/run/agentrsi/colab-io/out/job-0002-451135/s010_sar_dino.npz')
d=pd.read_parquet('/app/research/sar_keys.parquet');e=np.load(raw);out=[];models={};meta={}
with threadpool_limits(limits=4):
 for k in e.files:
    p=PCA(n_components=64,svd_solver='full');v=p.fit_transform(e[k]);out.append(pd.DataFrame(v,columns=[f'{k}_pc{i:02}' for i in range(64)]));models[k+'_mean']=p.mean_;models[k+'_components']=p.components_;meta[k]=float(p.explained_variance_ratio_.sum())
pd.concat([d]+out,axis=1).to_parquet(root/'features.parquet',index=False);np.savez_compressed(root/'pca_transform.npz',**models);(root/'compression.json').write_text(json.dumps(meta,indent=2))
for f in ['prepare_sar_gpu.py','dino_sar_gpu.py','compress_sar.py']:shutil.copy2(Path('/app/research')/f,root/f)
(root/'MANIFEST.md').write_text('''# Frozen DINOv2 multitemporal SAR patch covariates

Source ALOS2-Subset_EST_DATE_Cal_Spk_TC.tif dates260610 and261014,EPSG32648,6.43m. Native CRS reprojection from exact WGS84 source keys before pixel indexing. Channels HH/HV/VV (bands1/2/4), linear power ->10log10 dB; finite and>0 support, all extracted crops have100% support. Fixed display ranges[-25,0]/[-35,-5]/[-25,0]dB mapped/clipped0..255,uint8. Trunk-containing pixel centre anchored8x8 (~51m) and16x16(~103m) windows for each date. Change103 is.5+(OctoberDisplay-JuneDisplay),clipped0..1 before frozen encoder. Not a calibrated change in linear power; not radiometrically sharpened.

Frozen official DINOv2 dinov2_vits14 RGB encoder,224 bilinear upsample,ImageNet normalization,CLS384 features; no labels/fine-tuning/GPU head. Five views jun51/oct51/jun103/oct103/change103 compressed independently to64 PCs on all7220 covariate-only vectors. PCA transforms saved. Downstream training-only PCA/scaling and spatially held-out labels. Exact7220 original coordinate keys,323columns,zeroNaNs/duplicates. No forbidden columns. Encoder outputs SHA256def388f02d668a49de34a51d19914a2ee99e11c6075abb75d0d1c512d5c11d76.

Reproduce prepare_sar_gpu.py,remote dino_sar_gpu.py,then compress_sar.py (job-specific input path). No raster or GPU dependency at fit_predict. Out-of-domain RGB pretrained encoder may discard SAR-specific information; negative results do not rule out SAR-native pretraining. Citation https://arxiv.org/abs/2304.07193.
''');print(meta,flush=True)
