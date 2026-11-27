"""Label-free transductive PCA compression; no supervised labels loaded."""
from pathlib import Path
import shutil,json
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
root=Path('/app/datasets/dino_optical');root.mkdir(exist_ok=True)
raw=Path('/run/agentrsi/colab-io/out/job-0001-8dd9be')
d=pd.read_parquet('/app/research/optical_chips/keys.parquet')
e=np.load(raw/'s010_dino.npz');out=[];models={};meta={}
with threadpool_limits(limits=4):
 for k in e.files:
    p=PCA(n_components=64,svd_solver='full');v=p.fit_transform(e[k]);out.append(pd.DataFrame(v,columns=[f'{k}_pc{i:02}' for i in range(64)]));models[k+'_mean']=p.mean_;models[k+'_components']=p.components_
    meta[k]={'variance_retained':float(p.explained_variance_ratio_.sum()),'n':len(v),'dimension':64}
pd.concat([d]+out,axis=1).to_parquet(root/'features.parquet',index=False)
np.savez_compressed(root/'pca_transform.npz',**models)
(root/'compression.json').write_text(json.dumps(meta,indent=2));shutil.copy2(raw/'s010_dino_meta.json',root/'encoder_meta.json')
for f in ['prepare_gpu.py','dino_gpu.py','compress_dino.py']:shutil.copy2(Path('/app/research')/f,root/f)
shutil.copy2('/app/datasets/optical_jitter/extract_optical_chips.py',root/'extract_optical_chips.py')
(root/'MANIFEST.md').write_text('''# Frozen multiscale optical DINOv2 embeddings

Source cloud-controlled Sentinel-2 mosaic as optical_jitter: S2 SR Harmonized Apr15–Aug15 2026, SCL4 and CloudScore+ cs_cdf>=.65; wider SCL4 Mar15–Sep15 fallback. Bands B4/B3/B2 and B8/B4/B3, reflectance DN/10000. Native10m RGB/NIR; no resolution enhancement claim. sampleRectangle centre/corner offset corrected to true grid corners (-5m east,+5m north relative to saved raster origin). Integer pixel window anchored on trunk-containing pixel, sizes8/16/32 =>80/160/320m. No label-derived centering, no label use.

Frozen facebookresearch/dinov2 dinov2_vits14 weights (official dl.fbaipublicfiles.com). RGB3 views scaled by.3, NIR-red-green160m view by.5, clipped0..1, bilinear upsample224, ImageNet mean/SD normalization. CLS384-dimensional embedding per view; no finetuning or trained GPU head. Compressed each view to64 PCA components using all7220 covariate-only patches, allowed transductive feature preparation. Downstream supervised learning and any further PCA use training rows only. Compression does not use ground truth. Four64-PC views plus3 exact key columns,7220 rows. No missing/duplicate keys or forbidden columns. pca_transform.npz allows projection of reproduced full encoder outputs. Source keys preserved without rounding.

Reproduce extract_optical_chips.py, prepare_gpu.py, remotely execute dino_gpu.py via broker, then compress_dino.py (update only raw output job path). GPU not required at fit_predict: final features are offline CPU-readable. DINOv2 citation https://arxiv.org/abs/2304.07193. No pathogen-specific interpretation assigned to embedding axes.
''')
print(meta,flush=True)
