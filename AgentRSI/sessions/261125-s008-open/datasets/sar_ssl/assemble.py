from pathlib import Path
import numpy as np,pandas as pd,json,shutil
from sklearn.decomposition import PCA
D=Path(__file__).parent;source=Path('/run/agentrsi/colab-io/out/job-0001-c364f4');a=np.load(source/'embeddings.npz');keys=pd.read_parquet(D/'keys.parquet');out=keys.copy()
for mode in ['vicreg','masked_ae']:
 x=a[mode].astype('float64');sc=x.std(0);z=(x-x.mean(0))/np.maximum(sc,1e-5);p=PCA(n_components=32,svd_solver='full');u=p.fit_transform(z)
 for i in range(32):out[f'{mode}_pc{i+1:02}']=u[:,i]
 np.savez_compressed(D/f'{mode}_transform.npz',mean=x.mean(0),scale=sc,components=p.components_)
 print(mode,'32var',p.explained_variance_ratio_.sum())
out['reconstruction_error']=a['reconstruction'];out.to_parquet(D/'features.parquet',index=False)
for n in ['vicreg.pt','ae.pt']:shutil.copy2(source/n,D/n)
shutil.copy2(source/'embeddings.npz',D/'raw_embeddings.npz')
print(out.shape,'nan',out.isna().sum().sum())
