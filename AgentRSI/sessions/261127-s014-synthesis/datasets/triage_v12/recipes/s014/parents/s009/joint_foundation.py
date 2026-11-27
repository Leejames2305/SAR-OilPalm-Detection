import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from directions import offset_fit,oof_base
import optical

def features(d,ctx):
 q=optical.cov(d,ctx,'gee_quality',['B2','B3','B4','B8']);blue,green,red,nir=q.T/10000
 return np.c_[(2*green-red-blue)/(2*green+red+blue+1e-6),(nir-red)/(nir+red+.16),np.sqrt(np.maximum((1-blue)*(1-green)*(1-red),0))]

def fit_predict(labeled,unlabeled,ctx):
 d=labeled[labeled.label.isin(['Healthy','Unhealthy'])];y=(d.label=='Unhealthy').to_numpy(int)
 cols=[f'c2025_A{i:02}' for i in range(64)]
 x=optical.cov(d,ctx,'alphaearth',cols);t=optical.cov(unlabeled,ctx,'alphaearth',cols)
 scale=StandardScaler();x=scale.fit_transform(x);t=scale.transform(t);p=PCA(n_components=8,svd_solver='full');x=p.fit_transform(x);t=p.transform(t)
 x=np.c_[features(d,ctx),x];t=np.c_[features(unlabeled,ctx),t];scale=StandardScaler();x=scale.fit_transform(x);t=scale.transform(t);x=np.clip(x,-3,3);t=np.clip(t,-3,3)
 a=offset_fit(x,y,oof_base(d,ctx),.5)
 return np.asarray(optical.fit_predict(labeled,unlabeled,ctx)+.3*(t@a),dtype=np.float64)
