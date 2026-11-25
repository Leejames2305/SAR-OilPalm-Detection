"""Registration-integrated canopy texture residual learned on spatial folds."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import optical
import joint_foundation
from directions import offset_fit

TEXTURE=['ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
def residual(labeled,unlabeled,ctx,kind='texture_small'):
    d=labeled[labeled.label.isin(['Healthy','Unhealthy'])]
    base=joint_foundation.fit_predict(labeled,unlabeled,ctx)
    x=optical.cov(d,ctx,'optical_jitter',TEXTURE)
    t=optical.cov(unlabeled,ctx,'optical_jitter',TEXTURE)
    z=StandardScaler();x=z.fit_transform(x);t=z.transform(t)
    x=np.clip(x,-3,3);t=np.clip(t,-3,3)
    y=(d.label=='Unhealthy').to_numpy(int)
    b=np.zeros(len(d))
    for tr,va in GroupKFold(5).split(d,groups=optical.groups(d)):
        b[va]=joint_foundation.fit_predict(d.iloc[tr],d.iloc[va],ctx)
    a=offset_fit(x,y,b,.5)
    return base+.3*(t@a)
