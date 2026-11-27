"""Historical-origin reproduction ONLY. These folds split original80m blocks."""
import sys
import numpy as np,pandas as pd
sys.path.insert(0,'/app/research');import evaluate
L=pd.read_csv('/app/data/L_labelled.csv')
evaluate.ORIGINS={e:(d.Long.min(),d.Lat.min(),d.Lat.mean()) for e,d in L.groupby('estate')}
for model in ['incumbent','coords','s009','s012','s013']:
 evaluate.run(model,['legacyL80'])
