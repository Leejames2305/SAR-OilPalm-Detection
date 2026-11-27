import sys
sys.path.insert(0,"/app/research")
from engine import predict,adaptive
CONFIG={'adaptive': True, 'options': [{'context_scale': 0.5}, {'context_scale': 1.0}, {'context_scale': 1.5}]}
def fit_predict(labeled,unlabeled,ctx):
 return (adaptive if CONFIG.get("adaptive") else predict)(labeled,unlabeled,ctx,CONFIG)
