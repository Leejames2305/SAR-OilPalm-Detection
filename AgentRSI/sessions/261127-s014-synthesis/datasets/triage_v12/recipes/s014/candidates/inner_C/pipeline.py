import sys
sys.path.insert(0,"/app/research")
from engine import predict,adaptive
CONFIG={'adaptive': True, 'options': [{'C': 0.05}, {'C': 0.15}, {'C': 0.45}]}
def fit_predict(labeled,unlabeled,ctx):
 return (adaptive if CONFIG.get("adaptive") else predict)(labeled,unlabeled,ctx,CONFIG)
