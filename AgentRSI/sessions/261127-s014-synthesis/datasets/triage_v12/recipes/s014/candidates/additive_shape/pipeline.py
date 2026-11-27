import sys
sys.path.insert(0,"/app/research")
from engine import predict,adaptive
CONFIG={'additive': True}
def fit_predict(labeled,unlabeled,ctx):
 return (adaptive if CONFIG.get("adaptive") else predict)(labeled,unlabeled,ctx,CONFIG)
