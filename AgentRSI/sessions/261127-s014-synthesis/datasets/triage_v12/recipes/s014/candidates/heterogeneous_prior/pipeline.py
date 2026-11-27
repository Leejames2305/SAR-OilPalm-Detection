import sys
sys.path.insert(0,"/app/research")
from engine import source_prior
CONFIG={'heterogeneity': True}
def fit_predict(labeled,unlabeled,ctx):
 return source_prior(labeled,unlabeled,ctx,CONFIG)
