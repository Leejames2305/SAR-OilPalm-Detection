from engine import source_prior
CONFIG={}
def fit_predict(labeled,unlabeled,ctx):
 return source_prior(labeled,unlabeled,ctx,CONFIG)
