from legacy_extra import fit_predict as _predict
def fit_predict(labeled,unlabeled,ctx):
    return _predict(labeled,unlabeled,ctx,kind='structure',offset=False)
