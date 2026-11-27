from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('budgetfloor25_engine',Path(__file__).with_name('model_engine.py'))
_engine=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
CONFIG={'guard': 'budgetfloor'}
def fit_predict(labeled,unlabeled,ctx):
    return _engine.fit_predict(labeled,unlabeled,ctx,CONFIG)
