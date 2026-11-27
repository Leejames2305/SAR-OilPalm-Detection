from pathlib import Path
import argparse,json,sys
ROOT=Path('/app');sys.path.insert(0,str(ROOT/'research'))
CONFIGS={
 'uniform_sar':{'patches':['sar']},
 'uniform_optpatch':{'patches':['optical']},
 'uniform_bothpatch':{'patches':['optical','sar']},
 'sar_offset':{'patches':['sar'],'offset':True},
 'both_offset':{'patches':['optical','sar'],'offset':True},
 'uniform_pool':{'pool':True},
 'uniform_pool_patch':{'pool':True,'patches':['optical','sar']},
 'pool_fullopt':{'pool':True,'shared_cols':'full'},
 's010_uniform':{'pool':True,'shared_cols':'legacy','quality_blend':True,'optical':'legacy','alpha_std':False,'opt_scale':.3,'alpha_scale':.3,'height':False,'mixture':False,'patches':['optical','sar'],'C':.1},
 'uniform_rawalpha':{'alpha_std':False},
 'uniform_blend':{'quality_blend':True},
 'uniform_nocoords':{'no_coords':True},
}
def build(name,cfg):
 from synthesis import OPT
 if cfg.get('shared_cols')=='full':cfg['shared_cols']=OPT
 if cfg.get('shared_cols')=='legacy':cfg['shared_cols']=['NDVI_contrast35','NDRE1_contrast35','ndvi_dip','ndvi_sd20','ndvi_edge','ndre_contrast']
 out=ROOT/'research/candidates'/name;out.mkdir(parents=True,exist_ok=True)
 (out/'pipeline.py').write_text('"""Controlled synthesis: '+name+'."""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0,"/app/research")\nfrom synthesis import predict\nCONFIG = '+repr(cfg)+'\ndef fit_predict(labeled,unlabeled,ctx):\n    return predict(labeled,unlabeled,ctx,CONFIG)\n')
 (out/'requirements.txt').write_text('threadpoolctl==3.8.0\n')
 (out/'config.json').write_text(json.dumps(cfg,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=list(CONFIGS));a=p.parse_args()
 for name in a.names:build(name,dict(CONFIGS[name]))
