from pathlib import Path
import json
ROOT=Path('/app/research'); OUT=ROOT/'candidates';OUT.mkdir(exist_ok=True)
CONFIGS={
 'control':{},
 'domain_optical':{'domain':'optical'},
 'domain_context':{'domain':'context'},
 'context_regularized':{'context_scale':.5},
 'context_regularized_signs':{'context_scale':.5,'signs':True},
 'inner_C':{'adaptive':True,'options':[{'C':.05},{'C':.15},{'C':.45}]},
 'inner_context':{'adaptive':True,'options':[{'context_scale':.5},{'context_scale':1.},{'context_scale':1.5}]},
 'additive_shape':{'additive':True},
 'additive_signs':{'additive':True,'signs':True},
 'additive_grouped':{'additive':True,'grouped':True},
 'grouped':{'grouped':True},
 'signs':{'signs':True},
 'grouped_signs':{'grouped':True,'signs':True},
 'orthogonal':{'orthogonal':'all'},
 'spectral_protect':{'orthogonal':'spectral'},
 'anchor_background':{'anchor':.1},
 'block_balanced':{'block_balance':True},
 'no_sar':{'patches':[]},
 'change_context':{'patches':['change103_']},
 'date_context':{'patches':['jun103_','oct103_']},
}
for name,cfg in CONFIGS.items():
 p=OUT/name;p.mkdir(exist_ok=True)
 (p/'config.json').write_text(json.dumps(cfg,indent=2))
 (p/'pipeline.py').write_text('import sys\nsys.path.insert(0,"/app/research")\nfrom engine import predict,adaptive\nCONFIG='+repr(cfg)+'\ndef fit_predict(labeled,unlabeled,ctx):\n return (adaptive if CONFIG.get("adaptive") else predict)(labeled,unlabeled,ctx,CONFIG)\n')
 (p/'requirements.txt').write_text('threadpoolctl==3.8.0\n')
