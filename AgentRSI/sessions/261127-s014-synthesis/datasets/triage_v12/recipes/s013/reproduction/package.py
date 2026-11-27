"""Prepare a portable candidate and stage exactly its runtime dependencies."""
from pathlib import Path
import argparse,ast,json,shutil,hashlib
ROOT=Path('/app')
def package(name,stage=True):
    source=ROOT/'research/candidates'/name
    cfgfile=source/'config.json'
    dest=source/'portable';dest.mkdir(exist_ok=True)
    if cfgfile.exists():
        cfg=json.loads(cfgfile.read_text())
        (dest/'pipeline.py').write_text('"""Offline spatial oil-palm triage; risk scores are inspection priorities."""\nfrom engine import predict\nCONFIG = '+repr(cfg)+'\ndef fit_predict(labeled,unlabeled,ctx):\n    return predict(labeled,unlabeled,ctx,CONFIG)\n')
        shutil.copy2(ROOT/'research/synthesis.py',dest/'engine.py')
        dependencies={'triage_v12'}
        if 'sar' in cfg.get('patches',[]):dependencies.add('dino_sar')
        if 'optical' in cfg.get('patches',[]):dependencies.add('dino_optical')
        if cfg.get('optical')=='legacy':dependencies.update(['gee_quality','gee_canopy','optical_jitter'])
    else:
        for p in source.glob('*.py'):shutil.copy2(p,dest/p.name)
        dependencies={'gee_quality','gee_canopy','optical_jitter','alphaearth','dino_optical','dino_sar'}
        if (source/'DEPENDENCIES.json').exists():dependencies=set(json.loads((source/'DEPENDENCIES.json').read_text()))
    shutil.copy2(source/'requirements.txt',dest/'requirements.txt')
    record={'name':name,'datasets':sorted(dependencies),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*') if p.is_file()}}
    (source/'package.json').write_text(json.dumps(record,indent=2))
    if stage:
        shared=Path('/shared')/('s013_'+name);shared.mkdir(exist_ok=True)
        shutil.copytree(dest,shared/'submission',dirs_exist_ok=True)
        for dep in sorted(dependencies):
            shutil.copytree(ROOT/'datasets'/dep,shared/'datasets'/dep,dirs_exist_ok=True)
        print(shared)
    return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:package(name)
