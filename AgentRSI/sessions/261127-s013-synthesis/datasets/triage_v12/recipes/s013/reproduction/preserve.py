"""Carry exact alternatives and reproduction code without duplicating features."""
from pathlib import Path
import shutil,json,hashlib
ROOT=Path('/app');out=ROOT/'datasets/triage_v12/recipes/s013';out.mkdir(parents=True,exist_ok=True)
parents={'incumbent':ROOT/'incumbent','s009':ROOT/'prior/ref_sessions/261125-s009-open/submission','s010':ROOT/'prior/ref_sessions/261125-s010-open/submission','s011':ROOT/'prior/ref_sessions/261127-s011-ablation/submission','s012':ROOT/'prior/prev_submission'}
for name,path in parents.items():
    dest=out/'parents'/name;dest.mkdir(parents=True,exist_ok=True)
    for p in path.iterdir():
        if p.is_file() and p.suffix in ['.py','.txt']:shutil.copy2(p,dest/p.name)
dest=out/'parents/s012_bag';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'research/block_bag.py',dest/'pipeline.py');shutil.copy2(ROOT/'submission/requirements.txt',dest/'requirements.txt')
for source in (ROOT/'research/candidates').iterdir():
    if not source.is_dir():continue
    dest=out/'candidates'/source.name;dest.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ['.py','.txt','.json']:shutil.copy2(p,dest/p.name)
    if (source/'portable').exists():
        for p in (source/'portable').iterdir():
            if p.is_file() and p.suffix in ['.py','.txt']:
                d=dest/'portable';d.mkdir(exist_ok=True);shutil.copy2(p,d/p.name)
for name in ['selected','reproduction','evidence']:(out/name).mkdir(exist_ok=True)
for p in (ROOT/'submission').glob('*'):
    if p.is_file() and p.suffix in ['.py','.txt']:shutil.copy2(p,out/'selected'/p.name)
for p in (ROOT/'research').glob('*.py'):shutil.copy2(p,out/'reproduction'/p.name)
for name in ['selection.json','validation_plan.json','OFFICIAL_EXPERIMENTS.csv','EXPERIMENT_SUMMARY.csv','bootstrap.json','package_equivalence_final.json','dataset_audit.json']:
    p=ROOT/'report'/name
    if p.exists():shutil.copy2(p,out/'evidence'/name)
for p in (ROOT/'report/verifier').glob('*.json'):shutil.copy2(p,out/'evidence'/p.name)
(out/'README.md').write_text('''# Session 261127-s013-synthesis preserved sources

No feature extraction or feature values changed here. Selected model uses exactly
triage_v12 and dino_sar, through ctx.datasets_dir. All 7220 keys remain unchanged.
There are no labels, fitted disease weights, or sealed split assignments here.

- selected/: clean final pooled_sar pipeline and exact runtime requirements.
- parents/: all audited byte-preserved parent implementations, including the
  conservative s012 alternative and its original 32-block bag.
- candidates/: all research configurations. Eight scored variants include a
  portable/ directory preserving the exact source submitted to V. Their
  package.json or DEPENDENCIES.json names needed feature tables.
- reproduction/: evaluator, mechanism source, analysis and report scripts.
  Restore into /app/research to use documented absolute development paths.
  evaluate.py can find preserved parents/candidates here if prior/ref_sessions
  is absent. L-only wrappers expect synthesis.py restored to /app/research.
- evidence/: V observations, explicit exploratory selection-policy amendment,
  paired summaries, bootstrap and packaging-equivalence records.

The feature tables' existing manifests/scripts remain authoritative. Research
plotting uses matplotlib==3.10.7; runtime requires only threadpoolctl==3.8.0
beyond the grading base stack. No GPU work or new acquisition was performed.
''')
index={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'}
(out/'checksums.json').write_text(json.dumps(index,indent=2));print('Preserved',len(index),'files')
