"""Preserve source-only recipes and small evidence without altering covariates."""
from pathlib import Path
import shutil,json,hashlib
ROOT=Path('/app');dest=ROOT/'datasets/triage_v12/recipes/s014';dest.mkdir(parents=True,exist_ok=True)
for name,source in [('selected',ROOT/'submission'),('parents',ROOT/'research/parents'),('candidates',ROOT/'research/candidates'),('checkpoint_v2',ROOT/'research/checkpoint_v2')]:
 shutil.copytree(source,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
repro=dest/'reproduction';repro.mkdir(exist_ok=True)
for p in (ROOT/'research').glob('*.py'):shutil.copy2(p,repro/p.name)
# Store canonical result evidence, no row-level labels/scores as feature covariates.
evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
for suffix in ['*.csv','*.json','*.md','*.txt']:
 for p in (ROOT/'report').glob(suffix):
  if p.name.startswith('unlabelled_ranking'):continue
  shutil.copy2(p,evidence/p.name)
shutil.copytree(ROOT/'report/verifier',evidence/'verifier',dirs_exist_ok=True)
readme='''# Session s014 reproduction and evidence

This folder preserves source recipes, not fitted disease weights or risk-feature
inputs. Runtime only loads the parent features.parquet and dino_sar/features.parquet.

Restore reproduction/*.py into /app/research and parents/ into /app/research/parents;
restore candidates/ into /app/research/candidates and checkpoint_v2/ similarly.
selected/ contains the clean final pipeline. Copy its pipeline.py and requirements.txt
to /app/submission. Reproduction programs expect authorized original L/U CSVs.
See evidence/REPORT.md and HANDOFF.md for exact commands and findings.

CRITICAL: corrected validation uses all L+U coordinate origins/mean latitude.
legacy_origin_probe.py intentionally reproduces the invalid-for-current-task
L-only historical geometry. Do not use legacyL80 as the primary official score.
Required original80m blocks remain intact in coarser/quadrant stress protocols.

Final source_prior: independent source fits, frozen mean optical coefficient prior,
local target contextual response. No score ensemble or estate-name branch.
Two V queries used; V2 selected with .226709 composite. No fresh noise queries
for packaging. Clean package matches immutable V2 in57fits exactly.

No new data extraction occurred. All source feature manifests/scripts remain
under their dataset directories; recipes/s013 preserves earlier lineages.
Small evidence tables summarize experiments. Row-level OOF arrays remain in
/app/research/results when available and can be regenerated from source.
'''
(dest/'README.md').write_text(readme)
checksums={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='checksums.json'}
(dest/'checksums.json').write_text(json.dumps(checksums,indent=2))
print('Preserved',len(checksums),'files;',sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'bytes')
