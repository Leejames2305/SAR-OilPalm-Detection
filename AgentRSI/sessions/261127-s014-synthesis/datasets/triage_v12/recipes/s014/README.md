# Session s014 reproduction and evidence

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
