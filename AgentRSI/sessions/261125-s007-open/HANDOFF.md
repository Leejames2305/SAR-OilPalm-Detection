## State of play

Session `261125-s007-open` nominates a locally improved offline model in `/app/submission/`. Official 80 m CV lift is **AirHitam +0.03834, Palong +0.30596, Serting +0.15393; composite +0.16608**, versus incumbent +0.03097/+0.25564/+0.13496; composite +0.14052. The paired composite difference is +0.02556, conditional 5,000-block-bootstrap 95% interval [+0.00678, +0.05156]. All 24 estate × eight outer-block-assignment primary comparisons improve. The original `/app/incumbent/` is unchanged.

**Final V performance is unavailable, not verified.** The derived-dataset verifier fails before model execution even for the unchanged incumbent. Existing incumbent V remains +0.118751. Budget used 3/15: one valid rejected spatial candidate and two infrastructure failures. Do not confuse the rejected candidate's V +0.110036 with the final model.

The final recipe is 75% Bayesian whole-block model averaging plus 25% uniform quadratic NDVI/coordinates. Six candidates compare linear/quadratic base, prior-year raw Sentinel-2 bands, and current NDVI+NDRE1. Two inner 80 m assignments and bootstrap AP votes select complexity/mechanism from training labels. No estate-name branches. Required data: only `gee_s2_stress`. Middle is excluded from fitting/scoring; all target rows are retained. All three estates' labels are equally verified/trusted.

`selfcheck` and official CLI CV pass. Fresh scratch workers with networking disabled, guarded writes and shuffled target rows pass; peak memory ~234 MiB, about one second per full fit. `threadpoolctl==3.8.0` is pinned. The 4,473 aligned deployment scores are in `/app/report/unlabelled_ranking.csv` and `.parquet`.

Operational caveat: official AirHitam Recall@25% falls 0.4821→0.4107 while Enrichment@10% rises 1.25→2.14. Its mean Recall@25% over eight assignments improves 0.3996→0.4598. Palong/Serting official Recall@25% stays 0.7424/0.6119. Region Serting lift loses 0.00472; the unanchored ensemble lost 0.01822. No majority-recall-at-25% guarantee exists for AirHitam.

## What I tried

All table values are official five-fold 80 m lift, ordered AirHitam/Palong/Serting; evidence lives under `/app/research/` unless noted. These are exploratory trials, with multiple-comparison caveats in the full report.

| Idea | Per-estate lift | Composite | Verdict / evidence |
|---|---|---:|---|
| Spatial RBF k4/w.6/C.01 | .01805/.22757/.08803 | .11122 | Reject; `spatial.log` |
| Weak Student KDE 10% blend | .03097/.25754/.13682 | .14178 | V .110036, Δ incumbent −.008715; reject; `context.log` |
| Block-honest Middle frequency 40 m/C.03 | .03097/.23511/.15423 | .14010 | Palong loss, seed instability; `context.log` |
| June–October tabular changes, shrink .1 | .03117/.24863/.13634 | .13872 | Reject; `fusion_tab.log` |
| Correct Yamaguchi fractions, shrink .3 | .01756/.25150/.13731 | .13546 | Reject; `fusion_sar.log` |
| SAR texture, shrink .3 | .03148/.22757/.11129 | .12344 | Reject; `fusion_sar.log` |
| Current optical indices, shrink .3 | .03596/.26337/.13593 | .14509 | Small improvement, superseded; `fusion_s2.log` |
| Prior-season raw bands, full strength | .07861/.31266/.11393 | .16840 | Material Serting loss; no uniform deployment |
| Compact NDVI+NDRE1, C.03 | .03258/.25387/.18197 | .15614 | Useful Serting mechanism, not universal; `optical.log` |
| Repeated inner-CV soft selector | .04323/.29952/.15643 | .16639 | Positive repeated checks; `selector_soft.log` |
| Unanchored Bayesian average | .04547/.30180/.16845 | .17191 | Region Serting loss; `bayes.log` |
| Final anchored Bayesian average | .03834/.30596/.15393 | .16608 | Nominee; `final_validation.log`, `/app/report/final_metrics.csv` |
| October optical changes, shrink .3 | .02659/.24703/.13820 | .13727 | Reject; `autumn.log` |
| Focal γ1/C.1 | .03084/.25238/.13633 | .13985 | Reject; `focal.log` |
| Recall-aware selector tolerance .02 | .03732/.28522/.14577 | .15610 | No AirHitam recall repair; `recall_guard02.log` |

New datasets: `alos_physics` re-extracts correctly masked log power, linear Yamaguchi fractions, three window scales, contrasts and changes (7,220×611, no NaNs). `gee_s2_autumn` adds October-aligned physiology and canopy context (7,220×28, no NaNs). Neither is used in the final model. Total datasets ~53.2 MiB, all exact keys, no symlinks/forbidden columns. The carried ALOS table's T3 NaNs and Yamaguchi caveats are documented.

Ablations: optical-only official composite .11489; same-recipe coordinates-only .07967; required distance-weighted 15-NN .08959. Removing prior bands gives .15039; removing red-edge gives .15308. There is spectral information beyond spatial smoothness, but prior bands also encode stand/environment context; no causal/pathogen signature is claimed.

## Dead ends

- More spatial capacity (splines/RBFs/cubic polynomials), single-seed complexity/optical gating, or weak KDE alone do not robustly replace the compact baseline.
- L-band temporal levels/changes, Yamaguchi fractions and crown/background textures have no consistent multiple-estate additive gain. Several effects flip signs; equal label trust does not remove that issue. Do not re-extract the same windows without a new mechanism.
- Uniform strong prior-year raw-band models harm Serting. Current compact red-edge is complementary; broad feature expansion is not a safe fix.
- October optical changes/crown contrasts, focal loss, and recall guards failed to improve the final recipe. The guards did not solve official AirHitam Recall@25%.
- Do not add another +1000 DN offset to `S2_SR_HARMONIZED`; harmonization is already applied. Do not model the carried topo table's temporary `row` sequence.
- Verifier failures with derived datasets are infrastructure errors: both `/shared/s007_datasets` and `/shared/datasets` pass audits then fail with read-only-filesystem errors before prediction. `/app/datasets` is rejected by its path guard. No model-side workaround was used to bypass the execution contract.

## Open leads, ranked

1. **Repair blind V staging, then evaluate the frozen nominee once.** `/shared/s007_nominee` contains the final code and pin; `/shared/s007_datasets` contains only its required `gee_s2_stress` dataset. Use these `shared_submission_dir` and `shared_datasets_dir` arguments after the host issue is fixed. Twelve queries remain.
2. **Independent spatial/prospective labels.** The bootstrap conditions on explored L predictions; repeated folds reuse the same labels. AirHitam has only 56 positives and Serting only 13 scored blocks.
3. **Operational budget validation including unmarked Middle.** Official AirHitam needs ~31.4% of binary-scored trees for 50% recall, and real inspection lists may be diluted by Middle. Do not claim a 25% majority-capture product from AP alone.
4. **Far-field uncertainty/extrapolation.** Disease pockets >80 m from labelled positives remain hard. Test a justified uncertainty-aware fallback or independent broad-region labels, rather than tuning to the same quadrant outcome.
5. **Finer crown geometry/new independent imagery.** Native 10–20 m S2 mixes palms at ~9 m spacing. Higher-resolution, canopy-centred sampling would be a new mechanism; another broad model sweep is not.

## How to resume

Read `/app/report/REPORT.md` and `/app/prior/MainChecklist.md`; keep all three estates equally trusted. The final pipeline is already staged and tested. Reproduce with:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/validate_final.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/diagnostics.py final --reference incumbent_all --reps 5000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/summarize.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/plots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/write_report.py
```

`validate_final.py` recomputes identical official/repeated/region comparisons. The report documents the second coordinates bootstrap and contract/prediction scripts. All feature extraction scripts are in their dataset directories. Runtime uses no network, no raw rasters, no prior reports and no training data paths. Source artifacts are read-only during `fit_predict`; writes are unnecessary.

## Artifact index

- `/app/submission/pipeline.py`, `requirements.txt` — frozen CPU/offline nominee; `/app/incumbent/` — untouched reference.
- `/app/datasets/gee_s2_stress/` — only required derived table, manifest and extraction script.
- `/app/datasets/alos_physics/`, `gee_s2_autumn/` — new rejected covariates with reproducible extraction and masking/NaN audits. Carried `alos_winstats/`, `gee_topo/` are unused.
- `/app/report/REPORT.md` — full methods, experimental/ablation tables, primary literature, caveats and reproduction commands.
- `/app/report/EXPERIMENTS.csv`, `EXPERIMENT_SUMMARY.csv`, `final_metrics.csv` — per-estate/protocol lift and paired deltas.
- `/app/research/final_oof.npz`, `incumbent_all_oof.npz`, `coords15_all_oof.npz`, `final_vs_*_bootstrap.json` — paired evidence.
- `/app/report/inspection_budget_diagnostics.csv`, `error_by_distance.csv`, `feature_effects.csv`, three PNG figures — operating/error/feature analysis.
- `/app/report/unlabelled_ranking.csv`, `.parquet` — all 4,473 target scores in original order, within-estate ranks, illustrative budget flag.
- `/app/report/contract_verification.json`, `dataset_audit.json`, `environment.json` — execution/data/environment audits.
- `/app/research/validate_final.py`, `diagnostics.py`, `summarize.py`, `plots.py`, `write_report.py`, `verify_contract.py`, `predict.py` — reproducible final evidence; other scripts/logs preserve the research campaign.
