# Session 261127-s015-open

## State of play

**The final submission retains s014 source_prior byte-for-byte. No changed candidate earned replacement.** The designated incumbent under `/app/incumbent` remains untouched. All estates have equally trusted labels. The submitted recipe has no estate-name branches, target label-neighbor scores, Middle target, or inference-time network calls.

Official union-anchored 80 m lifts, AirHitam / Palong / Serting: **+.159644 / +.311518 / +.391065**, composite **+.287409**. Comparators: incumbent +.122984, coordinates +.043863, raw −NDVI +.050841, s009 +.160430.

This session's genuine V reference: **+.224729** composite, delta incumbent +.105979 and coordinates +.158268. Estate lifts: **+.058767 / +.344529 / +.270893**. Historical s009 AirHitam V +.065513 remains unsurpassed by a candidate preserving the strong composite. Twelve of fifteen V queries were used; three remain. No additional V tuning is recommended.

Ten frozen group assignments, seeds 901–910: selected composite beats incumbent, coordinates and s009 **10/10**. Mean estate lifts +.12651 / +.29470 / +.33037; composite +.25053. Mean delta incumbent +.11525. Palong loses to incumbent once, by −.01707. Conditional paired 4,000-block-bootstrap composite delta incumbent 95% interval [+.08335, +.23425], coordinates [+.15989, +.30441]. Palong's individual incumbent interval crosses zero. These checks reuse L and omit adaptive selection and model-refitting uncertainty.

Recall at 25% binary inspection capacity: **48.2% / 65.2% / 74.6%**. Including Middle capacity: **48.2% / 63.6% / 73.1%**. AirHitam majority capture remains unmet. Serting finds 0/10 positives located 80–150 m from training positives. All 4,473 aligned U rankings are exported; U health outcomes and V recall are unknown.

**Central audit finding:** 92 AirHitam U trees below −2.5 training SD in raw NDVI average the 21st risk percentile under s014, versus the 83rd under s009. Only 1/92 enters s014's top quarter. But **all 16 labelled AirHitam trees below −3 SD are Healthy**. Therefore the ranking reversal and density shift are confirmed; wrong disease predictions on U are not established. Only two U points are below the absolute L NDVI minimum. Forcing all low-NDVI crowns upward is not empirically justified.

Final selfcheck, CLI CV, import audit, data audit and fresh-subprocess checks pass. Runtime ≤.289 s per estate, peak RSS ≤275 MiB in fresh four-CPU processes. Outputs match exported scores exactly. Twelve feature tables cover all 7,220 exact keys with zero NaNs, duplicates, forbidden names or symlinks. Total datasets size is about 55.84 MiB. Runtime needs only `triage_v12` and `dino_sar`; the sole non-base pin is `threadpoolctl==3.8.0`. GPU job `job-0001-a350d2` completed, outputs were fetched, and its VM was stopped. No remote resources remain active.

## What I tried

There are 86 recipes/comparators plus 11 immutable V aliases, and 1,095 completed estate/protocol records. All local metrics below use corrected union origins. Full per-estate and paired results are in `OFFICIAL_SUMMARY.csv` and `EXPERIMENT_SUMMARY.csv`.

| Mechanism | L lift A / P / S | L composite | V composite | Verdict |
| --- | --- | --- | --- | --- |
| Retained s014 | .159644 / .311518 / .391065 | .287409 | .224729 | Strongest verified reference |
| Clip ±3, C=.02 | .134924 / .288375 / .263817 | .229039 | .188324 | AirHitam V .065385; other estates regress |
| Support fallback | .113318 / .284625 / .269016 | .222320 | .194671 | AirHitam V .065055; Palong loses |
| Sign-consistent additive fit | .156222 / .282498 / .334368 | .257696 | .193539 | Restricted pooling fails |
| Physical projection and prior | .173838 / .321183 / .363791 | .286271 | .216294 | Buffered AirHitam gain fails to transfer |
| Clustering-based ridge | .072578 / .305941 / .368307 | .248942 | .210023 | Generic adaptation unstable |
| Training covariance constraint | .184351 / .311518 / .388982 | .294950 | .218169 | Large L gain, no V gain |
| Protected OOF residual | .115492 / .317229 / .265672 | .232798 | .176784 | Sacrifices other estates |
| Block-robust loss, DRO02 | .165307 / .313053 / .380211 | .286190 | .224045 | Broad CV gains; AirHitam V .048047 |
| New later-season PSRI | .160852 / .316114 / .386532 | .287832 | .223247 | No verified increment |
| New temporal SAR CNN | .159451 / .321607 / .384691 | .288583 | .223052 | No V or coarse-holdout gain |
| Inner-quadrant selector | .134924 / .291261 / .391065 | .272417 | .224085 | Complexity adds no benefit |

Other controlled families include clipped and hinged polynomials, C=.002–.15, differential nuisance penalties, compact physical/red-edge bases, linear/spatial/full protected cores, grouped optical features, context projection, sign bounds, no pooling, training/target moment constraints, derivative-box constraints, buffered inner selection, clustering-conditioned clipping/support guards, robust-loss temperatures, later-season physical/change features, and CNN replacement/strength/residual ablations. Evidence: `round1.log` through `round10.log`, actual OOF files under `research/results`, and complete paired tables.

Two new label-free datasets are reproducible:

- `late_optical`, 7,220 × 93: harmonized S2 April 15–August 15 versus September 1–November 25; SCL4 and CloudScore+ ≥.65; native 20 m sampling; optical/red-edge/moisture/senescence medians, percentiles and changes. Later NDRE2 marginal d is −.502 / −.490 / −.686; PSRI is +.228 / +.339 / +.425. Consistent signs do not establish incremental prediction value.
- `temporal_sar`, 7,220 × 35: two-date, 22-channel, 7×7 (~45 m) polarimetric patches; a denoising CNN with optical-covariate auxiliary prediction, followed by label-free PCA. No disease labels enter training. Native CRS and signed Yamaguchi masking are correct. The representation is transductive and optical-informed. Encoder, PCA, scripts and audits are saved. Neither new dataset is used by final inference.

## Dead ends

- Respect `MainChecklist.md`: do not repeat mean-only SAR, imbalance/model grids, random CV, label-neighbor leakage or Middle weak-positive targets.
- Strong clipping/ridge, support fallback, global sign restrictions and compact protected residuals did not preserve Palong and Serting V. An AirHitam-only gain is insufficient.
- Marginal and conditional signs differ under collinearity. Texture SD/edge marginal signs are negative in AirHitam and positive elsewhere; height reverses in the other direction. Source-sign restrictions were tested and rejected as improvements.
- Covariance or derivative constraints can raise L without V gains. Holding derived covariates fixed while changing raw NDVI measures a partial derivative, not a physical canopy intervention.
- Clustering-based regularization and inner-region/buffer selectors did not solve the L/V disconnect. Inner-region mean repeat composite is −.01308 versus retained, with only 1/10 wins.
- DRO02 mean repeat increment is +.00021 (5/10 wins), later PSRI −.00006 (4/10), CNN −.00449 (2/10). V composites around .223–.224 are noise-scale ties. Prefer the retained package.
- Never mix historical L-only origins with current union geometry. Faithful s009 retains its legacy inner origin and inherited estate-name branch only as a reference.

## Open leads, ranked

1. Prospectively inspect original 80 m blocks containing AirHitam low-NDVI crowns and s009/s014 disagreements. Include unranked random audits in all three estates. Distinguish disease from empty, young or soil-mixed crowns and check crown registration.
2. Validate on independent estates or years, including unseen disease pockets and contrasting management/stand regimes. More same-label CV/V has diminishing value.
3. Test source transfer under independently observed conditional shifts. Current marginal signs cannot establish invariant physiological coefficients. Preserve physical and full-context references without V-derived estate branches.
4. Keep the new imagery/encoder assets for independent data. They are reproducible covariates, not demonstrated improvements; further current-data micro-sweeps are low value.

## How to resume

Read `REPORT.md`, `selection.json`, `selection_plan.json`, `prior/MainChecklist.md`, `VERIFIER_SUMMARY.csv`, `L_tail_labels.csv` and `covariate_shift.csv`.

Exact V snapshots live under `research/verified`. Use `v_NAME` wrappers, which load uniquely named helper modules, to avoid shared Python import-cache contamination. Exploratory helpers evolved during research; immutable V reruns and recorded hashes are authoritative. Submission is byte-identical to previous s014 and this session's V query 1.

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords raw_ndvi s009 s012 s013 s014 --protocols official block160 buffer80 blockquad blockquad80
OPENBLAS_NUM_THREADS=1 python /app/research/evaluate.py --models v_clip3_C02 v_guard25 v_sign_consistent v_stress_project_physical v_regime_ridge v_moment_train v_resid_protect_full v_dro02 v_late_psri v_ae_sar_add v_inner_region --protocols official
OPENBLAS_NUM_THREADS=1 python /app/research/confirmation.py
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python /app/research/final_diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/audit.py
OPENBLAS_NUM_THREADS=1 python /app/research/contract_subprocess.py
OPENBLAS_NUM_THREADS=1 python /app/research/plots.py
OPENBLAS_NUM_THREADS=1 python /app/research/write_report.py
```

The source archive `datasets/triage_v12/recipes/s015` contains `selected`, `parents`, `candidates`, `verified`, `reproduction` and instructions. Restore `reproduction/*.py` to `research` and the other folders alongside. Metrics CSVs and fitted score tables remain outside datasets. Do not rerun `freeze_verified.py` without the original `/shared` snapshots; archived wrappers already reference immutable helpers. Research plotting uses matplotlib 3.10.7, separately from submission requirements. No GPU is active.

## Artifact index

- `submission/pipeline.py`, `requirements.txt`: final unchanged offline CPU recipe.
- `report/REPORT.md`, `HANDOFF.md`, `selection.json`, `selection_plan.json`: conclusions and decision.
- `EXPERIMENTS.csv`, `EXPERIMENT_SUMMARY.csv`, `OFFICIAL_SUMMARY.csv`, `VERIFIER_SUMMARY.csv`: actual paired results and immutable V aliases.
- `verifier/full_history.json`, `per_estate_results.json`, `research/verified`: genuine query history, estate values and frozen packages.
- `frozen_repeat_plan.json`, `repeat_comparison.csv`, `bootstrap.csv`, `bootstrap_method.json`: stability and conditional uncertainty.
- `tail_predictions.csv`, `parent_u_predictions.parquet`, `covariate_shift.csv`, `L_tail_labels.csv`, `marginal_signs.csv`, `regime_diagnostics.csv`, `ndvi_counterfactual.csv`: support and sign audits.
- `final_coefficients.csv`, `partial_derivatives.csv`, `diagnostic_model_equivalence.json`: exact selected coefficients and partial-response diagnostics.
- `inspection_diagnostics.csv`, `error_positive_distance.csv`, `unlabelled_ranking.parquet`/`.csv`: inspection metrics and all 4,473 rankings.
- `selfcheck.txt`, `official_cv.txt`, `import_audit.json`, `dataset_audit.json`, `contract_verification.json`, `fresh_subprocess_contract.json`: execution verification.
- `tail_ranking`, `verifier_comparison`, `inspection_curves` PNG/PDF pairs; `literature`: standalone figures and primary references. Web search failed with service errors; direct primary-page/metadata requests worked, with access limits recorded.
- `datasets/late_optical`, `temporal_sar`: new covariates, manifests, extraction/training scripts and audits.
- `datasets/triage_v12/recipes/s015`: reproducible source archive and checksums. Runtime still needs only `triage_v12` and `dino_sar`.
