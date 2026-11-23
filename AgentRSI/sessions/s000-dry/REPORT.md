# Oil-palm Unhealthy triage — session s000-dry report

## Hypothesis
Per-estate spatial label prior (distance-weighted mean label of the K nearest labelled trees, from the training fold only) added to the NDVI logistic baseline will lift PR-AUC, because Unhealthy trees are spatially clustered (MainChecklist §A: 10-NN Unhealthy probability 0.12/0.34/0.26 vs 0.06–0.07 for Healthy).

## Method
- Candidate: `/app/submission/pipeline.py`. Features = [NDVI, log(kNN prior + 0.5·prevalence)], standardised, balanced LogisticRegression.
- Training rows get a leave-one-out kNN prior (self excluded) so the logistic fit does not see its own label. Validation rows use all training-fold labels.
- Evaluated with `agentrsi cv` (5-fold, 80 m spatial blocks, `GroupKFold`) on `L_labelled.csv`. Baseline = NDVI-only logistic (incumbent).

## Results (agentrsi cv, composite L-CV lift)
| Variant | AirHitam | Palong | Serting | Composite |
|---|---|---|---|---|
| Incumbent NDVI-LogReg | +0.0146 | +0.0328 | +0.0887 | +0.0454 |
| kNN prior K=5 | +0.0178 | +0.1389 | +0.1020 | +0.0863 |
| **kNN prior K=10 (submitted)** | +0.0138 | +0.1859 | +0.0791 | **+0.0929** |
| kNN prior K=20 | +0.0219 | +0.1566 | +0.0648 | +0.0811 |

Values are PR-AUC minus prevalence. Prevalence (L-set, scored rows): 0.0581 / 0.0857 / 0.1009.

## Feature and error analysis
- The gain is concentrated in Palong (+0.15 PR), where Unhealthy trees are the most clustered. It is absent or negative for AirHitam (−0.001) and Serting (−0.010 at K=10).
- Across K the composite stays above the baseline (+0.036 to +0.048), so the gain is not a single-K artefact. Serting is the weak estate and its sign depends on K.

## Caveats (important)
- The prior is an **in-estate spatial-neighbour signal**, not a disease signature. The checklist records that unseen, far-from-label pockets are near chance (pocket-holdout recall 0.26 vs 0.25 coin-flip). Its value depends on unlabelled trees lying near labelled ones, as in the product's sparse-label-over-a-field setting, but CV only partly tests that (80 m block edges leak neighbours).
- The 80 m block CV does not isolate far-field generalisation. A corner-holdout test (labels in one quadrant, score the rest) has not been run and is the next check.
- The handoff reported incumbent V-lift +0.054469 on a different protocol. The numbers here come only from `agentrsi cv`; I did not recompute on V/T.
- Requirements: no pins (numpy/scipy/pandas/scikit-learn only). selfcheck passes on all three estates.

## Literature
Not searched in this session (budget-limited). Spatial-label propagation / kNN smoothing for spatially clustered labels is standard in remote-sensing point classification.

## Artifacts
- Submission: `/app/submission/pipeline.py`, `/app/submission/requirements.txt`
- Scratch variants: `/tmp/exp/knn` (K=10), `/tmp/exp/k5`, `/tmp/exp/k20` (not deliverables)
