## State of play

A valid nominee is in `/app/submission/pipeline.py`. It fits each estate independently using NDVI and local-metre coordinates, with a linear logistic model for AirHitam and regularized degree-2 spatial/NDVI terms for Palong and Serting. `requirements.txt` has no third-party pins because all imports are verifier-base packages.

## What I tried

- Incumbent benchmark: NDVI balanced logistic regression, composite lift `+0.0454` (AirHitam `+0.0146`, Palong `+0.0328`, Serting `+0.0887`). Evidence: `agentrsi cv --submission /app/incumbent`.
- Coordinates + NDVI linear logistic: composite lift about `+0.1032` in the first candidate check; all estates improved over NDVI. Evidence: `/tmp/sub` experiment during this session.
- Degree-2 coordinates/NDVI logistic (`C=0.1`): fixed 5-fold candidate lifts `+0.0180/+0.2556/+0.1350` for AirHitam/Palong/Serting when applied uniformly; repeated checks showed higher variance at AirHitam.
- Estate-specific nominee (linear AirHitam, degree-2 Palong/Serting): fixed candidate lifts `+0.0310/+0.2556/+0.1350`, composite `+0.1405`, with AP gains over incumbent in every estate. Evidence: `/app/report/REPORT.md`, final `agentrsi cv` output.
- Coordinates-only prior: fixed-fold AP `0.0882/0.2078/0.1469`; the nominee adds little over coordinates in AirHitam and a larger complement in Palong/Serting.

## Dead ends

Prior reports close RF/SMOTE sweeps, RVI families, temporal Sentinel-1/Sentinel-2 add-ons, anomaly detection, clustering, broad learner sweeps, and cross-estate pooling. Do not repeat these without a new mechanism. Spatial gains must always be compared with a coordinates-only prior.

## Open leads, ranked

1. Validate the nominee on additional spatial seeds and true far-field region holdouts; quantify confidence intervals for AP lift and top-budget recall.
2. Test a second independent ALOS-2 date or canopy-centred sampling, which could separate disease response from spatial smoothness.
3. Investigate plot/row hotspot aggregation as a separate operational target.
4. Add more estates for leave-one-estate-out validation before claiming transfer.

## How to resume

Read `/app/prior/MainChecklist.md`, then run `agentrsi selfcheck --submission /app/submission` and `agentrsi cv --submission /app/submission`. Preserve the exact 80 m block construction. If changing features, compare against both `/app/incumbent` and a coordinates-only model, report per-estate deltas, and refresh this handoff after each verified candidate.

## Artifact index

- `/app/submission/pipeline.py` — nominee `fit_predict()` implementation.
- `/app/submission/requirements.txt` — empty/comment-only dependency file.
- `/app/report/REPORT.md` — technical report, metrics, caveats, and references.
- `/app/prior/MainChecklist.md` — settled facts and closed experiments.
- `/app/prior/POCReport/PathForwardTrials.md` — prior spatial-fusion and transfer evidence.
