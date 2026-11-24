## State of play

The strongest verified nominee remains `/app/submission/pipeline.py`, copied from `/app/incumbent/pipeline.py`. It uses NDVI and local-metre coordinates, with linear logistic regression for AirHitam and regularized degree-2 terms for Palong and Serting. `requirements.txt` is empty of third-party pins because imports are verifier-base packages. `agentrsi cv` reports composite lift `+0.1405` (AirHitam `+0.0310`, Palong `+0.2556`, Serting `+0.1350`).

## What I tried

- **Regularization sweep:** `C=0.01/0.03/0.3/1.0`; composite lifts `+0.1180/+0.1322/+0.1409/+0.1355`. `C=0.3` has a tiny mean increase but materially lowers Palong (`+0.2475` vs `+0.2556`), so the incumbent is retained.
- **Degree simplification:** degree 2 in all estates gives `+0.1362` and lowers AirHitam to `+0.0180`; reject.
- **ALOS window statistics:** broad means/statistics reduce Palong/Serting; reject for variance and missingness.
- **Two-date ALOS change:** compact date-change features give `+0.0613` composite (`+0.0470/+0.0790/+0.0579`), below the checkpoint and below it on Serting; reject.

Evidence: `/app/report/REPORT.md`, `/app/datasets/alos_winstats/`, and the `agentrsi cv` outputs from this session.

## Dead ends

Prior reports close RF/SMOTE sweeps, RVI families, temporal Sentinel-1/Sentinel-2 add-ons, anomaly detection, clustering, broad learner sweeps, cross-estate pooling, and treating `Middle` as a positive. The current ablations also close broad ALOS window-statistic additions and unvalidated date-change fusion.

## Open leads, ranked

1. Validate the checkpoint on additional spatial seeds and quadrant/region holdouts.
2. Obtain an independent ALOS-2 acquisition or canopy-centred extraction and test it with the same paired protocol.
3. Evaluate plot/row hotspot aggregation as a separate operational target.
4. Add independent estates for leave-estate-out validation.

## How to resume

Read `/app/prior/MainChecklist.md`, then run `AGENTRSI_CPUS=3 agentrsi selfcheck --submission /app/submission` and `AGENTRSI_CPUS=3 agentrsi cv --submission /app/submission`. Preserve exact 80 m groups, compare every candidate with both the checkpoint and coordinates-only prior, and refresh this file after each verified experiment.

## Artifact index

- `/app/submission/pipeline.py` — retained nominee.
- `/app/submission/requirements.txt` — exact dependency contract; no extra pins.
- `/app/datasets/alos_winstats/features.parquet` — existing ALOS window-statistics artifact, not used by the nominee.
- `/app/datasets/alos_winstats/MANIFEST.md` and `extract.py` — provenance and reproducible extraction.
- `/app/report/REPORT.md` — ablation report.
- `/app/prior/MainChecklist.md` — settled facts and closed experiments.
