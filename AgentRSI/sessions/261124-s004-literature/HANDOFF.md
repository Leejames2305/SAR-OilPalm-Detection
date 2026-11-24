## State of play

The verified nominee is `/app/submission/pipeline.py`: estate-wise balanced logistic regression on NDVI and local-metre coordinates, linear for AirHitam and degree-2 for Palong/Serting. Current `agentrsi cv` composite lift is +0.1405 versus the NDVI baseline; the previous verifier checkpoint is V lift +0.118751 versus the incumbent and +0.052289 versus coordinates-only. `/app/submission/requirements.txt` is base-only and selfcheck passes.

## What I tried

- Incumbent spatial-NDVI recipe: lifts +0.0310/+0.2556/+0.1350, composite +0.1405. Retained. Evidence: `agentrsi cv --submission /app/submission`.
- ALOS-2 3x3 window statistics (mean, SD, min, max, IQR for Spk/HAlpha/Yamaguchi over both dates): lifts +0.0288/+0.0836/+0.0299, composite +0.0474. Rejected because Serting degraded. Evidence: `/app/datasets/winstats/`, `/app/report/REPORT.md`.
- Two-date SAR log-change and H/A/alpha deltas: lifts +0.0202/+0.1120/+0.0723, composite +0.0682. Rejected.
- `clear_count` addition: lifts +0.0257/+0.2068/+0.0957, composite +0.1094. Rejected.
- Regularization checks (`C=.03`, `.3`) did not produce a consistent multi-estate gain.

## Dead ends

Prior reports close RF/SMOTE, RVI families, broad learner sweeps, anomaly detection, clustering, cross-estate pooling, temporal Sentinel-1/Sentinel-2 add-ons, and Middle-class relabeling. Coordinates remain a strong comparator; do not present spatial skill as transferable disease physiology.

## Open leads, ranked

1. Repeated spatial seeds and quadrant/region holdouts for confidence intervals and far-field robustness.
2. Independent second ALOS-2 L-band acquisition or canopy-centred sampling.
3. Plot/row hotspot aggregation as a separate operational target.
4. Additional estates for leave-one-estate-out validation.

## How to resume

Read `/app/prior/MainChecklist.md`, then run `agentrsi selfcheck --submission /app/submission` and `agentrsi cv --submission /app/submission`. Keep `Middle` excluded from scoring, preserve exact 80 m blocks, and compare every change with both the incumbent and coordinates-only prior. Derived data load through `ctx["datasets_dir"]` only.

## Artifact index

- `/app/submission/pipeline.py` — verified nominee.
- `/app/submission/requirements.txt` — exact dependency pins (none required).
- `/app/datasets/winstats/features.parquet` — reproducible raster window covariates, not promoted.
- `/app/datasets/winstats/extract.py` and `MANIFEST.md` — extraction recipe and provenance.
- `/app/report/REPORT.md` — literature, experiments, metrics, and caveats.
- `/app/prior/MainChecklist.md` — settled facts and closed experiments.
