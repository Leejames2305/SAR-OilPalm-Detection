## State of play

The verified nominee in `/app/submission/` is the incumbent NDVI plus local-coordinate logistic model. Fixed 5-fold 80 m spatial CV gives composite lift +0.1405, with per-estate lifts AirHitam +0.0310, Palong +0.2556, Serting +0.1350.

## What I tried

- ALOS window/statistic/change table: `/app/datasets/alos_winstats/features.parquet` (7,220 rows, 400 derived columns). Radius 1/2 statistics and two-date differences were extracted from local rasters with CRS-aware sampling.
- Fold-wise selected 260610 statistics: exploratory composite lift about +0.0692; rejected.
- Fold-wise selected 261014 statistics: exploratory composite lift about +0.0773; rejected.
- Fold-wise selected two-date deltas with degree-2 logistic: exploratory composite lift +0.1146 (AirHitam +0.0886, Palong +0.1582, Serting +0.0971); below incumbent and rejected.
- 45% score blend of incumbent and delta model: verifier composite +0.0948; rejected.
- Evidence: `/app/datasets/alos_winstats/extract.py`, `/app/report/REPORT.md`, `agentrsi cv --submission /app/submission`.

## Dead ends

Prior checklist and POC reports close broad RF/SMOTE, RVI, temporal optical/SAR, anomaly, clustering, and transfer sweeps. This session also closed the current ALOS window-statistic/change candidates because they failed the incumbent composite/per-estate acceptance criteria.

## Open leads, ranked

1. Obtain an independent ALOS date and test pre-registered change features on repeated spatial seeds.
2. Run far-field quadrant/region holdouts to quantify coordinate-prior degradation.
3. Add more estates for leave-one-estate-out validation.
4. Evaluate plot/row hotspot aggregation as a separate operational target.

## How to resume

Read `/app/prior/MainChecklist.md`, run `agentrsi selfcheck --submission /app/submission`, then `agentrsi cv --submission /app/submission`. Keep comparisons against both the incumbent and coordinates-only prior; preserve 80 m block-disjoint folds and per-estate reporting.

## Artifact index

- `/app/submission/pipeline.py` — verified nominee.
- `/app/submission/requirements.txt` — empty/comment-only dependency file.
- `/app/datasets/alos_winstats/features.parquet` — raster-derived covariates.
- `/app/datasets/alos_winstats/extract.py` — reproducible extraction script.
- `/app/datasets/alos_winstats/MANIFEST.md` — provenance and masking.
- `/app/report/REPORT.md` — technical report.
