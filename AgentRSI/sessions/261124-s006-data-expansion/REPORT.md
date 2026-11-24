# Oil-palm disease triage: data-expansion report

## Objective and validation

The target is `Unhealthy` versus `Healthy`; `Middle` rows are excluded from scoring. The primary measure is the mean of estate PR-AUC minus estate prevalence on spatially disjoint 80 m blocks. Every candidate was compared with the incumbent coordinate plus NDVI logistic model and the NDVI-only reference. The supplied `agentrsi cv` uses five `GroupKFold` spatial folds.

## External covariates

Two leak-free Google Earth Engine datasets were extracted for all 7,220 labelled and unlabelled coordinates. `gee_s2_stress` uses masked Sentinel-2 Harmonized median composites for 2026-05-15–2026-07-15 and a matched 2025 window, including red-edge/SWIR bands, stress indices, and seasonal deltas. `gee_topo` uses SRTM elevation/slope/aspect plus MERIT Hydro HAND, upstream area, and relative elevation. Exact coordinate keys are restored from the input CSVs; both tables have zero missing values.

## Model experiments

The checkpoint incumbent CV was AirHitam +0.0310, Palong +0.2556, Serting +0.1350, composite +0.1405. A compact degree-2 logistic model using current S2 raw bands and coordinates was useful in AirHitam but degraded Palong/Serting. Prior-season S2 indices were complementary in Palong but unstable in AirHitam. A fixed 10% score blend of the incumbent and current-S2 raw-band model produced AirHitam +0.0355, Palong +0.2592, Serting +0.1316, composite +0.1421 in the final local `agentrsi cv` run. A fold-wise alpha selector was unstable (selected 0.05–0.85 across folds) and scored composite +0.1318, so it was rejected. Topography alone and S2-topography combinations did not consistently improve the incumbent and were rejected.

The fixed blend is retained as the submission because it improves the composite and two estates in the held-out block CV while avoiding a large estate-level change. Results are model-selection evidence; small differences remain sensitive to fold assignment. Coordinates-only remains a required comparator because spatial clustering can dominate scores.

## Limitations and caveats

Sentinel-2 10–20 m pixels mix neighbouring ~9 m palms. GEE extraction is a research artifact and is not called during deployment: `fit_predict` reads only the carried Parquet table through `ctx["datasets_dir"]`. The verifier runtime has no GEE package requirement. Seasonal deltas were evaluated but not selected for the final blend. Cross-estate sign consistency remains mixed, consistent with the prior checklist warning; this is why the final contribution is compact and heavily shrunk.

## Artifact paths

- `/app/submission/pipeline.py`
- `/app/datasets/gee_s2_stress/features.parquet`, `extract.py`, `MANIFEST.md`
- `/app/datasets/gee_topo/features.parquet`, `extract.py`, `MANIFEST.md`
- `/app/datasets/alos_winstats/features.parquet`
