# Oil-palm disease triage research report

## Objective and validation

The target is `Unhealthy` versus `Healthy`; `Middle` rows are excluded from scoring. The primary metric is estate-mean average-precision lift over prevalence on 80 m spatial-block held-out folds. Coordinates are converted from EPSG:4326 to local metres. All three estates are treated as equally trusted verified labels.

## Verified baseline

The submitted incumbent is a compact balanced logistic model using NDVI and local coordinates, with degree-2 terms for Palong and Serting. `agentrsi cv --submission /app/submission` reports:

| estate | prevalence | PR-AUC lift | ROC-AUC |
|---|---:|---:|---:|
| AirHitam | 0.0581 | 0.0310 | not emitted by CLI |
| Palong | 0.0857 | 0.2556 | not emitted by CLI |
| Serting | 0.1009 | 0.1350 | not emitted by CLI |
| composite | — | **0.1405** | — |

The incumbent improves every estate over the NDVI-only reference in this fixed 5-fold block protocol.

## Data expansion experiment

`/app/datasets/alos_winstats/features.parquet` contains 7,220 exact coordinate keys and 400 derived ALOS-2 window/statistic/change columns. It was extracted from the local GeoTIFFs after CRS reprojection, using finite/positive support masks for linear power and H-Alpha, T11 support for the AirHitam T3 raster, and finite signed values for Yamaguchi dB. Radius-1 and radius-2 window summaries include mean, standard deviation, extrema, percentiles, and valid count; matched-date differences and ratios are included. Estate-specific ROI coverage creates missing columns in Palong/Serting, which are imputed from training medians during experiments.

A block-honest screening script ranked candidate change/statistic columns inside each training fold and fit regularized logistic models. The best exploratory family was two-date ALOS delta features with a degree-2 model: AirHitam +0.0886, Palong +0.1582, Serting +0.0971 lift (composite +0.1146), below the incumbent. A 261014 family reached +0.1215 composite in exploratory fold screening but reduced Serting and used fold-wise feature selection with high variance. Blending these scores with the incumbent was unstable; a 45% blend scored +0.0948 in the verifier CLI. These candidates were rejected because they failed the incumbent comparison and/or materially degraded an estate.

The final submission therefore remains the incumbent pipeline. The raster feature table and extraction script are retained as reproducible artifacts for a future independent-date or far-field validation campaign.

## Error analysis and caveats

Coordinates explain much of the repeatable gain, so spatial smoothness can be mistaken for disease signal. The model should be interpreted as an in-estate ranking layer. Cross-estate feature signs are inconsistent, especially for SAR-derived statistics, so pooled coefficients or a single global sign are unsafe. Small positive counts and spatially clustered labels make feature selection noisy; all exploratory additions were checked against per-estate block scores and rejected when they harmed an estate.

The incumbent uses only verifier-available packages and performs no network or raster access at prediction time. The optional derived dataset is loaded through `ctx["datasets_dir"]` only by exploratory code and is not required by the final pipeline.

## Literature context

The design follows established remote-sensing practice: NDVI is a canopy-vigor proxy, L-band backscatter and Cloude–Pottier/Yamaguchi decompositions are physically interpretable but sensitive to geometry and moisture, and spatially blocked validation is required when labels cluster at plantation scale. The present data support NDVI plus spatial context as the most reproducible triage signal; they do not establish a transferable disease-specific SAR signature.
