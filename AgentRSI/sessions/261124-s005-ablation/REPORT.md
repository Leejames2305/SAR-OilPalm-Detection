# Oil-palm disease triage ablation report

## Objective and validation

The target is a ranking score for `Unhealthy` trees under severe imbalance. `Middle` rows are excluded from scoring. The primary metric is estate-wise average precision minus the estate prevalence, averaged equally across AirHitam, Palong, and Serting. Validation uses the supplied 80 m spatial blocks and five `GroupKFold` folds. All reported candidate scores below come from the same `agentrsi cv` implementation, so fold assignments and the NDVI reference are paired.

The deployed checkpoint is the compact coordinate-plus-NDVI logistic model in [`/app/submission/pipeline.py`](../submission/pipeline.py). Coordinates are converted from WGS84 to local metres using the labelled-estate centroid. AirHitam uses a linear model; Palong and Serting use standardized degree-2 terms. `C=0.1` and balanced class weights are fixed. The coordinates-only comparator is the same logistic recipe without NDVI; its fixed-fold lifts were approximately AirHitam `+0.030`, Palong `+0.122`, and Serting `+0.045`, showing that much of the ranking gain is spatial smoothness. The model is therefore an operational in-estate prior plus the optical signal, rather than evidence of a transferable disease mechanism.

## Controlled ablations

| Recipe | AirHitam lift | Palong lift | Serting lift | Composite | Verdict |
|---|---:|---:|---:|---:|---|
| Incumbent: linear AirHitam, degree-2 Palong/Serting, NDVI + coordinates | +0.0310 | +0.2556 | +0.1350 | **+0.1405** | Checkpoint |
| Degree 2 in all estates, `C=0.1` | +0.0180 | +0.2556 | +0.1350 | +0.1362 | AirHitam loss |
| Incumbent, `C=0.01` | +0.0331 | +0.2221 | +0.0987 | +0.1180 | Broad degradation |
| Incumbent, `C=0.03` | +0.0318 | +0.2521 | +0.1127 | +0.1322 | Broad degradation |
| Incumbent, `C=0.3` | +0.0308 | +0.2475 | +0.1444 | +0.1409 | Slight mean gain, Palong loss |
| Incumbent, `C=1.0` | +0.0308 | +0.2437 | +0.1321 | +0.1355 | Broad degradation |
| Linear NDVI + coordinates + all ALOS date-change summaries | +0.0470 | +0.0790 | +0.0579 | +0.0613 | Serting loss; reject |
| Broad ALOS window means/statistics + NDVI + coordinates | about +0.030 | about +0.091 | about +0.062 | below incumbent | Overfit/noise; reject |

The ALOS experiment used the existing `/app/datasets/alos_winstats` artifact (403 columns, two dates, 3x3 and 5x5 statistics). Features were merged on exact `estate`, `Long`, and `Lat` keys. The date-change candidate retained only features available across estates and imputed missing values within each training fold. It improved AirHitam relative to the incumbent but reduced Palong and Serting enough to fail the requirement of improving at least two estates without material degradation. The broader statistic families were even less stable, especially at radius 2.

These results isolate a useful simplification: the raw SAR and window-statistic families do not add enough independent signal to justify their variance at roughly 56--67 positives per estate. Degree-2 terms help where the spatial pattern is strong (Palong and Serting), while adding that capacity to AirHitam overfits. Stronger or weaker regularization also trades one estate against another rather than producing a consistent paired gain.

## Error and feature interpretation

NDVI is the only supplied non-spatial feature retained in the checkpoint. Its sign and effect are not stable enough across estates to support a pooled coefficient, and prior checks found cross-estate Cohen's-d sign flips. The coordinate terms capture clustered labels and should be interpreted as a deployment-specific spatial triage prior. The model must therefore be revalidated when the labelled footprint moves to a new region or estate.

Operationally, ranking quality matters more than threshold accuracy. The incumbent's spatial CV PR-AUC lifts are positive in all three estates and its top-fraction diagnostics should be read as a review-budget curve. The coordinates-only comparator is required because adjacent palms and disease pockets are spatially clustered; a large apparent lift can arise without a disease-responsive sensor feature. Random row splits and same-block nearest-neighbour features were excluded because they leak this structure.

## Literature context

The SAR feature definitions follow the Cloude--Pottier entropy/alpha decomposition and Yamaguchi model-based decomposition: Cloude and Pottier, *IEEE Transactions on Geoscience and Remote Sensing* 35 (1997), DOI [10.1109/36.551935](https://doi.org/10.1109/36.551935); Yamaguchi et al., *IEEE Transactions on Geoscience and Remote Sensing* 43 (2005), DOI [10.1109/TGRS.2005.853187](https://doi.org/10.1109/TGRS.2005.853187). Sentinel-2 surface reflectance and the red/NIR vegetation-index rationale are described by Drusch et al., *Remote Sensing of Environment* 120 (2012), DOI [10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026). These references motivate the covariates; they do not establish that any one feature is predictive for this estate.

## Limitations and next steps

The labelled positives are few, the unlabelled blocks are spatially disjoint, and all three estates occupy a narrow geographic setting. The current evidence supports an in-estate screening layer, not disease diagnosis or leave-estate-out transfer. A future improvement should use a new independent ALOS acquisition, canopy-centred sampling, or additional estates for validation. Any such feature must beat the checkpoint and the coordinates-only prior under repeated block or far-field region holdouts.
