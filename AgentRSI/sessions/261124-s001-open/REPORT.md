# Oil-palm disease triage model

## Objective and validation

The model ranks `Unhealthy` trees for a fixed inspection budget. `Healthy` is the negative class; `Middle` rows are excluded from scoring. The primary metric is the unweighted estate mean of average-precision lift (`AP - prevalence`) on spatially held-out 80 m blocks. Secondary metrics are recall in the top 25%, enrichment in the top 10%, and ROC-AUC. Group keys use the prescribed local-metre transform from longitude/latitude, and all preprocessing is fitted inside each training fold.

The incumbent is balanced logistic regression on NDVI. The nominee is a balanced logistic model on NDVI plus local-metre coordinates. For AirHitam it uses linear terms; for Palong and Serting it adds degree-2 interactions and squares with `C=0.1`. This compact estate-specific choice was made from repeated block checks: the quadratic form was less stable at AirHitam's lower signal level, while it was consistently stronger in the other two estates.

## Main experiment

The following table is the exact `agentrsi cv` 5-fold result. The baseline column is the CLI's independently fitted NDVI model; prevalence is computed after dropping `Middle`.

| Estate | Prevalence | Incumbent AP / lift | Nominee AP / lift | AP delta | ROC-AUC | Recall@25% | Enrichment@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| AirHitam | 0.0581 | 0.0727 / +0.0146 | 0.0891 / +0.0310 | +0.0164 | 0.653 | 0.482 | 1.25x |
| Palong | 0.0857 | 0.1186 / +0.0328 | 0.3414 / +0.2556 | +0.2228 | 0.819 | 0.742 | 4.55x |
| Serting | 0.1009 | 0.1896 / +0.0887 | 0.2359 / +0.1350 | +0.0463 | 0.737 | 0.612 | 2.84x |
| **Composite mean lift** | — | **+0.0454** | **+0.1405** | **+0.0951** | — | — | — |

The AP, lift, and composite values above are from the required CLI. The secondary metrics use the same out-of-fold scores and tie-aware top-fraction selection. All three estates improve over the incumbent. The Palong gain is the largest and reflects strong spatial clustering; it should be interpreted as an in-estate triage result.

## Coordinates-only disentanglement

A coordinates-only balanced logistic spatial prior gave AP values of 0.0882, 0.2078, and 0.1469 for AirHitam, Palong, and Serting in the same fixed folds. Thus the nominee's AP increment over the coordinate prior is approximately +0.0009, +0.1336, and +0.0890 respectively. AirHitam's improvement is primarily spatial; Palong and Serting retain a substantial NDVI/nonlinear complement. This comparison prevents the large composite gain from being reported as a pure disease-spectrum signal.

A repeated 10-seed shuffled 5-fold block check gave nominee AP means (standard deviations) of 0.0836 (0.0033), 0.3339 (0.0521), and 0.2508 (0.0432) for AirHitam, Palong, and Serting. The quadratic estates have higher fold variance because the number and arrangement of spatial blocks are limited, especially in Serting. A four-quadrant leave-one-region-out check preserved positive lift over prevalence for all estates, although AirHitam's result was close to its coordinate prior.

## Features and error analysis

NDVI is the only non-spatial input used. It is available in both labelled and unlabelled tables, requires no derived dataset, and is transformed only through the in-fold scaler. Coordinates are converted from EPSG:4326 to approximate local metres using the training-estate latitude. Polynomial expansion supplies smooth spatial curvature and NDVI interactions; regularization limits the effect of the small positive class.

The cross-estate sign warning remains relevant: prior SAR and optical effect sizes can flip sign between estates. The nominee therefore does not pool a global coefficient across estates. Each estate is fit independently in its subprocess, and no labels from `ctx["other_labeled"]` are used. This respects the equal trust of all three verified label sets while avoiding an unsupported transfer assumption.

The main failure mode is unseen spatial structure. A high score can identify a location similar to labelled disease clusters without proving a transferable canopy-level disease mechanism. Block-disjoint CV is more realistic than random CV, but the labelled sample still contains relatively few positive blocks. Unlabelled `Middle` trees remain in the returned prediction array and are never reordered or removed; their scores are simply outside the evaluation target.

## Reproducibility and execution

`/app/submission/pipeline.py` imports only NumPy, pandas, and scikit-learn packages supplied by the verifier. `requirements.txt` is intentionally empty apart from a comment. `agentrsi selfcheck --submission /app/submission` passes for all three estates, and the full fit takes well below the runtime cap. No raster, network, Earth Engine, or credential access occurs in `fit_predict()`.

## Literature context

The use of NDVI follows the normalized-difference vegetation index introduced by Rouse et al. (1974), while the decision to treat spatial structure as a validation concern follows Roberts et al. (2017), who show that ordinary random cross-validation can be optimistic for spatially structured ecological data. The SAR decomposition context is consistent with Cloude and Pottier (1997), but the prior experiments found no stable independent RVI/SAR lift here. These references motivate the measurement choices; they are not claims that the present score is a mechanistic disease detector.

* Rouse, J. W. et al. (1974), “Monitoring the vernal advancement and retrogradation of natural vegetation,” NASA/GSFC Type III final report.
* Cloude, S. R. & Pottier, E. (1997), “An entropy based classification scheme for land applications of polarimetric SAR,” *IEEE Transactions on Geoscience and Remote Sensing*, 35(1), 68–78. https://doi.org/10.1109/36.551935
* Roberts, D. R. et al. (2017), “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881

## Decision

The nominee is a verified improvement over the incumbent for the stated in-estate ranking task and is saved in `/app/submission`. It should be operated as a screening priority score with a declared review budget, and its coordinate dependence should be reported whenever results are presented.
