# Oil-palm disease triage model report

## Objective and validation

The target is binary triage of `Unhealthy` versus `Healthy` trees. `Middle` rows are excluded from scoring. The primary score is the unweighted mean, over AirHitam, Palong, and Serting, of spatial 80 m block PR-AUC minus estate prevalence. All three label sets are treated as equally verified. Validation uses the supplied `agentrsi cv` five-fold spatial-block protocol; coordinates-only performance is the required spatial-smoothness comparator.

## Verified nominee

`/app/submission/pipeline.py` fits each estate independently. It converts WGS84 coordinates to local metres, combines them with NDVI, standardizes the three predictors, and fits balanced logistic regression. AirHitam uses linear terms; Palong and Serting use degree-2 terms with `C=0.1`. This compact recipe is deterministic, uses only verifier-base packages, and returns one finite score per unlabelled row.

The current spatial CV result is:

| estate | prevalence | candidate PR-AUC | candidate lift | NDVI PR-AUC | delta PR-AUC |
|---|---:|---:|---:|---:|---:|
| AirHitam | 0.0581 | 0.0891 | +0.0310 | 0.0727 | +0.0164 |
| Palong | 0.0857 | 0.3414 | +0.2556 | 0.1186 | +0.2228 |
| Serting | 0.1009 | 0.2359 | +0.1350 | 0.1896 | +0.0463 |
| composite mean lift | — | — | **+0.1405** | — | **+0.0951** |

The prior verifier checkpoint reports V lift `+0.118751` versus the earlier incumbent and `+0.052289` versus coordinates-only, with estate lifts `+0.037028/+0.253306/+0.065918`. The candidate improves all three estates over the NDVI baseline, while most of its gain is spatial context. Therefore it is an in-estate ranking aid rather than evidence of a transferable disease signature.

## Literature-guided hypotheses

Published oil-palm Ganoderma work supports optical stress indices and supervised ranking as plausible signals, while also motivating caution about small samples and disease stage. Izzuddin et al. developed spectral indices for early Ganoderma detection in seedlings ([doi:10.1080/01431161.2017.1335908](https://doi.org/10.1080/01431161.2017.1335908)). Santoso et al. reported a Random Forest basal-stem-rot classification study in plantations ([doi:10.1080/01431161.2017.1331474](https://doi.org/10.1080/01431161.2017.1331474)). Noor Azmi et al. evaluated SVM detection in seedlings ([doi:10.3390/rs12233920](https://doi.org/10.3390/rs12233920)). A field study explicitly combining C- and L-band SAR backscatter time series for Ganoderma detection is also reported at [doi:10.1088/1755-1315/1412/1/012003](https://doi.org/10.1088/1755-1315/1412/1/012003), motivating the second-date change ablation here. These papers motivate testing optical stress, polarimetric descriptors, and compact regularized learners, but their seedling/plantation protocols do not establish transfer to this spatially disjoint Malaysian estate split.

## Experiments this session

| hypothesis | implementation | AirHitam lift | Palong lift | Serting lift | composite | verdict |
|---|---|---:|---:|---:|---:|---|
| incumbent spatial-NDVI recipe | local-metre coordinates + NDVI; degree-2 for Palong/Serting | +0.0310 | +0.2556 | +0.1350 | +0.1405 | retained |
| stronger L2 regularization | same, `C=0.03` | +0.0318 | +0.2521 | +0.1127 | +0.1322 | rejected |
| weaker regularization | same, `C=0.3` | +0.0308 | +0.2475 | +0.1444 | +0.1409 | tiny aggregate change, AirHitam/Palong lower; rejected |
| second-date SAR change | log changes in HH/HV/VH/VV, H/A/alpha plus NDVI and coordinates | +0.0202 | +0.1120 | +0.0723 | +0.0682 | rejected |
| clear-count quality covariate | NDVI + `clear_count` + spatial terms | +0.0257 | +0.2068 | +0.0957 | +0.1094 | rejected |
| 3x3 raster window statistics | derived mean/SD/min/max/IQR for ALOS-2 Spk, HAlpha, and Yamaguchi bands, linear logistic | +0.0288 | +0.0836 | +0.0299 | +0.0474 | rejected |

The raster extraction is preserved at `/app/datasets/winstats/` for reproducibility. It uses native CRS reprojection, positional band order from the data specification, finite/positive masking for linear Spk/HAlpha bands, finite-only masking for Yamaguchi dB, and exact coordinate keys for all 7,220 trees. The features were not promoted because they weakened Serting and did not beat the verified nominee.

## Error and feature interpretation

Coordinates explain a large share of PR-AUC, consistent with clustered disease labels. The nominee should therefore be interpreted as a within-estate inspection ranking. Cross-estate sign flips in Cohen's d and weak leave-estate-out results in the prior reports caution against calling NDVI or SAR coefficients universal disease physiology. The model is intentionally low-capacity because prior RF, SVM, MLP, anomaly, RVI, temporal Sentinel-1/Sentinel-2, and imbalance sweeps either tied or degraded under block-honest validation. A top-fraction review policy is the operational output; raw accuracy is uninformative under 6–10% prevalence.

## Limitations and next tests

The data contain only three estates and one labelled spatial slice per estate. The current nominee has no explicit canopy-centroid correction, independent second ALOS-2 acquisition beyond the supplied tabular date pair, or additional estate transfer validation. Future work should prioritize repeated block seeds, quadrant/region holdouts, canopy-centred sampling, and a genuinely independent L-band change acquisition. Any spatial gain must remain paired with a coordinates-only baseline.
