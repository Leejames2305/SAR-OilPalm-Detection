# POC: RVI (ALOS quad-pol) vs NDVI (Sentinel-2) for Healthy/Unhealthy separation - AirHitam

**Date:** 5 October 2026


**Scope:** AirHitam only (2511 trees: 2351 Healthy / 160 Unhealthy @ 6.37% positive) -
  the sole estate with one-to-one field ground truth.


**Question:** Can RVI separate the classes statistically? If not, does adding
  Sentinel-2 NDVI help?

## Verdict

1. **RVI barely separate Healthy from Unhealthy.**
   Four dual-pol-style ratios and the true eigenvalue quad-pol RVI are all coin flips (|d| <= 0.04, MWU p 0.61-0.99, ROC-AUC ~0.50, PR lift ~0).
2. **NDVI can - modestly but robustly.**
   d = -0.51 (medium), p = 3.6e-11, PR-AUC 0.130 vs 0.064 no-skill (~2x enrichment), ROC-AUC 0.65, and the signal survives honest spatial-block CV (0.133 / 0.653).
3. **Fusion adds nothing over NDVI alone.** 
   RVI+NDVI == NDVI (spatial PR 0.132 vs 0.133); +HV is within noise (0.135). RVI contributes zero ranking power.
4. **Mechanism:** 
   Unhealthy trees are dimmer in all SAR channels proportionally (HV d=-0.34, span d=-0.27), so every ratio normalisation erases the level shift. Optical NDVI measures a different physical quantit (chlorophyll/red-edge absorption) that ratios do not cancel.
5. **No ML pushes past NDVI.** 
   Best honest spatial PR 0.14 (+RVI/LogReg) vs 0.133 NDVI-only (noise); RF/XGB trail linear LogReg everywhere andMLP collapses below no-skill. Capacity hurts at n=160 positives.
6. **Red-edge/SWIR is the only genuine lift** 
   (spatial PR 0.145, both protocols agree); temporal NDVI over +/-90 d is flat.


## Data & Method

- **SAR:** 
  - `ALOS2-Subset_AirHitam_260610_Cal_ML_Spk_TC.tif` ([HH, HV, VH, VV], EPSG:32648, 6.43 m; Scene date 2026-06-10)
  
  - `ALOS2-Subset_AirHitam_260610_Cal_mat_Spk_TC.ti` - T3 coherence
( T11, T12re/im, T13re/im, T22, T23re/im, T33, EPSG:4326)
  - Per-tree 3x3 means
  - RVI variants:
    - RVI_classic: `8HV/(HH+VV+2HV)`
    - RVI_xavg: Cross-pol averaged)
    - RVI_hh: `4HV/(HH+HV)`, RVI_vv: `4VH/(VV+VH)`
    - RFDI
    - Eigenvalue quad-pol RVI, RVI_qp: `4lam3/span` ; from per-tree mean-T3
(eigen-decomposition | 2511/2511 windows sampled)
    
- **Optical:** 
  - `COPERNICUS/S2_SR_HARMONIZED` median composite (+/-30 d around SAR: 2026-June-10),
  - NDVI: `(B8-B4)/(B8+B4)` sampled at tree coordinates (10 m)
  - Coverage: 0/2511 missing; 20-29 clear obs/tree

## Results

### Univariate (Healthy vs Unhealthy)

| Feature | Cohen d | MWU p | ROC-AUC* | PR-AUC (lift) |
|---|---:|---:|---:|---:|
| NDVI | **-0.51** | **3.6e-11** | **0.66** | **0.130 (+0.066)** |
| RVI_classic | -0.04 | 0.61 | 0.49 | 0.074 (+0.010) |
| RVI_xavg | -0.01 | 0.99 | 0.50 | 0.066 (+0.003) |
| RVI_qp (eigenvalue) | -0.01 | 0.89 | 0.50 | 0.064 (+0.000) |
| RFDI | +0.01 | 0.96 | 0.50 | 0.082 (+0.018) |
| HV (ref) | -0.34 | 1.4e-5 | 0.60 | 0.111 (+0.048) |
| span (ref) | -0.27 | 2.9e-4 | 0.59 | 0.091 (+0.027) |

\*direction-optimised (Unhealthy = lower value).

### Fusion (balanced LR)

| Features | random PR / ROC (ceiling) | spatial-block PR / ROC (honest) |
|---|---:|---:|
| RVI only | 0.079 / 0.49 | 0.071 / 0.47 |
| RVIqp only | 0.062 / 0.46 | 0.072 / 0.47 |
| **NDVI only** | **0.144 / 0.66** | **0.133 / 0.65** |
| RVI+NDVI | 0.142 / 0.65 | 0.132 / 0.65 |
| RVI+NDVI+HV | 0.146 / 0.65 | 0.135 / 0.65 |
  
  
### Rerun on non-multilook scenes


(`ALOS2-Subset_AirHitam_260610_Cal_Spk_TC.tif`: 4 bands [HH, HV, VH, VV],
EPSG:4326, ~5.1 m grid, 444x578, coregistered with the T3 subset; 3x3 ~ 15 m
vs ~19 m before
| Feature | old d | new d | new p | new PR lift | changed? |
|---|---:|---:|---:|---:|---|
| RVI_classic | -0.04 | -0.09 | 0.56 | +0.018 | no (still null) |
| RVI_xavg | -0.01 | -0.04 | 0.91 | +0.013 | no |
| RVI_qp | -0.01 | -0.01 | 0.89 | +0.000 | no (identical) |
| HV | -0.34 | -0.32 | 2.9e-6 | +0.044 | no |
| NDVI | -0.51 | -0.51 | 3.6e-11 | +0.066 | no (same GEE pull) |
| RVI+NDVI spatial PR | 0.132 | 0.132 | - | - | no |

Unsmoothed backscatter does not rescue RVI - the ratio-cancellation mechanism
is independent of speckle filtering/multilooking. All verdicts stand.

### Results (`.../RVI_NDVI/Results/plots`)

- `fig_rvi_kde.png`, `fig_rvi_roc.png` - RVI/RFDI overlap + diagonal ROC (null).
- `fig_rvi_qp_kde.png`, `fig_rvi_qp_box.png` - eigenvalue RVI null; span shift.
- `fig_ndvi_kde.png` - Unhealthy left-shifted NDVI, heavy low tail.
- `fig_rvi_ndvi_scatter.png` - separation is vertical (NDVI); RVI axis is noise.
- `fig_fusion_roc.png` - NDVI curves dominate; RVI curves hug diagonal.

### ML probe (LogReg / RF / XGB / MLP, spectral-only, no coords)

No model beats univariate NDVI by more than noise. Nonlinear models do worse than linear everywhere (160 positives cannot support interactions); MLP collapses (0.048 on NDVI-only, below no-skill).

| Features (spatial-block PR) | LogReg | RF | XGB | MLP |
|---|---:|---:|---:|---:|
| NDVI only | 0.133 | 0.080 | 0.094 | 0.048 |
| NDVI+B4/B8 | 0.132 | 0.122 | 0.108 | 0.085 |
| +SAR levels | 0.133 | 0.124 | 0.109 | 0.079 |
| +RVI | 0.140 | 0.122 | 0.096 | 0.064 |
| SAR only | 0.140 | 0.095 | 0.072 | 0.068 |

Triage: top-25% review finds ~45% of Unhealthy (~1.8x enrichment).
`fig_ml_spatial_pr.png`, `fig_ml_recall_budget.png`.

### Richer spectra + temporal NDVI

Red-edge/SWIR (same window): NDRE5 d=-0.41, NDRE6 d=-0.36, NDMI d=-0.30 (all p<1e-3, same direction as NDVI); EVI weak. Fusion NDVI+rededge/SWIR: spatial PR **0.145** / ROC **0.680** vs 0.133/0.653 NDVI-only, both protocols agree - first genuine lift from adding features (modest, +0.012).

Temporal NDVI (+/-90 d of SAR): slope/delta/min/std all null
(|d|<=0.20, p>=0.01); +temporal fusion 0.131 = nothing. Oil palm is stable; single-composite NDVI already captures what is there.
`fig_spectral_kde.png`, `fig_spectral_d.png`.

### Block-size sweep

| PR-AUC | random | K100 (~25 trees, ~38 m) | K50 | K15 | K5 (~500 trees, ~212 m) |
|---|---:|---:|---:|---:|---:|
| NDVI only | 0.142 | 0.144 | 0.142 | 0.133 | 0.129 |
| +rededge/SWIR | 0.152 | 0.157 | 0.157 | 0.145 | 0.123 |
| +RVI | 0.164 | 0.165 | 0.175 | 0.145 | 0.125 |

NDVI base is flat across the whole leakage spectrum; every additive gain shrinks with block size and vanishes at ~200 m. Measured train-test separation explains why: median nearest-train-tree 8.5 m (random) / 24 m (K50) / 52 m (K15) / 113 m (K5) - at K50, 91% of test trees sit within 50 m of a train tree, inside the disease-correlation radius.
`fig_block_gradient.png`.

### Corner deployment (product decision)

Disease anatomy: 160 Unhealthy = 33 pockets of 2-10 trees + 35 isolated singles (30 m rule); median nearest-sick-neighbour 17 m. Recall by distance to nearest train sick tree: 0.55 (<25 m) -> 0.36 (25-100 m). Pocket-holdout (never-seen pockets): recall **0.26** vs 0.42 random-split - and 0.25 is coin-flip under a top-25% rule. On unseen pockets the model is chance; near labelled pockets it works.

Deployment test (train one estate quadrant, predict rest):
NDVI-only corner-train mean PR 0.127 vs 0.131 size-matched random-train - the spatial gap costs ~nothing. Corner lottery applies: SW (63 pos) -> 0.083 worst, NW (33 pos) -> 0.148 best; representativeness beats quantity. At corner sizes NDVI-only beats +rededge (extra features overfit). SAR-only corner: 0.06-0.09 (~no-skill).
`fig_pockets.png`, `fig_corner.png`.

## Limitations

- S2 10 m vs SAR 6.4 m vs ~9 m crown spacing: per-pixel NDVI mixes neighbours;
  the reported effect is thus conservative (mixing dilutes, not creates, signal)
- Single-date S2 composite; multi-temporal NDVI change may be stronger still
## Recommended follow-ups (ranked)

1. **S2 pull for Palong/Serting** - optical-transfer evidence (labels are prediction-derived; treat as supporting, not validation).
2. Closed, do not rerun: RVI variants, temporal NDVI (+/-90 d), DL at this sample size, threshold tuning.

## Artifacts (`.../RVI_NDVI/*`)

- `poc_rvi_ndvi_stage0_resample.py` - 3x3 resampling from (re-exported) scene
- `poc_rvi_ndvi_stage1.py` - intensity RVI stats
- `poc_rvi_ndvi_stage1b_quadpol.py` - T3 sampling + eigenvalue RVI
- `poc_rvi_ndvi_stage2_gee.py` - GEE pull (needs SA + role above)
- `poc_rvi_ndvi_stage2b_fusion.py` - offline NDVI/RVI fusion stats
- `poc_rvi_ndvi_stage3_ml.py` - classical ML probe (spectral-only)
- `poc_rvi_ndvi_stage4_spectrotemporal.py` - GEE red-edge/SWIR + temporal pull and test
- `poc_rvi_ndvi_stage5_blocksize.py` - leakage-gradient sweep
- `poc_rvi_ndvi_stage5b_distances.py` - train-test separation diagnostic
- `poc_rvi_ndvi_stage6_pockets.py` - pocket anatomy + holdout tests
- `poc_rvi_ndvi_stage7_corner.py` - corner-deployment simulation
- Artifacts: `.../RVI_NDVI/Results` (CSVs, JSONs, 15 figures)



