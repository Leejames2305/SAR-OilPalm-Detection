# POC: RVI (ALOS quad-pol) vs NDVI (Sentinel-2) for Healthy/Unhealthy separation - AirHitam

**Date:** 5 October 2026


**Scope:** AirHitam only (2511 trees: 2351 Healthy / 160 Unhealthy @ 6.37% positive) -
  the sole estate with one-to-one field ground truth.


**Question:** Can RVI separate the classes statistically? If not, does adding
  Sentinel-2 NDVI help?

## Verdict

1. **RVI barely separate Healthy from Unhealthy.**
   Four dual-pol-style ratios and the true eigenvalue quad-pol RVI are all
   coin flips (|d| <= 0.04, MWU p 0.61-0.99, ROC-AUC ~0.50, PR lift ~0).
2. **NDVI can - modestly but robustly.** d = -0.51 (medium), p = 3.6e-11,
   PR-AUC 0.130 vs 0.064 no-skill (~2x enrichment), ROC-AUC 0.65, and the
   signal survives honest spatial-block CV (0.133 / 0.653).
3. **Fusion adds nothing over NDVI alone.** RVI+NDVI == NDVI (spatial PR 0.132
   vs 0.133); +HV is within noise (0.135). RVI contributes zero ranking power.
4. **Mechanism:** Unhealthy trees are dimmer in all SAR channels
   proportionally (HV d=-0.34, span d=-0.27), so every ratio normalisation
   erases the level shift. Optical NDVI measures a different physical quantity
   (chlorophyll/red-edge absorption) that ratios do not cancel.

This is the first positive evidence for backlog item 4 (optical complement)
in `docs/AgentPlan/MainChecklist.md`, and it closes the "RVI rescue" branch:
no further RVI variants are warranted.

## Data & method

- **SAR:** 
  - `ALOS2-Subset_AirHitam_260610_Cal_ML_Spk_TC.tif` ([HH, HV, VH, VV], EPSG:32648, 6.43 m; Scene date 2026-06-10)
  
  - `ALOS2-Subset_AirHitam_260610_Cal_mat_Spk_TC.ti` - T3 coherence
( T11, T12re/im, T13re/im, T22, T23re/im, T33, EPSG:4326)
  - Per-tree 3x3 means
  - RVI variants:

    - RVI_classic: `8HV/(HH+VV+2HV)`
    - RVI_xavg: Cross-pol averaged)

    - RVI_hh: `4HV/(HH+HV)`, RVI_vv: `4VH/(VV+VH)`
    - `RFDI`
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

### Results (`misc/POC_Results/RVI_NDVI/`)

- `fig_rvi_kde.png`, `fig_rvi_roc.png` - RVI/RFDI overlap + diagonal ROC (null).
- `fig_rvi_qp_kde.png`, `fig_rvi_qp_box.png` - eigenvalue RVI null; span shift.
- `fig_ndvi_kde.png` - Unhealthy left-shifted NDVI, heavy low tail.
- `fig_rvi_ndvi_scatter.png` - separation is vertical (NDVI); RVI axis is noise.
- `fig_fusion_roc.png` - NDVI curves dominate; RVI curves hug diagonal.

## Limitations

- S2 10 m vs SAR 6.4 m vs ~9 m crown spacing: per-pixel NDVI mixes neighbours;
  the reported effect is thus conservative (mixing dilutes, not creates, signal)
- Single-date S2 composite; multi-temporal NDVI change may be stronger still
## Recommended follow-ups (ranked)

1. **Temporal S2:** time-series / change NDVI around census dates 
2. **SAR + NDVI spatial-fusion triage test** under the broad-region protocol -
   does NDVI raise the deployable operating point or just the univariate?

## Artifacts

- `misc/analysis/poc_rvi_ndvi_stage1.py` - intensity RVI stats
- `misc/analysis/poc_rvi_ndvi_stage1b_quadpol.py` - T3 sampling + eigenvalue RVI
- `misc/analysis/poc_rvi_ndvi_stage2_gee.py` - GEE pull (needs SA + role above)
- `misc/analysis/poc_rvi_ndvi_stage2b_fusion.py` - offline NDVI/RVI fusion stats
- Artifacts: `misc/POC_Results/RVI_NDVI/` (CSVs, JSONs, 8 figures)
