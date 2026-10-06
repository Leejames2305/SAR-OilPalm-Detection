# POC: Limited-labels triage - train small corner, predict rest (AirHitam)

**Date:** 6 October 2026

**Scope:** AirHitam only (2511 trees: 2351 Healthy / 160 Unhealthy @ 6.37%) -
  sole estate with one-to-one ground truth. Follows `RVI_NDVI.md` stages 0-7;
  no RVI re-test, no new GEE pull, no coordinates as features.

**Question:** Owner labels one contiguous area (B = 1000-1600 trees), model
  triages the rest. Which labelling walk, which budget, which model?

## Verdict

1. **NDVI-LogReg triages at ~2x enrichment, flat across everything.**
   PR ~0.124-0.152 vs 0.064 no-skill, ROC ~0.64-0.67, recall@25% ~0.43-0.47
   (~1.8x). Same band under every design, every budget, and both ceilings.
2. **Labelling design barely matters for ranking; it matters for variance.**
   Corner-block vs strip vs dispersed vs stratified vs random all land within
   ~0.01 PR at fixed B. Corner/strip SD is 10-40x dispersed SD (corner
   lottery); dispersed covers pockets more evenly but converts to only +0.01
   PR at B=1600, nothing at B=1000.
3. **Budget 1000 -> 1600 buys nothing.**
   E_random PR 0.133 -> 0.130; A_block 0.129 -> 0.124. Oracle 80% train
   (2008 trees) PR 0.123; spatial-K15 OOF PR 0.127. The ceiling is the floor:
   NDVI-LogReg saturates by ~1000 labels; extra walking does not lift it.
4. **PCA hurts; QDA adds nothing.**
   PCA2/3+LogReg drops PR by ~0.02 everywhere (first PCs keep healthy-canopy
   variance, not the disease direction). QDA-1D == LogReg-1D exactly; QDA-2D
   trails LogReg-spec slightly. At ~64-103 train positives there is nothing
   quadratic to estimate beyond the linear shift.
5. **+rededge/SWIR helps only slightly, only at B=1600 off-corner**
   (+0.009-0.011 PR on random masks; zero or negative on B=1000 corners).
   Deployment baseline stays NDVI-only LogReg; revisit spectra only if the
   labelling strategy changes first.
6. **Pocket coverage != ranking power.**
   Stratified/random hit 79-96% of DBSCAN pockets; corners hit 40-66%. Yet
   recall@25% differs by ~0.02. NDVI carries a global level shift, not a
   pocket fingerprint - consistent with pocket-holdout recall 0.26 (chance)
   in `RVI_NDVI.md`.

## Data & Method (frozen)

- Frame `tables/frame_frozen.csv` (2511 x 25): join of reused
  `airhitam_rvi_features.csv` + `airhitam_ndvi.csv` + `airhitam_rvi_quadpol`
  + `airhitam_spectral` + labels. Join loss 0, NDVI missing 0.
- Estate ~1026 x 492 m (~50 ha), spacing ~9 m; quadrants SW 586/63,
  NW 671/33, SE 670/40, NE 584/24 (median split).
- P1: B in {1000,1200,1400,1600}; masks A corner-block (B-nearest to 4
  extreme corners) | B strips (contiguous windows on lon/lat x5 positions
  each) | C dispersed (KMeans-5 mini-blocks x10 seeds) | D NDVI-stratified
  x30 | E random x30. Model fixed: balanced LogReg, NDVI-only.
- P2: oracle 80/20 x10 + spatial-K15 group-5-fold OOF x3 + quadrant-train
  continuity. P3: 6 configs (logreg_NDVI | logreg_spec | pca2/3_logreg |
  qda_ndvi | qda_2d) on A_block (4 corners) + E_random (12 reps) at
  B=1000/1600.
- Metrics on held-out rest: PR-AUC (primary), ROC, recall@10/25%,
  enrichment, pos_train, pocket/single hit (DBSCAN eps=30 m: 33 pockets,
  35 singles - reproduced).

## Results

### P1 - design x budget (NDVI-only LogReg, mean over placements)

| B | A_block PR / r25 | B_strip PR / r25 | C_disp PR / r25 | D_strat PR / r25 | E_rand PR / r25 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.129 / 0.450 | 0.131 / 0.449 | 0.134 / 0.444 | 0.133 / 0.466 | 0.133 / 0.456 |
| 1200 | 0.126 / 0.458 | 0.131 / 0.440 | 0.140 / 0.469 | 0.139 / 0.471 | 0.137 / 0.459 |
| 1400 | 0.128 / 0.449 | 0.129 / 0.444 | 0.148 / 0.466 | 0.133 / 0.451 | 0.137 / 0.450 |
| 1600 | 0.124 / 0.450 | 0.132 / 0.434 | 0.152 / 0.469 | 0.138 / 0.470 | 0.130 / 0.458 |

Pocket-hit at B=1000: A 0.40 / B 0.43 / C 0.48 / D 0.79 / E 0.78.
pos_train lottery (corners): B=1000 min 39 (NE-side) vs mean 61.

### P2 - ceilings

Oracle 80/20 mean PR 0.123 (rep range 0.089-0.165, small-test variance);
spatial-K15 OOF 0.127; quadrants 0.083 / 0.148 / 0.137 / 0.138 (exact
stage-7 replication). Nothing above the P1 band.

### P3 - model check (spatial PR, mean)

| B / mask | LR-NDVI | LR-spec | PCA2+LR | PCA3+LR | QDA-1D | QDA-2D |
|---|---|---:|---:|---:|---:|---:|
| 1000 A_block | 0.129 | 0.125 | 0.114 | 0.105 | 0.129 | 0.123 |
| 1000 E_rand | 0.133 | 0.136 | 0.114 | 0.106 | 0.133 | 0.123 |
| 1600 A_block | 0.124 | 0.135 | 0.102 | 0.089 | 0.124 | 0.110 |
| 1600 E_rand | 0.131 | 0.140 | 0.117 | 0.112 | 0.131 | 0.124 |

### Operating point (example, B=1200)

Rest ~1311 trees (~84 sick). Review top-25% (~328 trees) -> ~40 sick
found (~47% recall, ~1.9x enrichment). Same arithmetic holds +/-2% for
every design and B.

### Figures (`misc/POC_Results/LimitedLabelsTriage/figs/`)

- `fig_p0_qa.png` - estate map + quadrants + NDVI hist.
- `fig_p1_designs.png` - PR / recall@25% vs B by design (flat bands).
- `fig_p1_maps.png` - one train mask per design at B=1200.
- `fig_p2_curve.png` - learning curve vs oracle/OOF ceilings.
- `fig_p3_model.png` - PCA-down / QDA-flat / spec-tiny-gain bars.

## Limitations

- Single-estate evidence; corner lottery (SW 0.083 vs NW 0.148) persists.
- AirHitam B=1000 leaves ~60% unvisited, not >90% - real estates are larger;
  quadrants (~75% left) are the closest proxy; curve extrapolates the rest.
- S2 10 m mixes crowns (conservative bias); Palong/Serting excluded by
  label quality.


## Artifacts

- `misc/analysis/poc_limitedlabels_p0_audit.py` - freeze + QA
- `misc/analysis/poc_limitedlabels_p1_designs.py` - 336 train/rest runs
- `misc/analysis/poc_limitedlabels_p2_curve.py` - ceilings + curve
- `misc/analysis/poc_limitedlabels_p3_model.py` - PCA/QDA/spec check
- Tables/figs: `misc/POC_Results/LimitedLabelsTriage/{tables,figs}/`
