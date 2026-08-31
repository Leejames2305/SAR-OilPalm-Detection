# Mean-Sampling POC — Can 'Unhealthy' Trees Be Seen Statistically in SAR Linear-Power Backscatter?

**Run:** standalone notebook `misc/notebook/POC_MeanSampling.ipynb` on Colab. **Artifacts generated:** `misc/POC_Results/MeanSampling/*` 

**Scale:** the scene was re-uploaded to GCS with **linear (power) bands only**; the notebook and
every statistic in this report operate on **linear power** (no dB). The linear-power rerun
reproduced the earlier run's numbers exactly — see §5 — so all conclusions below are unchanged.

## TL;DR

1. **Yes — but only very weakly, and it does not generalise.** A small, statistically
   detectable mean-backscatter shift exists between Healthy and Unhealthy, but every effect
   size is small (`|Cohen's d| <= 0.36`, most in `0.1–0.3`). The two class distributions overlap
   heavily at per-tree resolution.
2. **The direction of the effect flips between estates.** At Palong, Unhealthy trees sit
   *lower* than Healthy in every band (VH `d = -0.35` is the strongest). At Serting the same
   bands point *positive* (VV `d = +0.27`, HV `d = +0.24`). A rule fit on one estate is therefore
   reversed at the other — no transferable mean-intensity cue.
3. **The "disease spreads regionally" hypothesis is NOT supported by mean sampling.**
   Widening the window 3 -> 21 px does not widen the class gap; it dilutes it. The strongest
   separation is at the smallest window and decays monotonically as the window grows.
4. **Significance does not equal separability.** With ~1500–2000 Healthy vs ~150–200
   Unhealthy per estate, Mann-Whitney p-values go significant at tiny effect sizes. Per-tree
   samples carry too little overlap-free separation to support a per-tree ML classifier
   (consistent with the prior audits).


## 1) What this POC tests

The main notebook's per-tree ML model reaches `PR-AUC ~0.15` under *honest* spatial-block CV
(no-skill baseline ~0.09) and Recall(Unhealthy) ~ 0 on held-out spatial blocks
(`docs/POCReport/FeatureTests.md`, `PipelineAudit.md`). The leading hypothesis is
that disease spreads regionally — a patch of trees turns Unhealthy together, so a single tree's
isolated label is noisy.

This POC therefore asks two simple, non-ML questions against the **linear-power backscatter
GeoTIFF** (the scene was re-uploaded to GCS with **linear (power) bands only** — no dB bands):

- If we sample mean backscatter at every tree coordinate with a **3x3** and **5x5** mean window,
  do Healthy and Unhealthy separate (central tendency, spread, distribution)?
- As we widen the window (3 -> 5 -> 7 -> ... -> 21 px), does the class gap reveal or widen
  (the "regional spread" hypothesis)?

### Design (mirrors main.ipynb)
- **Scene:** `ALOS2-HBQR1_1__D-ORBIT__ALOS2650583560-260610_Cal_ML_Spk_TC` from
  `gs://sar-oilpalm/data/SAR-scenes/` — **linear power scale** (`band_scale` is recorded in
  `poc_summary.json`).
- **Labels:** `<estate> Classification.csv` (`id / Long / Lat / Class`), reprojected EPSG:4326 ->
  raster CRS.
- **Bands:** HH, HV, VV, VH. Windows: 3, 5, 7, 9, 11, 15, 21 px (odd, centred on the tree;
  nodata / non-finite / `<= 0` samples masked — linear power is strictly positive, so the old
  dB `<= -99` rule no longer applies). All statistics are computed **directly on the linear
  power values**; log axes in the plots are display-only.
- **Classes:** Healthy vs Unhealthy. Middle is kept in the wide CSV but excluded from the tests.
- **Statistics per (estate, band, window):** n, mean, std, median; **Cohen's d** (pooled SD,
  var `ddof=1`); **Mann-Whitney U** p (asymptotic); `frac_unhealthy_below_healthy_median`
  (non-parametric overlap index).

### Counts
```
estate    Healthy  Unhealthy  Middle  total
Palong    1971     200        493     2664
Serting   1513     154        378     2045
pooled    3484     354        871     4709
```
The Healthy vs Unhealthy contrast uses **3838** trees (3484 + 354), ratio **9.8:1**.

## 2) Headline results (3x3 and 5x5 windows)

`stats_h_vs_u.csv`. `d = (mean_Unhealthy - mean_Healthy) / pooled SD`. Near 0 = heavy overlap.

| estate | band | win | Cohen's d | MWU p  | direction(Unhealthy) |
|--------|------|-----|-----------|--------|----------------------|
| Palong | HH   | 3   | -0.256    | 0.0145 | lower |
| Palong | HV   | 3   | -0.243    | 0.1615 | lower (ns) |
| Palong | **VH** | 3 | **-0.353**| 0.0005 | **lower** |
| Palong | **VH** | 5 | **-0.356**| 0.0035 | **lower** |
| Palong | VV   | 3   | -0.255    | 0.0345 | lower |
| Serting| **HV** | 5 | **+0.243**| 0.0015 | **higher** |
| Serting | **VV** | 5| **+0.274**| 0.0012 | **higher** |

Pooled across estates (`poc_summary.json`), only HH and VH clear `p<0.05` at the headline
windows — and only because the larger **Palong** estate dominates the pooled sample.


## 3) Key observations

**a) Every effect is small.** All `|d|` lie between 0.03 and 0.36. Only one band/estate/window
pair (Palong VH, ~0.35) even approaches small-moderate; the rest are clearly small (0.1–0.3).
The two class distributions overlap massively.

**b) Same sign within Palong, sign flip at Serting.** Palong is internally consistent: all four
bands point *lower* for Unhealthy. Serting's VV / HV point *higher*:
```
window 3:  HH -0.05  HV +0.18  VH +0.07  VV +0.27   (3 of 4 opposite to Palong)
window 5:  HH -0.03  HV +0.24  VH +0.03  VV +0.27   (HV, VV opposite to Palong)
```
A "Unhealthy = darker" rule that holds at Palong is reversed at Serting for the same bands.
A single mean-intensity cue cannot generalise across estates.

**c) The "regional spread" hypothesis is not supported.** `window_trend.csv`:
```
Palong  mean|d|:  3px=0.277  5px=0.272  7px=0.257  9px=0.227  11=0.188  15=0.114  21=0.090
Serting mean|d|: 3px=0.141  5px=0.145  7px=0.134  9px=0.133   11=0.129  15=0.100  21=0.101
```
Enlarging the window never increases the class-mean gap; it averages the local cue away.
A genuinely regional cue would grow (or at least hold) with window size — it does not.

**d) Large-`n` p-values are misleading.** With ~1500–2000 Healthy vs ~150–200 Unhealthy per
estate, Mann-Whitney `p<0.05` appears even at `|d|` ~ 0.1–0.2 (e.g. Palong HH at 21 px has
`d = +0.03` yet `p = 0.022`). The practical overlap (`frac_unhealthy_below_healthy_median`
~0.40–0.57, near 0.5) confirms the classes are not separable at per-tree scale.


## 4) What this means for the main ML pipeline

1. **A weak, estate-specific statistical cue exists but does not transfer.** The only coherent
   statement is "at Palong, Unhealthy = slightly lower mean intensity, strongest in VH." At
   Serting, the relationship is partially reversed. A single global rule (or an estate-agnostic
   "Unhealthy = darker" prior) is not supported.

2. **Per-tree classification cannot be rescued here.** The absolute mean differences are tiny
   relative to within-class spread; only a small fraction of Unhealthy ever crosses the Healthy
   median. This is exactly why honest spatial-block / cross-estate models collapse to ~no-skill:
   the intrinsic signal at per-tree granularity is too weak. It is a data property, not a bug.

3. **Bigger mean windows are not the answer.** The data says disease is not resolved by
   whole-region means; widening the window smooths the weak local signal. A plot/region-labeling
   reframe built on "means over larger neighbourhoods" is therefore unlikely to help.

4. **Better next probes** (consistent with `PipelineAudit.md`):
   - **multi-temporal / change** SAR (disease is a change response, not an absolute level);
   - **detrended local-anomaly** features (tree value minus its background ring), rather than
     absolute means;
   - **per-estate models** / a second scene, always reported under spatial-block honest CV.


## 5) Sanity & reproducibility notes

- **Scale is confirmed linear power.** The scene was re-uploaded to GCS with linear (power) bands
  only and the notebook overhauled to sample and analyse linear power (nodata = non-finite or
  `<= 0`; window statistics are means of linear power). All sampled values are strictly positive
  (~0.008–1.1, per-band medians ~0.02–0.16), consistent with a power scale.
- **The linear rerun reproduces the previous run exactly.** Every number in this report (headline
  d / p, pooled effects, window trend, per-estate significance counts) is identical to the
  pre-rerun results, to the last stored digit. The earlier "near-linear, not dB" suspicion about
  the sampled values was therefore correct — the previous scene was already effectively on the
  linear scale, so no conclusion changes.
- All 16 rows of `stats_h_vs_u.csv` were re-derived independently from `backscatter_means.csv`
  (pooled SD with `ddof=1`, MW-U asymptotic) — 0 mismatches (re-verified on the linear rerun).
- 0% missing samples per band/window (every tree mapped inside the raster).
- Plots: boxplot y-axes use a **log scale** and the KDE panels are drawn on `log10(power)` —
  display only, for readability of the right-skewed linear distributions. No analysis number is
  derived from a log-transformed value.
- Per-estate significance counts: Palong 5/28 and Serting 11/28 band-by-window pairs reach
  `p<0.05`, but all with small |d|.


## 6) Recommendation

- Keep per-estate, per-spatial-block honest evaluation as the protocol. Do **not** invest further
in a mean-sampled, intensity-only per-tree classifier for Unhealthy: the sample-level cue is
weak, opposite-signed across estates, and diluted by bigger windows. The highest-value next
step is a multi-temporal change or detrended local-anomaly probe, each analysed separately per
estate, under the existing spatial-block honest-CV protocol.

## 7) Addendum — RVI/RFDI extension & the Middle-class investigation

Follow-up probes after the main POC, run on the linear-power rerun artifacts. Analysis scripts:
`misc/analysis/middle_class_analysis.py`, `misc/analysis/middle_predictive.py` (repo `.venv`).

### 7a) RVI / RFDI (window means, W3 & W5, per-estate evaluation)

- **RVI `8·HV/(HH+VV+2HV)` and RFDI `(HH−HV)/(HH+HV)` cannot indicate Unhealthy trees** at
  per-tree granularity: AUC 0.45–0.55, class shift ≤ 0.19 SD, spatial-block-CV PR-AUC ≈ baseline.
- Root cause: the two indices are nearly redundant (corr ≈ −0.99) — both collapse to the HH/HV
  polarisation ratio, cancelling the absolute-brightness information where the (weak) signal lives.
- Family-wise permutation test (full 112-candidate sweep re-run per permutation) says the small
  cross-pol signal that does exist is real but minuscule: best AUC ≈ 0.58 vs chance, p ≈ 0.008.

### 7b) What the "Middle" class actually is (and why it can't help classification)

Three hypotheses tested (intermediate health state / label noise / independent class), per estate:

- **Spectrally inconsistent**: on raw bands Palong shows a genuine graded ordering
  H > M > U (Jonckheere–Terpstra p down to 4.8e-6 on VH W3), but Serting does not replicate it
  (orderings absent or inverted). Mixture fits put Middle mostly Healthy-like at Serting and
  ~50/50 H/U at Palong — no consistent spectral identity.
- **Spatially unambiguous — Middle is the disease front** (replicates at both estates):
  median distance Middle→nearest Unhealthy is **14.6 m** vs 26.5 m (Palong) / 20.7 m (Serting)
  for Healthy trees; Middle trees have ~2.5× more Unhealthy neighbours within 30 m.
- **Exploiting it does not improve detection** (all under honest spatial-block CV, scored on H/U):
  folding M into the target (U+M vs H) dilutes the signal (median |AUC−0.5| gain −0.017);
  augmenting training with M as weak positives (w = 0.5/1.0) consistently lowers ROC
  (Palong HGB 0.606 → 0.562–0.590; Serting 0.500 → 0.462–0.492).

- **Interpretation:** Middle is a *spatially*-defined label — annotators ring diseased patches with
it — so its ground truth encodes neighbourhood structure that a single-date 6.4 m SAR snapshot
does not contain. This independently confirms why spatial-block CV collapses every per-tree model,
and reinforces the Section-6 recommendation: the remaining live direction is **multi-temporal
change** (for which the clustered-patches-plus-transition-rim label structure is well suited).
Operationally, the Middle ring remains valuable as the spread-risk perimeter once a diseased
patch is identified.
---