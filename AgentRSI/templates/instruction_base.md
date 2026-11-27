# AgentRSI Session: {{SESSION_ID}}

## Objective

Build a reproducible, self-contained **oil-palm disease triage model** distinguishing `Unhealthy` (positive) from `Healthy` (negative) trees across three Malaysian estates (`AirHitam`, `Palong`, and `Serting`, Negeri Sembilan; ~9 m palm spacing).

This model serves as an aerial/satellite screening layer for a downstream, out-of-scope on-site polarized-imagery inspection system. Raw classification accuracy is not the target. Under severe class imbalance (~6–10% positive rate), the operational goal is to rank the majority of `Unhealthy` trees into a practical top-fraction inspection budget while keeping false-positive inspection overhead reasonable. In deployment, a model trained on a spatially sampled labelled slice (`L`) must generalize across the remaining spatially disjoint unlabelled blocks of each estate.

All three estates' labels were collected under the same screening methodology and are verified and trusted equally. Older reports under `/app/prior/POCReport/` (`AirHitamRF.md`, `RVI_NDVI.md`, `LimitedLabelsTriage.md`, `TimeSeries.md`) contain an outdated assumption describing `Palong` and `Serting` labels as derived or less trustworthy; their empirical metrics remain valid, but treat all three estates with equal weight and trust.

## Data Specification

### Tabular Inputs (`/app/data/`)

- `/app/data/L_labelled.csv` — 2,747 rows × 22 columns, 0 NaNs. Per-estate class counts: `AirHitam` (908 `Healthy`, 56 `Unhealthy`, 0 `Middle`), `Palong` (704 `Healthy`, 66 `Unhealthy`, 198 `Middle`), `Serting` (597 `Healthy`, 67 `Unhealthy`, 151 `Middle`).
- `/app/data/unlabelled.csv` — 4,473 rows × 21 columns, 0 NaNs (`AirHitam`: 1,547; `Palong`: 1,696; `Serting`: 1,230). Identical feature columns and no `label` column. Trees come from 80 m square spatial blocks completely disjoint from `/app/data/L_labelled.csv`. Unmarked `Middle` trees are present in `Palong` and `Serting` and are excluded before metric scoring.

| Columns | Units / Scale | Provenance & Sampling Convention |
|---|---|---|
| `estate`, `Long`, `Lat` | string, WGS 84 (`EPSG:4326`) deg | Estate name and tree trunk coordinates |
| `HH_<date>`, `HV_<date>`, `VH_<date>`, `VV_<date>` | linear backscatter power ($\sigma^0$) | ALOS PALSAR-2 L-band (`260610` = 2026-06-10, `261014` = 2026-10-14); 3×3 pixel window mean from `*_Cal_Spk_TC.tif`, masking non-finite and `<= 0` pixels ($\ge 5$ valid pixels required) |
| `HAlpha_H_<date>`, `HAlpha_A_<date>`, `HAlpha_alpha_<date>` | $H \in [0, 1]$, $A \in [0, 1]$, $\alpha$ in degrees | Cloude–Pottier decomposition (`260610`, `261014`); 3×3 window mean from `*_Cal_mat_HAlpha_Spk_TC.tif`, same masking rule |
| `B4`, `B8`, `NDVI`, `clear_count` | raw SR DN (0–10000), `[-1, 1]`, integer count | Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` median composite (`2026-05-15` to `2026-07-15`, SCL classes 4–7 kept, sampled at 10 m via `reduceRegions(mean)`); `NDVI = (B8 - B4) / (B8 + B4)` |
| `label` (`/app/data/L_labelled.csv` only) | `Healthy`, `Unhealthy`, `Middle` | Verified ground-truth class; `Middle` is visible for optional auxiliary/ordinal/spatial exploration but never scored |

### Raw ALOS-2 GeoTIFF Rasters (`/app/rasters/`)

All 19 ROI-masked GeoTIFFs carry **no embedded band descriptions and `nodata=None`**. The positional 1-based band orders, CRSs, and scales below are authoritative; always reproject `(Long, Lat)` from `EPSG:4326` to each raster's native `src.crs` before pixel indexing.

| Filename pattern | Coverage | CRS & resolution | Bands (1-based order) | Scale & masking |
|---|---|---|---|---|
| `ALOS2-Subset_<estate>_<date>_Cal_Spk_TC.tif` | All 3 estates × `260610`, `261014` | `EPSG:32648`, 6.43 m | 1: `HH`, 2: `HV`, 3: `VH`, 4: `VV` | Linear power; non-finite or `<= 0` is nodata |
| `ALOS2-Subset_<estate>_<date>_Cal_mat_HAlpha_Spk_TC.tif` | All 3 estates × `260610`, `261014` | `EPSG:32648`, 6.43 m | 1: `H`, 2: `A`, 3: `alpha` (deg) | Linear; non-finite or `<= 0` is nodata |
| `ALOS2-Subset_<estate>_<date>_Cal_mat_Yamaguchi_Spk_TC.tif` | All 3 estates × `260610`, `261014` | `EPSG:32648`, 6.43 m | 1: `Pd` (double-bounce), 2: `Pv` (volume), 3: `Ps` (surface), 4: `Pc` (helix) | **dB** (SNAP export default; negative values — do not mask `<= 0`; cross-mask ROI support from `Cal_Spk_TC` and check finite values) |
| `ALOS2-Subset_AirHitam_260610_Cal_mat_Spk_TC.tif` | `AirHitam`, `260610` only | `EPSG:4326`, $4.61 \times 10^{-5}$ deg | 1..9: `T11`, `T12_re`, `T12_im`, `T13_re`, `T13_im`, `T22`, `T23_re`, `T23_im`, `T33` | Linear T3 coherency matrix; off-diagonals are signed — mask using `T11 <= 0` or non-finite |

### Prior Knowledge & Workspace Tools

- `/app/prior/MainChecklist.md` — concise index of settled facts (§A), closed dead ends (§B — do not re-run without a new mechanism), open leads (§C), and known pitfalls (§E, including cross-estate Cohen's $d$ sign flips). Read this before planning experiments.
- `/app/prior/POCReport/` — detailed prior POC reports (start with `/app/prior/POCReport/PathForwardTrials.md`, `/app/prior/POCReport/RVI_NDVI.md`, `/app/prior/POCReport/LimitedLabelsTriage.md`, `/app/prior/POCReport/TimeSeries.md`).
- `/app/prior/Plan-GEECatalogSurvey.md` — surveyed Google Earth Engine collections and sensor caveats (e.g., Sentinel-2 baseline 04.00 `+1000` harmonisation offset).
- `/app/incumbent/` — current best verified `pipeline.py` and `requirements.txt`.
- `/app/prior/prev_HANDOFF.md` — previous session handoff (present when continuing from a prior session).
- `/app/prior/prev_REPORT.md` — previous session technical report (present when continuing from a prior session).
- `/app/prior/prev_submission/` — previous session's submitted `pipeline.py`, helper modules, and `requirements.txt` (present when continuing from a prior session, even if it did not replace `/app/incumbent/`).
- `/app/prior/ref_sessions/<session_id>/` — optional reference sessions' `HANDOFF.md`, `REPORT.md`, and `submission/` staged for cross-session comparison or synthesis (present when built with `--ref`).
- `agentrsi` CLI:
  - `agentrsi cv [--submission /app/submission]` — 5-fold 80 m spatial-block `GroupKFold` CV on `/app/data/L_labelled.csv` against the NDVI baseline.
  - `agentrsi selfcheck [--submission /app/submission]` — validates `/app/submission/requirements.txt` pins, audits non-base third-party imports and symlinks, and runs `fit_predict()` on all three estates.
  - `agentrsi gcp-init` — verifies Earth Engine access using `$GOOGLE_APPLICATION_CREDENTIALS` (`/run/secrets/gcp-sa.json`) and `$GCP_PROJECT`.
{{MCP_TOOLS_SECTION}}
## Evaluation Metrics & Validation Discipline

Primary objective is **composite PR-AUC lift**: the unweighted mean across `AirHitam`, `Palong`, and `Serting` of `AP(y, s) - prevalence`, evaluated on spatially held-out 80 m blocks where `y` is 1 for `Unhealthy` and 0 for `Healthy` (`Middle` rows excluded prior to scoring), `prevalence = mean(y)`, and `AP(y, s)` is tie-aware step-function average precision matching `sklearn.metrics.average_precision_score` (constant scores yield `0.0` lift).

Secondary triage diagnostics tracked alongside PR-AUC lift are tie-averaged `Recall@top-25%`, `Enrichment@top-10%` (`Recall@top-10% / 0.10`), and `ROC-AUC`.

A viable improvement must satisfy four scientific criteria on held-out 80 m spatial blocks:
1. Deliver a statistically consistent composite PR-AUC lift gain over `/app/incumbent/pipeline.py` across spatial blocks.
2. Improve at least two of the three estates (`AirHitam`, `Palong`, `Serting`) over `/app/incumbent/pipeline.py` — single-estate wins that leave the other two estates flat or worse are insufficient.
3. Avoid materially degrading any individual estate (`AirHitam`, `Palong`, or `Serting`).
4. Disentangle true disease signal from pure spatial smoothness — predictions are benchmarked against both `/app/incumbent/pipeline.py` and a coordinates-only spatial prior (distance-weighted 15-NN on standardized `Long`, `Lat`).

**Spatial-validation warning:** `L_labelled.csv` and `unlabelled.csv` are split on 80 m square blocks (`floor(x/80) * 100000 + floor(y/80)` in local metres). Random K-fold splits and same-block leave-one-out spatial neighbors severely overestimate generalization because adjacent palms are ~9 m apart, whereas held-out blocks sit across 80 m boundaries. Similarly, wide-footprint neighborhood or patch features (e.g., image crops spanning > 80 m) can overlap across adjacent 80 m block boundaries and inflate local CV; verify such features under coarser spatial holdouts or wider buffers alongside 80 m CV. Construct any out-of-fold spatial priors, pseudo-labels, or stacked features using block-disjoint folds so training feature distributions match held-out blocks, and test far-field robustness (e.g., quadrant/region holdouts or repeated spatial-block seeds) before trusting large spatial gains.

## Submission & Execution Contract

### 1. `/app/submission/pipeline.py` and `/app/submission/requirements.txt`

`/app/submission/pipeline.py` must define:

```python
def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
```

- Invoked once per estate (`AirHitam`, `Palong`, `Serting`) in a fresh subprocess with `cwd = ctx["scratch_dir"]`.
- `labeled`: DataFrame for the current estate from `L_labelled.csv` (includes `Middle` rows where present; filter them out if unused).
- `unlabeled`: DataFrame for the current estate from `unlabelled.csv` (includes unmarked `Middle` rows; **never drop or reorder rows**).
- `ctx`: dict containing `"estate"` (`str`), `"datasets_dir"` (`Path` to `/app/datasets`), `"other_labeled"` (`dict[str, pd.DataFrame]` holding the other two estates' `L` tables), `"scratch_dir"` (`Path`, writable temp directory), and `"cpus"` (`int`; use `ctx["cpus"]` rather than `os.cpu_count()`).
- Return value: 1D finite `float64` NumPy array of exact length `len(unlabeled)` aligned to `unlabeled` rows, where larger values indicate higher `Unhealthy` risk.
- Helper `.py` modules placed inside `/app/submission/` are permitted.

`/app/submission/requirements.txt` must list only exact `package==version` PyPI pins (or be empty / comments only). Pip flags (`-e`, `-r`, `--...`), local paths, and URLs are forbidden.

**Execution-environment differences from this development container:**
- The isolated evaluation runtime runs Python 3.12 with **only** `numpy==2.5.5`, `pandas==3.1.1`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `pyarrow==26.0.1`, and `pyyaml==6.0.3` preinstalled. Any other package imported by `/app/submission/pipeline.py` — **including `xgboost`, `lightgbm`, `geopandas`, or `rasterio` even though they are preinstalled here** — must be explicitly pinned in `/app/submission/requirements.txt`.
- Runtime caps: {{VERIFIER_CPUS}} CPUs, {{VERIFIER_MEMORY_MB}} MB RAM, {{PER_ESTATE_TIMEOUT_SEC}} s per estate, and {{PIP_INSTALL_TIMEOUT_SEC}} s for `pip install`.
- Isolation rules during `fit_predict()`: `/app/submission` and `/app/datasets` are **read-only**; `/app/data`, `/app/rasters`, and `/app/prior` are **not present**; write temporary files only inside `ctx["scratch_dir"]`; perform **zero network calls** inside `fit_predict()`; and place **no symlinks** anywhere under `/app/submission/` or `/app/datasets/`.

### 2. Optional Derived Covariates (`/app/datasets/<name>/`)

If you extract new per-tree features (from `/app/rasters/` or Google Earth Engine), save them under `/app/datasets/<name>/` with:
- `/app/datasets/<name>/features.parquet` (or `features.csv`)
- `/app/datasets/<name>/MANIFEST.md` (source, acquisition window, band formulas, masking/preprocessing, NaN summary)
- The reproducible extraction script used to build it

Every `features.parquet` / `features.csv` must satisfy:
- Key columns `estate`, `Long`, `Lat` covering all 7,220 trees (`L_labelled.csv` ∪ `unlabelled.csv`) with zero duplicate keys and exact float coordinates preserved from `/app/data/*.csv` (do not round `Long`/`Lat`).
- Zero forbidden column names (case-insensitive): `label`, `class`, `id`, `fid`, `objectid`, `tree_id`, `treeid`, `row_id`, `block`, `part`.
- Total `/app/datasets/` size <= {{MAX_DATASETS_MB}} MB (keep under {{PRIOR_DATASETS_MAX_MB}} MB so derived datasets automatically carry forward into the next session) and zero symlinks.
- `fit_predict()` must load derived datasets via `ctx["datasets_dir"]`.

### 3. Research Documentation (`/app/report/`)

- `/app/report/REPORT.md` — full technical report documenting hypotheses, experimental table (per-estate and composite 80 m spatial-block CV lift, paired deltas vs incumbent and vs coords-only baseline), feature/error analysis (including cross-estate sign consistency), literature citations, and caveats.
- `/app/report/HANDOFF.md` — structured handoff for the next session containing **these exact six H2 headings**:
  - `## State of play`
  - `## What I tried` (idea, per-estate & composite CV result, verdict, evidence path)
  - `## Dead ends`
  - `## Open leads, ranked`
  - `## How to resume`
  - `## Artifact index`

## Research Latitude & Operational Discipline

- During this session you have public internet access for remote-sensing / plant-pathology literature, PyPI packages, and Google Earth Engine (`agentrsi gcp-init`).
- You have full modeling latitude: multi-date SAR polarimetry/change, window/texture statistics, optical/terrain/moisture covariates via GEE, semi-supervised or transductive learning over `labeled` + `unlabeled`, `Middle`-class ordinal/spatial signals, cross-estate transfer via `ctx["other_labeled"]`, and probability/rank calibration — provided `fit_predict()` stays offline and self-contained.
- **Research depth & persistence:** Treat the session budget as an iterative multi-hypothesis research campaign, not a stop-at-first-pass benchmark. Once you establish a working candidate that beats `/app/incumbent/pipeline.py`, checkpoint it and keep pushing across at least 3–4 orthogonal scientific directions (for example: multi-date ALOS-2 `261014` vs `260610` change, Yamaguchi dB decomposition and multi-scale `winstats` from `/app/rasters/`, Sentinel-2 red-edge/SWIR/topography/moisture covariates via GEE, block-honest spatial/semi-supervised/`Middle`-class signals, and disciplined ablations) until you observe clear diminishing returns or approach the final checkpoint window.
- **Generalizable recipe design:** Prefer data-driven per-estate model selection (e.g., choosing regularization or feature complexity via inner 80 m spatial-block CV on `labeled`) over hardcoding estate-name branches (`ctx["estate"] == ...`), so the training recipe generalizes cleanly to unseen estates.
- **Small-sample collinearity & sign discipline:** With only ~56–67 positives per estate, jointly fitting many correlated contextual features can dilute or invert core physical predictors. Inspect conditional weights alongside marginal cross-estate signs, and compare joint fits against compact baselines with bounded residual/offset refinements or sign-consistent feature subsets.
- **Checkpointing:** Within your first 5–10 minutes, copy `/app/incumbent/pipeline.py` (and `requirements.txt`) to `/app/submission/` and write an initial `/app/report/HANDOFF.md` skeleton containing all six required H2 headings. Refresh `/app/submission/` and `/app/report/HANDOFF.md` after every verified experiment (and at least every 30 minutes) so an early exit or timeout still preserves your strongest verified nominee and live research log.
- **Pre-exit verification:** Before finishing, run `agentrsi selfcheck --submission /app/submission` and `agentrsi cv --submission /app/submission`, and verify that every non-base import in `/app/submission/pipeline.py` is pinned in `/app/submission/requirements.txt`.

## Constraints

- Development container is CPU-only ({{AGENT_CPUS}} CPUs, {{AGENT_MEMORY_MB}} MB RAM); respect `AGENTRSI_CPUS={{AGENT_CPUS}}` and `ctx["cpus"]`.
- Never touch `/tests` or attempt to recover sealed labels or split assignments.
- Never access, query, or clone `https://github.com/Leejames2305/SAR-OilPalm-Detection`.
- Never print, log, or copy `/run/secrets/gcp-sa.json` or any API keys/credentials into `/app/` or stdout/stderr.

---

{{FOCUS_LENS_SECTION}}

{{HANDOFF_SECTION}}

You have {{AGENT_TIMEOUT_SEC}} seconds to complete this task. Use public literature, PyPI, and Google Earth Engine freely for research, but never access https://github.com/Leejames2305/SAR-OilPalm-Detection, touch /tests, attempt to recover sealed labels, or expose credentials.
