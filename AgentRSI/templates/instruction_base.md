# AgentRSI Session: {{SESSION_ID}}

## Mission
Build the strongest self-contained **oil-palm disease triage model** (`Healthy` vs `Unhealthy`) across three Malaysian estates (**AirHitam**, **Palong**, **Serting**).
The product shape: given a small labelled slice (`L`, ~667–1,004 trees per estate) and the rest of the field unlabelled (`V+T`), train a reproducible training recipe `fit_predict(labeled, unlabeled, ctx) -> risk_scores` that ranks `Unhealthy` trees highest for downstream targeted inspection.

All three estates are verified and trusted equally. Your target metric is **composite validation lift** = equal-weight mean across the three estates of `(PR-AUC - prevalence)`.

---

## Harness Map (Inside `/app`)
- `/app/data/L_labelled.csv`: combined labelled slice (`2,747` rows across `AirHitam`, `Palong`, `Serting`). Columns: `estate`, `Long`, `Lat`, multi-date ALOS-2 SAR (`HH_<date>`, `HV_<date>`, `VH_<date>`, `VV_<date>`, `HAlpha_H_<date>`, `HAlpha_A_<date>`, `HAlpha_alpha_<date>` for `260610` and `261014`), Sentinel-2 (`B4`, `B8`, `NDVI`, `clear_count`), and `label` (`Healthy`, `Unhealthy`, or `Middle`).
- `/app/data/unlabelled.csv`: combined unlabelled pool (`4,473` rows) with the same feature columns and **no** `label`. `Middle` trees are unmarked here (as in a real field) and are excluded before metric scoring.
- `/app/rasters/`: ROI-masked ALOS-2 GeoTIFF scenes for all three estates (`260610` and `261014`, including `Spk`, `HAlpha`, `Yamaguchi`, and AirHitam `Cal_mat_Spk_TC`).
- `/app/prior/`: prior POC reports (`POCReport/*.md`), `MainChecklist.md` (settled facts and dead ends — read before trying old dead ends!), `Plan-GEECatalogSurvey.md`, and `/app/incumbent/` (current best `pipeline.py` + `requirements.txt`).
- Helper CLI `agentrsi`:
  - `agentrsi cv [--submission /app/submission]`: 5-fold 80 m spatial-block CV on `/app/data/L_labelled.csv` vs NDVI baseline.
  - `agentrsi selfcheck [--submission /app/submission]`: verifies `requirements.txt` pins and tests `fit_predict()` across all three estates.
  - `agentrsi gcp-init`: checks Earth Engine initialization (`$GOOGLE_APPLICATION_CREDENTIALS` at `/run/secrets/gcp-sa.json`, `$GCP_PROJECT`).

---

## Pipeline Contract
Your submission directory `/app/submission/` **must** contain:
1. `/app/submission/pipeline.py` defining:
   ```python
   def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
   ```
   - Called **once per estate** (`AirHitam`, `Palong`, `Serting`).
   - `labeled`: DataFrame for that estate (includes `Middle` rows; filter them out if not using them).
   - `unlabeled`: DataFrame for that estate (unmarked `Middle` rows included; do NOT drop rows!).
   - `ctx`: dict with `"estate"` (str), `"datasets_dir"` (`Path` to `/app/datasets`), `"other_labeled"` (`dict[str, pd.DataFrame]` for the other two estates' L data), `"scratch_dir"` (`Path`), and `"cpus"` (`int`, respects container quota `AGENTRSI_CPUS={{AGENT_CPUS}}` — do NOT rely on `os.cpu_count()`).
   - Must return a **1D finite float array** of exact length `len(unlabeled)` aligned to `unlabeled` rows (higher score = higher risk of `Unhealthy`).
   - Must be **self-contained at grading time**: read only `labeled`, `unlabeled`, files inside `/app/submission/` or `/app/datasets/`, and never make network calls during `fit_predict()`.
2. `/app/submission/requirements.txt`: exact `package==version` PyPI pins only (or empty/comments if base packages suffice: `numpy`, `scipy`, `pandas`, `scikit-learn`, `pyarrow`, `pyyaml` are preinstalled in the verifier). Local paths, URLs, and pip flags (`-e`, `-r`, `--...`) are rejected.

---

## Deliverables (Absolute Paths)
1. `/app/submission/pipeline.py` and `/app/submission/requirements.txt`
2. `/app/report/REPORT.md` — full technical report (hypotheses, experiments, spatial-CV results, feature/error analysis, literature notes).
3. `/app/report/HANDOFF.md` — structured handoff for the next session with **these exact markdown H2 headings**:
   - `## State of play`
   - `## What I tried` (include idea, CV result, verdict, evidence path)
   - `## Dead ends`
   - `## Open leads, ranked`
   - `## How to resume`
   - `## Artifact index`
4. *(Optional, for new covariates)* `/app/datasets/<name>/features.parquet` keyed by `(estate, Long, Lat)` covering all `7,220` trees (zero forbidden columns like `label`/`class`/`id`/`block`/`part`), plus `MANIFEST.md` and the extraction script. Total `/app/datasets/` size must stay under `512 MB`.

---

## Freedom & Resources
- You have public internet access for PyPI packages, literature search, and Google Earth Engine (GEE) data expansion (`earthengine-api`, `geopandas`, `rasterio`, `xgboost`, `lightgbm` are installed).
- Pseudo-labeling / semi-supervised learning / spatial & transductive features using `labeled` + `unlabeled` (without V/T label assumptions) are permitted inside `fit_predict()`.
- **Checkpoint rule:** Write/update `/app/report/HANDOFF.md` and keep a working `/app/submission/pipeline.py` early and refresh at least hourly so a hard timeout still preserves a valid nominee.
- **Nominee rule:** Declare your single best verified pipeline at `/app/submission/` before finishing, and run `agentrsi selfcheck` + `agentrsi cv` on it.

---

## Constraints
- **CPU only** (`{{AGENT_CPUS}}` CPUs; verifier runs with `4` CPUs, `300s` per estate max).
- **Never** attempt to touch `/tests` or recover sealed labels.
- **Never** access or clone `https://github.com/Leejames2305/SAR-OilPalm-Detection`.
- **Never** print, log, or copy `/run/secrets/gcp-sa.json` or any API keys/credentials into `/app/`.

---

{{FOCUS_LENS_SECTION}}

{{HANDOFF_SECTION}}
