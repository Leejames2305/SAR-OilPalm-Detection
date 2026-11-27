## User Prompt (Manual Researcher Observations & Guidance)

> **Note:** This section is a **manual user prompt** written directly by the researcher (not an automated handoff). It reflects **user-side observations** from reviewing sessions `s009`–`s015` and sets the primary objective for this run. Verify every hypothesis empirically on `L`, `unlabelled.csv`, and M7 `V`.

### Primary Mission for This Session: Focus on `AirHitam` & Ship a Balanced, Tail-Safe Upgrade Over `/app/incumbent/`

1. **Critical course correction — judge candidates against `/app/incumbent/` (`s001`, V `+0.1188`), NOT unpromoted `s014` (`+0.2247`):**
   - **Every session from `s010` through `s015` failed promotion (`promoted = False`)**, even as composite `V` climbed to `+0.2247`.
   - In `s015`, you discovered two tail-safe / strongly regularized milestones that **matched `s009`'s all-time-best `AirHitam` `V` (`+0.0655`) while STILL beating `/app/incumbent/` by `+0.070` to `+0.076` across all three estates on `V`**:
     - `V2` (`clip3_C02`: Clip `±3`, `C = 0.02`): `AirHitam V = +0.0654 (+0.0284 vs inc)`, `Palong V = +0.3022 (+0.0489)`, `Serting V = +0.1974 (+0.1315)`, composite `+0.1883 (+0.0696 vs inc)`.
     - `V3` (`guard25`: physical-base fallback below `z < -2.5`): `AirHitam V = +0.0651 (+0.0280 vs inc)`, `Palong V = +0.2625 (+0.0092)`, `Serting V = +0.2564 (+0.1905)`, composite `+0.1947 (+0.0759 vs inc)`.
   - `s015` discarded both `V2` and `V3` and fell back to `s014` **only because its self-imposed rule required beating unpromoted `s014` (`+0.2247`)**! **Do NOT repeat that mistake.** A balanced, tail-safe model that protects `AirHitam` and delivers `+0.07` to `+0.10` composite lift over `/app/incumbent/` is far better than chasing `+0.225` composite `V` with an inverted `AirHitam` tail.

2. **Why `AirHitam` is the #1 focus this round (confirmed by `s015`'s own audit):**
   - `s015` proved that `s014` **inverts `AirHitam`'s `unlabelled.csv` low-`NDVI` tail**:
     - The `92` `AirHitam` `U` trees below `-2.5 SD` raw `NDVI` average the **21st risk percentile** (`1 / 92` in top 25%) under `s014`, versus **85th** under `/app/incumbent/` and **83rd** (`63 / 92` in top 25%) under `s009`.
     - The `57` `AirHitam` `U` trees below `-3 SD` drop to the **16th percentile** under `s014` vs **86th** under `s009`, and overall `AirHitam` `U` `Spearman(score, -NDVI)` flips to **`-0.133`** (`s014`) vs **`+0.529`** (`s009`).
   - Because `AirHitam` `L` happens to have `0 / 16` positives below `-3 SD` (and weak ~`2×` spatial clustering, narrower `L` `NDVI` variance, and opposite marginal signs on `ndvi_sd20` / `ndvi_edge` / `height_mean6`), light-regularization 49-feature joint models (`C = 0.15` + quadratic `NDVI^2` + collinear `log_green_red` / `deblur_GRVI` suppressors + cross-estate optical priors) actively push severely low-`NDVI` `AirHitam` crowns to the bottom of the risk ranking.
   - **Do NOT waste time re-testing `s015` dead ends** (block DRO, `Cov(NDVI, logit) <= 0` moment constraints, `late_psri`, `temporal_sar` CNN, or clustering-ridge heuristics — those did not resolve the `AirHitam` tail inversion or improve `AirHitam` `V`).

---

### Concrete Directions to Push Through on `AirHitam` (While Keeping Strong `Palong` & `Serting` Gains)

The exact code for `s015`'s `V2` (`clip3_C02`) and `V3` (`guard25`) is already preserved under `/app/datasets/triage_v12/recipes/s015/verified/`, and `s009`'s full code is under `/app/prior/ref_sessions/261125-s009-open/submission/`. Start directly from those insights:

1. **Fix the `AirHitam` tail inversion cleanly — and diagnose why `guard25` / `clip3_C02` lost `Palong` or `Serting` `V`:**
   - Look closely at `V3` (`guard25`): it recovered `AirHitam V = +0.0651` AND kept `Serting V = +0.2564`, only dropping `Palong V` to `+0.2625` — check whether a **scale/rank mismatch** between the fallback score and the in-support score (or firing the fallback too aggressively on `Palong`) caused that drop! Test smooth, rank- or z-score-calibrated tail transitions, or applying the fallback only where `L` support is genuinely sparse/extrapolating.
   - Look closely at `V2` (`clip3_C02`): it hit `AirHitam V = +0.0654` AND kept `Palong V = +0.3022`. Test whether **input clipping (`±2.5` or `±3`) + removing `NDVI^2` & `NDVI × coord` extrapolation + strong differential shrinkage on high-D context blocks (`C` / block weights) WITHOUT pooling sign-flipping texture features (`ndvi_sd20`, `ndvi_edge`)** unlocks **both** `AirHitam V >= +0.065` **and** stronger `Palong` (`+0.30–+0.34`) + `Serting` (`+0.22–+0.27`) `V`!

2. **Test a truly protected low-D monotonic core (`s009`-style frozen base) + bounded OOF residual with modern features:**
   - Remember why `s009` achieved the highest `AirHitam` `V` (`+0.0655`) and kept `AirHitam` `U` `Spearman(score, -NDVI) = +0.529`: its linear `NDVI/qNDVI + coords` base stayed dominant and could never be inverted by high-D collinear features.
   - Explore a **protected monotonic core + bounded OOF residual** (or **strong-shrinkage linear base + clipped/regularized residual** using `triage_v12` + `dino_sar`) where the core vegetation-vigor signal (`-NDVI` / `-qNDVI` / `-qNDRE`) remains strictly intact on unseen low-`NDVI` tails, while richer context (`alphaearth`, `highres_canopy`, `crown_mixture`, `dino_sar`) refines rankings inside the well-supported body of each estate.
   - Test dropping cross-estate optical source pooling altogether vs pooling only the strictly sign-consistent vegetation indices (`qNDVI`, `qNDRE`, `NDVI_contrast35`, `NDRE1_contrast35`, `ndvi_dip`, `ndre_contrast`) — never pooling `ndvi_sd20`, `ndvi_edge`, or `log_green_red`.

3. **Target shipping criteria (strictly anchored to `/app/incumbent/`):**
   - **Must fix `AirHitam` tail safety:** verify positive `Spearman(score, -NDVI)` on `AirHitam` `unlabelled.csv` (and sane high risk percentiles on the `z < -2.5` / `z < -3.0` low-`NDVI` tail — no longer inverting severely low-`NDVI` crowns to the bottom 20%!), **and** aim for `AirHitam V >= +0.065` (matching/beating `s009`, clearly ahead of `/app/incumbent/` `+0.0370` and `coords_only` `+0.0398`).
   - **Strong balanced gains over `/app/incumbent/` across all 3 estates:** target composite `V` well above `/app/incumbent/` (`+0.1188` — e.g., `+0.19` to `+0.22+`, `Δ >= +0.07` vs incumbent) with solid `Palong` (`>= +0.30`) and `Serting` (`>= +0.20`) `V`, strong separation from `coords_only`, and clean offline CPU execution.
   - **Do NOT fall back to `s014`!** Even if a balanced, `AirHitam`-safe model has a slightly lower composite `V` than unpromoted `s014` (`+0.20` vs `+0.225`), **ship the `AirHitam`-safe balanced model** if it decisively beats `/app/incumbent/` across all 3 estates and eliminates `AirHitam`'s tail inversion!
