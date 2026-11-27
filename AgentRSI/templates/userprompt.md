## User Prompt (Manual Researcher Observations & Guidance)

> **Note:** This section is a **manual user prompt** written directly by the researcher (not an automated handoff). The points below represent **user-side domain and empirical observations** from reviewing past sessions, prior POC reports, and verified `V` behavior. Treat these as high-priority hypotheses to investigate, verify, and test empirically on `L`, `unlabelled.csv` covariate distributions, and M7 `V` — do not hardcode arbitrary rules without checking the data yourself.

### 1. User-Side Observations: What May Be Going Wrong on `AirHitam`

Reviewing sessions `s009` through `s014` side-by-side reveals a striking pattern specific to `AirHitam`:

1. **`L`-CV vs verified `V` disconnect on `AirHitam`:**
   - Recent high-capacity joint recipes (`s010`–`s014`, 29–49 local features at `C = 0.10–0.15`) push `AirHitam` honest 80 m `L`-CV up to `+0.15–+0.16`, yet **none of them has surpassed `s009`'s verified `AirHitam` `V` lift (`+0.0655`)**, even though `s009` only scored `+0.0301` on `AirHitam` `L`-CV.
   - This suggests that much of the apparent `AirHitam` `L`-CV gain in `s010`–`s014` may be fitting `L`-specific pockets and collinear noise rather than generalizable disease signal.

2. **Different spatial & biophysical regime on `AirHitam` vs `Palong` / `Serting`:**
   - As visible directly in `L_labelled.csv`, `/app/datasets/`, and `/app/prior/POCReport/`, `AirHitam` has much weaker spatial clustering (~`2×` 10-NN positive enrichment vs ~`3.7–5.7×` in `Palong`/`Serting`), narrower `L` `NDVI` variance, no `Middle` class, weak prior-year `AlphaEarth` / historical height separation, and **opposite marginal signs** on canopy texture/structure variables (`ndvi_sd20`, `ndvi_edge`, `height_mean6`) compared to `Palong` and `Serting`.
   - Fitting the same 30–50 feature joint model at light regularization (`C = 0.10–0.15`) across only ~`56` `AirHitam` positives risks severe variance and domain mismatch.

3. **Unlabelled (`U`) tail shift & quadratic / collinear extrapolation risk:**
   - Inspecting `unlabelled.csv` vs `L_labelled.csv` shows unseen trees in the extreme low-`NDVI` / high-edge canopy tail that are sparse or absent in `AirHitam` `L`.
   - In joint unclipped polynomial models (`s011`–`s014`), negative `NDVI^2` curvature, `NDVI × coordinate` interactions, and large collinear suppressor pairs (e.g., strong positive weights on `log_green_red` / `deblur_GRVI` opposing negative `NDVI`/`qNDVI`/`qNDRE`, or inverted conditional signs on `ndvi_dip`, `ndre_contrast`, and `deblur_*`) can **dilute or even invert** the core physical vegetation-stress response (`-NDVI` / `-qNDVI`) on out-of-range `U` trees.

4. **Cross-estate pooling / source-prior sign contamination:**
   - In `s010`, `s013`, and `s014`, sharing or prior-centering all 12 optical coefficients across all three estates transfers `Palong`/`Serting`'s strong negative conditional weights on `ndvi_sd20` / `ndvi_edge` (and suppressor patterns) onto `AirHitam`, even though those texture features have the **opposite marginal sign** on `AirHitam`.

---

### 2. Suggested Directions to Investigate & Verify Yourself

Please test these systematically using honest union-anchored (`L ∪ U`) 80 m spatial-block CV, coarse/buffered/quadrant holdouts, `L`-vs-`U` covariate diagnostics, and strategic M7 `score_v` checks:

1. **Audit `L` vs `U` tail behavior & monotonicity on `unlabelled.csv` (no labels needed):**
   - Compare `s001`, `s009`, `s012`, `s013`, and `s014` predictions specifically on `unlabelled.csv` trees in the **extreme low-`NDVI` tail** (`zNDVI < -2.5` or `-3.0`) and high-`ndvi_sd20`/`ndvi_edge` regions of each estate.
   - Check whether joint/quadratic models are actually assigning *lower* risk scores (or inverting correlation with `-NDVI` / `-qNDVI`) on severely degraded crowns in `unlabelled.csv`!

2. **Test tail-safe input bounds & strong regularization (not just dropping a few columns):**
   - Removing only a handful of columns in a joint 40-feature fit often lets the remaining correlated predictors re-absorb the same overfit.
   - Instead, test **architectural restraint**:
     - Standardized input clipping (e.g., `[-3, 3]` or Winsorization) and removing/bounding `NDVI^2` and `NDVI × coord` extrapolation terms.
     - Much stronger ridge shrinkage (`C` swept down to `0.002–0.02`) and/or stronger differential penalty on contextual/optical/patch blocks relative to the core physical vegetation signal.

3. **Revisit `s009`'s protected low-D base + bounded OOF residual — paired with data-driven per-estate shrinkage:**
   - Note that `s009` preserved its top `AirHitam` `V` score (`+0.0655`) by **freezing a compact linear `NDVI/qNDVI + coords` base** and learning only clipped, heavily regularized, damped OOF residuals on top of it.
   - Test an **estate-agnostic, data-driven** architecture that can adapt its effective complexity / residual strength to each estate's observed training regime (e.g., via inner union-anchored spatial CV, spatial-clustering / SNR diagnostics, or `L`-vs-`U` support/tail shift): staying strongly anchored near the monotonic low-D physical base when spatial clustering is weak or `L`-to-`U` covariate shift is high, while leveraging richer `highres_canopy`, `crown_mixture`, `alphaearth`, and `dino_sar` context where signal-to-noise and spatial support are strong.

4. **Restrict cross-estate transfer to sign-consistent physical features (or test no pooling):**
   - If using partial pooling or a source-centered ridge prior (`s014`), test restricting shared/prior-centered coefficients strictly to features whose marginal `Unhealthy - Healthy` signs are **consistent across all source and target estates** (excluding sign-flipping texture/structure features like `ndvi_sd20`, `ndvi_edge`, `height`), and compare directly against zero-pooling.

5. **Validation & selection discipline:**
   - Compare every candidate against `incumbent`, `coords_only`, raw `-NDVI`, `s009`, and `s014`.
   - Use M7 `score_v` selectively to check whether a tail-safe, sign-disciplined, adaptively regularized architecture can **simultaneously preserve or beat `s009`'s `AirHitam` `V` (`+0.0655`) while keeping strong `Palong` and `Serting` `V` gains** — without hardcoding estate-name branches (`ctx["estate"] == ...`).
