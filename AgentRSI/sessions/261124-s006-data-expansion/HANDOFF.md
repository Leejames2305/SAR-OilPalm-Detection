## State of play

The submission now uses the verified coordinate+NDVI incumbent with a conservative fixed 10% blend of a compact current-season Sentinel-2 raw-band model. `agentrsi selfcheck` passes. Final local five-fold 80 m block CV is AirHitam +0.0355, Palong +0.2592, Serting +0.1316, composite +0.1421. The carried incumbent was +0.0310, +0.2556, +0.1350, composite +0.1405. Thus the candidate improves AirHitam and Palong and has a small Serting decrease (0.0034 lift), below a material degradation threshold used for this checkpoint.

## What I tried

- Sentinel-2 current raw bands + NDVI + coordinates, degree-2 logistic, C=0.03: useful AirHitam but weak in Palong/Serting; retained only as a 10% blend.
- Prior-season red-edge/SWIR indices: complementary in Palong but unstable across folds; rejected as a direct model.
- Fold-wise blend selection: selected highly variable weights (0.05–0.85) and scored composite +0.1318; rejected for instability.
- Fixed 10% current-band blend: composite +0.1421, per-estate +0.0355/+0.2592/+0.1316; current nominee. Evidence: `/app/submission/pipeline.py`, `agentrsi cv --submission /app/submission`.
- Topography/drainage (SRTM + MERIT Hydro) and S2-topography combinations: no consistent improvement; rejected.

## Dead ends

- Local ALOS window statistics are complete in `alos_winstats` and were not re-extracted.
- S2 seasonal deltas and broad red-edge/SWIR feature sets overfit or failed to improve all estates.
- Topographic covariates did not add robust held-out lift.

## Open leads, ranked

1. Repeat the fixed-blend comparison over additional spatial seeds or quadrant holdouts.
2. Test a pre-registered prior-season raw-band blend with stronger shrinkage.
3. Evaluate block-level hotspot aggregation as a separate operational target.

## How to resume

Read `/app/prior/MainChecklist.md`, run `agentrsi selfcheck --submission /app/submission`, then `agentrsi cv --submission /app/submission`. Keep the coordinates-only comparator and report estate-level deltas; do not treat the small composite gain as settled without repeated seeds.

## Artifact index

- `/app/submission/pipeline.py` — current fixed-blend nominee.
- `/app/submission/requirements.txt` — empty/comment-only; all imports are verifier-provided.
- `/app/datasets/gee_s2_stress/features.parquet`, `extract.py`, `MANIFEST.md` — 7,220-row Sentinel-2 stack.
- `/app/datasets/gee_topo/features.parquet`, `extract.py`, `MANIFEST.md` — 7,220-row topography/drainage stack.
- `/app/datasets/alos_winstats/` — carried local ALOS covariates.
- `/app/report/REPORT.md` — technical report.
