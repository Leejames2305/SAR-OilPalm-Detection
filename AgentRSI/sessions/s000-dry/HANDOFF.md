## State of play
Nominee in `/app/submission/`: NDVI + leave-one-out kNN spatial label prior (K=10), balanced logistic. `agentrsi cv` composite L-lift +0.0929 (incumbent +0.0454). `agentrsi selfcheck` passes. Gain is mostly Palong; Serting and AirHitam are flat or slightly negative.

## What I tried
- **Baseline NDVI-LogReg (incumbent)**: CV composite +0.0454. Verdict: reference. Evidence: `agentrsi cv --submission /app/incumbent`.
- **kNN label prior + NDVI, K=10 (nominee)**: CV composite +0.0929 (Palong +0.186, Serting +0.079, AirHitam +0.014). Verdict: keep, pending a far-field test. Evidence: `/tmp/exp/knn/pipeline.py`, `/app/report/REPORT.md`.
- **kNN prior, K=5**: composite +0.0863. Verdict: consistent gain. Evidence: `/tmp/exp/k5`.
- **kNN prior, K=20**: composite +0.0811. Verdict: consistent gain. Evidence: `/tmp/exp/k20`.

## Dead ends
Carried over from `/app/prior/MainChecklist.md` §B (not re-run here): class-imbalance methods, G-SMOTE tuning, RF feature selection, PCA/QDA, healthy-only anomaly detection, RVI features, temporal S1/S2 on top of NDVI, Middle-as-positive, cross-estate pooling, threshold tuning. Deep/MLP models at n≈160 positives.

## Open leads, ranked
1. **Far-field test of the kNN prior**: corner holdout (labels in one quadrant, score the rest). If the gain vanishes, the prior only works near labels; decide whether the product's deployment setting is near-labels.
2. **Distance-aware prior**: add a distance-to-nearest-labelled feature, or use a Gaussian kernel with a bandwidth tuned per estate. Could fix Serting's sign flip.
3. **Serting stability**: K sensitivity (Serting negative at K=10, positive at K=5). Check with repeated block seeds before trusting.
4. **Combine with spatial SAR fusion** from the checklist (coords + SAR), once the prior is verified in the far-field test.
5. Literature search on spatial label propagation for point-level disease mapping (not done).

## How to resume
```
cd /tmp && agentrsi cv --submission /app/submission
agentrsi selfcheck --submission /app/submission
```
Candidate source: `/app/submission/pipeline.py` (K is the module constant `K`). Scratch variants live in `/tmp/exp/`, which may not persist.

## Artifact index
- `/app/submission/pipeline.py`: nominee (kNN prior + NDVI logistic).
- `/app/submission/requirements.txt`: no pins.
- `/app/report/REPORT.md`: technical report with results table and caveats.
- `/app/report/HANDOFF.md`: this file.
- `/tmp/exp/knn`, `/tmp/exp/k5`, `/tmp/exp/k20`: K-sensitivity variants (scratch).
