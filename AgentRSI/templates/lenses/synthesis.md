## Focus Lens: Cross-Session Synthesis & Crossover

Your goal is to perform an **empirical cross-lineage synthesis** across `/app/incumbent/`, `/app/prior/prev_submission/`, and every reference session staged under `/app/prior/ref_sessions/<session_id>/`, diagnose why different sessions excel or falter on different estates, and synthesize their strongest complementary mechanisms into a single unified `/app/submission/pipeline.py`.

Work through three disciplined phases:

1. **Cross-lineage empirical audit (paired benchmarks, not prose-only review):**
   - Read `/app/prior/MainChecklist.md`, `/app/prior/prev_HANDOFF.md`, `/app/prior/prev_REPORT.md`, and the `HANDOFF.md`, `REPORT.md`, and `submission/` of every staged reference session under `/app/prior/ref_sessions/<session_id>/`, alongside the verifier-verified V comparison table in the handoff section below.
   - Run `/app/incumbent/pipeline.py`, `/app/prior/prev_submission/pipeline.py`, and every staged reference submission against `/app/datasets/` under the **exact same** validation protocols: official 80 m spatial-block CV (`agentrsi cv`), repeated 80 m block seeds, coarser/buffered spatial holdouts (e.g., 120–160 m blocks, 40–80 m buffers, quadrant holdouts), and the coordinates-only baseline.
   - Diagnose the **mechanistic causes** of per-estate divergence across sessions: compare each lineage's L-CV profile against its verified V profile, measure pairwise prediction rank correlations and top-25% disagreement across lineages, inspect fitted conditional weights versus marginal cross-estate signs (flagging collinear suppressor pairs or sign inversions that dilute core physical predictors on ~56–67 positives/estate), and separate architectural effects (e.g., frozen-baseline / bounded OOF residual offsets vs joint fits, regularization, pooling, bagging) from feature-family effects.

2. **Mechanistic crossover & controlled synthesis:**
   - Formulate concrete, testable crossover hypotheses that recombine the best validated mechanisms from different lineages — for example, pairing a lineage's stronger regularization or bounded OOF-residual/offset structure with another lineage's highest-signal feature families, while pruning collinear, noisy, or estate-brittle features.
   - Avoid naive score-averaging hacks or fitting ensemble weights to noisy V. Every synthesized candidate must earn its complexity under paired 80 m CV, repeated seeds, and coarser/buffered/quadrant holdouts against `/app/incumbent/`, the coordinates-only baseline, and the individual parent/reference lineages.
   - Do not spend time re-extracting already-available datasets under `/app/datasets/` unless your diagnosis identifies a clear, specific gap that requires a compact new feature or remote GPU step (if remote GPU offload is enabled in the MCP Tools section above, stop the VM promptly when finished).

3. **Strategic V verification, clean delivery, and consolidation handoff:**
   - When in-run noisy-V verification is enabled in the MCP Tools section above, spend queries strategically on genuinely distinct, L-verified synthesized architectures (staging each submission and only its required datasets under `/shared/<subdir>/`) to confirm balanced gains across **all three estates** (`AirHitam`, `Palong`, `Serting`) without degrading any estate or losing separation from the coordinates-only baseline.
   - Ship the cleanest, self-contained offline CPU `/app/submission/pipeline.py` (with exact `/app/submission/requirements.txt` pins), pass `agentrsi selfcheck` and `agentrsi cv`, and include in `/app/report/REPORT.md` and `/app/report/HANDOFF.md`:
     - A unified side-by-side comparison table of `incumbent`, `coords`, all audited parent/reference lineages, and your synthesized candidates.
     - Explicit attribution of which mechanisms and feature families from which sessions were adopted versus retired, with empirical evidence.
     - Concrete recommendations for the next session: which carried datasets under `/app/datasets/` are essential vs safe to prune, and the highest-leverage next research directions.
