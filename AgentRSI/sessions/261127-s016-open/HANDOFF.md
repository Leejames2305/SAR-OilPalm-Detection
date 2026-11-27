# Session 261127-s016-open

## State of play

**Ship budgetfloor25, not s014.** `/app/submission` is byte-identical to successful V query2. Designated s001 incumbent is unchanged. All three estates' labels are equally trusted; no estate-name branch.

V noisy lift A/P/S **+.072622 / +.336414 / +.279586**, composite **+.229541**, delta incumbent **+.110790**, coords **+.163079**. Official union80m L **+.157692 / +.310163 / +.384886**, composite **+.284247**, delta incumbent **+.161263**. This meets the requested balanced targets and improves all estates over s001.

AirHitam U Spearman(score,−NDVI)=**+.0920**, versus s014−.1326. For z<−2.5, mean risk percentile**80.82**, **71/92** top-quarter; z<−3, **86.82**, **55/57**. This is operational ranking safety, not proof of U disease outcomes. All16 L palms below−3SD are Healthy.

Ten frozen repeated whole-block assignments beat incumbent **10/10 in every estate**; mean composite gain+.120616. Conditional4000-block-bootstrap composite delta95%[+.080731,+.230362]; Palong individual interval crosses zero. At top25 binary inspection recall**44.6/65.2/74.6%**, inclusive Middle**44.6/62.1/71.6%**. AirHitam majority capture is unmet; Palong top25 recall falls versus incumbent despite AP/top10 improvements.

Final selfcheck and CLI CV pass. Fresh four-CPU prediction≤0.206s/estate, peak≤277.7MiB, exact score/checkpoint equivalence. Only non-base pin threadpoolctl==3.8.0. Six feature tables total32.14MiB,7220 exact keys each,0NaNs/duplicates/forbidden names/symlinks. Required runtime datasets triage_v12,dino_sar,gee_quality. No new GPU job; no active VM. V calls5/15, including failed staging call1;10 remain. No additional V tuning is recommended without a distinct mechanism.

## What I tried

Values below are A/P/S official80m lift, composite; evidence in OFFICIAL_SUMMARY.csv, EXPERIMENT_SUMMARY.csv, actual OOF arrays research/results and frozen configs candidates/.

- **Capped calibrated upward-only guard**, .157692/.310163/.384886, composite.284247; V.229541. Selected. Quantile-match low-D core to full scores, cap near training75th percentile plus weak core slope, smooth z−2.5 transition; never overwrite scores downward.
- **Uncapped control**, .105276/.267075/.268743, composite.213698; V.073348/.235234/.233469, composite.180684. Reject. It adds12 Healthy AirHitam low-tail palms to L OOF top10%; cap adds0 there. V Palong drops below incumbent. The score/rank overhead hypothesis is supported for this reconstruction.
- **Clip±3,C=.02 reconstruction**, .134924/.288375/.263817, composite.229039. Exactly matches documented s015 L; U tail remains weak (AirHitam rank~40th percentile). Missing s015 verified archive prevented byte-exact guard25 recovery; reconstructed guard is explicitly distinct.
- **Remove NDVI polynomials**, .160686/.294036/.349104, composite.267942; still AirHitam negative U stress correlation. Correlated optical/contextual suppressors matter. Heavy differential/nuisance shrinkage loses other-estate skill.
- **Protected core + block-honest pairwise ranking residual**, .033031/.248765/.187027, composite~.15627; physical/orthogonal controls also weak. Half full-model blend .13245/.30039/.30633, composite.24639, but weak tail safety. Reject before V.
- **Physical-only source pooling with same guard**, .146004/.309018/.366884, composite.273968; V.060659/.319079/.277015, composite.218918. Reject versus simpler selected. Removing all pooling also lower locally.
- **Orthogonal bounded annual-context residual**, .163789/.307359/.386264, composite.285804; V.064985/.336057/.276991, composite.226011. Five-repeat mean delta selected−.001696; no established improvement, do not add complexity.
- **DINO SAR change/short-change residuals**, .155381/.306329/.369810 and .155786/.308276/.379082; both weaker. These use carried label-free representations, not a repeat of s015's CNN.
- **Robustness**: selected beats incumbent in all estate cells under160m blocks,80m buffers,quadrants and80m-buffered quadrants. At160m buffers Serting loses; one fold has no training positives and a disclosed common fallback. Atquad160 Palong is near/no-skill. Do not claim universal far-field robustness.

## Dead ends

- Never use unpromoted s014 as the acceptance baseline or final fallback; only compare promotion against designated s001. A balanced tail-safe upgrade was achieved here.
- Uncapped/replacement guards can flood the head of the ranking. Scaling alone does not bound inspection overhead; upward-only capped corrections are preferable on current evidence.
- Removing quadratic terms alone does not prevent collinear context from reversing tails. Strong global shrinkage and protected global residuals sacrifice useful body ranking.
- Physical marginal sign restrictions do not guarantee better conditional source transfer. Texture SD/edge, green/red/GRVI and height have real cross-estate sign changes; equally trusted labels remain equally trusted.
- Respect MainChecklist and prior dead ends: no random CV, same-block label-neighbor leakage, Middle weak-positive target, generic SAR mean/model sweep, DRO/moment/temporalCNN/clustering reruns without new mechanism.
- Annual residual's tiny official gain does not survive repeats/V. Do not tune the remaining V budget on cap thresholds, ensemble weights or cosmetic hashes.

## Open leads, ranked

1. Prospective original-block inspection of protected low-NDVI crowns, s009/s014/selected disagreements and random controls in all estates. Distinguish disease, empty/young crowns, soil mixture and registration.
2. Independent estate/year validation and distant-pocket discovery. Current160m-buffered failures are an explicit limitation despite primary80m gains.
3. Better honest budget25 recall while preserving AP and tail safety; AirHitam44.6%, Palong drops from incumbent78.8% to65.2%. Use new labels or mechanistic evidence, not V micro-tuning.
4. Restore missing original compact/highres/crown extraction sources for full raw-imagery provenance. Packaged table/model refits are reproducible now; exact raw reconstruction of all carried consolidation sources is not established.

## How to resume

Read report/REPORT.md, selection.json, verifier/full_history.json and prior/MainChecklist.md. Immutable selected source candidates/budgetfloor25 matches submission and /shared/s016_budgetfloor25. Runtime uses only triage_v12,dino_sar,gee_quality; load via ctx datasets_dir. Source remains identical across estates. Preserve the protected nominee.

```
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
bash /app/research/reproduce.sh
```

`research/evaluate.py --models MODEL --protocols official block160 buffer80 quad quad80 --tails` evaluates frozen candidates. Add --force to refit existing cached predictions. Reference s009 retains legacy inner geometry and fails some thin160m-buffered nested folds; do not treat failed partial runs as complete metrics. New residual engines use union geometry and block-disjoint offsets. `make_candidates.py` and round2–round5.py contain experiment configs; frozen candidate sources, not the evolving research engine, are authoritative. The s015 recipes/verified archive and several advertised tables were absent at session start; recovery claims are explicitly limited. Literature web tool failed; direct primary-page/metadata fetches are archived. Plotting is research-only.

## Artifact index

- submission/pipeline.py,model_engine.py,requirements.txt,config.json: verified offline CPU nominee, hashes in import_audit.json/selection.json.
- report/REPORT.md,HANDOFF.md,selection.json: scientific decision, recipe, limitations and resume instructions.
- OFFICIAL_SUMMARY.csv,EXPERIMENT_SUMMARY.csv,EXPERIMENTS.csv: all46 official recipes/comparators and426 estate/protocol rows; per-estate/composite paired deltas vs incumbent and coords.
- verifier/v02_budgetfloor25.json,v03_uncapped_floor.json,v04_physical_pool.json,v05_annual_change.json,full_history.json; VERIFIER_SUMMARY.csv: actual noisy evidence, failed query included in history.
- frozen_repeat_plan.json,repeat_comparison.csv,repeat_summary.csv,bootstrap.csv,bootstrap_method.json: stability and conditional intervals.
- TAILS.csv,selected_tail_diagnostics.csv,tail_audit.txt,guard_rank_overhead.csv,marginal_signs.csv,parent_coefficients.csv: support/sign/guard mechanism diagnostics.
- inspection_diagnostics.csv,recall_budgets.csv,error_by_distance.csv,unlabelled_ranking.parquet/.csv: inspection tradeoffs and all4473 aligned U scores.
- selfcheck.txt,official_cv.txt,import_audit.json,dataset_audit.json,fresh_subprocess_contract.json: final contract checks.
- figures/: PNG/PDF AirHitam tail, inspection curves and verified mechanism plots; literature/: primary references and fetch log.
- research/evaluate.py,bootstrap.py,final_diagnostics.py,contract.py,summarize.py,plots.py,write_report.py,reproduce.sh; candidates/: frozen configs/helpers; research/results/: OOF scores.
- report/provenance/: carried non-table PCA/checksum sidecars relocated from datasets; dataset_layout_moves.json records moves. Feature table bytes unchanged. datasets/: six tables,manifests,extraction scripts only; no recipes or metrics archived there.
