# Oil-palm triage: balanced upgrade with protected AirHitam tails

Session `261127-s016-open`. Final selection: **budgetfloor25**, staged in `/app/submission`. The designated s001 incumbent is unchanged. All three estates' ground-truth labels are equally trusted.

## Decision and primary evidence

The selected model achieves noisy V AP lifts **+.072622 / +.336414 / +.279586**, composite **+.229541**, against incumbent **+.118751** and coordinates-only **+.066462**. Paired noisy deltas versus incumbent are **+.035595 / +.083108 / +.213668**, composite **+.110790**. Versus coordinates they are **+.032831 / +.208607 / +.247799**, composite **+.163079**. Every estate improves in the primary metric. AirHitam exceeds the requested approximate +.065 target, and the other two exceed +.30 and +.20. These are noisy point estimates, not prospective significance statements.

AirHitam U Spearman(score, −NDVI) changes from **−.132566 to +.0920**. Its 92 U palms below −2.5 training SD average approximately the **80.82nd risk percentile**, with **71/92** in the top quarter; the 57 below −3 SD average the **86.82nd percentile**, with **55/57** in the top quarter. The unprotected previous recipe put these groups near the 21st and 16th percentiles. This meets the requested ranking-safety condition, without claiming that their unknown disease outcomes are positive.

Official 80 m CV composite lift is **+.284247**, paired gains **+.161263** over incumbent and **+.240384** over coordinates. All estates improve versus incumbent in **all ten** prespecified repeated original-block assignments. Conditional block-bootstrap composite delta incumbent 95% interval is **[+.080731, +.230362]**. Palong's individual bootstrap interval crosses zero; equal-estate consistency is supported by repeats and V, not a claim of individual 95% significance.

The first verified capped model remains the nominee after testing shape/shrinkage, protected ranking residuals, restricted source transfer, and orthogonal annual/SAR-context changes. Later variants do not establish an improvement. The unprotected s014 pipeline was a diagnostic reference and was never a fallback nominee. Five V calls were used, including one failed missing-file staging call; ten remain. No cosmetic resampling was performed.

## Data, validation and reproducibility

L has 2,747 rows and 2,398 binary-scored observations: AirHitam 964 (56 unhealthy), Palong 770 (66), Serting 664 (67). Palong's 198 and Serting's 151 Middle labels are excluded from every disease target and binary metric. All held-out rows, including Middle, receive predictions. U has 4,473 rows, all returned in exact input order. No U outcomes or sealed split assignments were accessed.

For each estate, east=(Long−union_min_Long)×111320×cos(union_mean_Lat), north=(Lat−union_min_Lat)×110540, with origins anchored on authorized L∪U. Original cells are floor(east/80), floor(north/80); group=cell_east×100000+cell_north. Outer GroupKFold(5) uses the full L table before binary filtering. L occupies 26/18/13 original blocks. This reproduces the current CLI; historical L-only-origin metrics are not mixed with current results. The faithful s009 reference retains its legacy inner origin and inherited name branch only for comparison.

AP is sklearn's tie-aware step-function average precision; lift=AP−scored prevalence. Composite is the unweighted estate mean. Metrics pool held-out scores before computing AP. Recall budgets use fractional tie averaging. A fitted NDVI baseline in the CLI differs from the untrained raw −NDVI ranking reported here. Random-row CV and classification accuracy are not selection criteria.

The 160 m protocol joins original cells 2×2. Quadrants partition complete original cells at population-weighted cell-index medians. Buffers remove training trees within 80 or 160 m of held-out trees. Seeds 1201–1210 shuffle whole original groups, then assign five folds. Their values were recorded before repeat results; the experiment still reuses L after adaptive model exploration. Inner offsets for new residuals use the same union geometry and only block-disjoint core predictions.

All code, configs, metrics and OOF arrays are under `research`, `candidates` and `report`, outside datasets. `research/reproduce.sh` regenerates the principal analyses; use `--force` with evaluate.py to refit cached runs. Required runtime tables are **triage_v12, dino_sar and gee_quality**. Six actual carried feature tables total **32.14 MiB**, each with 7,220 exact keys, no duplicate/extra/missing keys, NaNs, nonfinite numeric values or forbidden column names. No symlinks exist.

The supplied s015 recipe archive, late_optical, temporal_sar, dino_optical and several advertised source tables were absent. `recon_clip3_C02` exactly reproduces the three documented L lifts; `recon_guard25` is explicitly a reconstruction, not the original V3. The s009 source package and previous s014 pipeline were present. The consolidated runtime table is sufficient for exact model refitting, but its original compact/high-resolution/crown source tables and extraction scripts are incomplete in this workspace; rebuilding those original covariates from raw imagery cannot be claimed byte-reproducible here. All available feature tables/scripts and provenance were preserved. Non-table PCA/checksum sidecars were moved to `report/provenance`, with moves recorded, to comply with the strict dataset layout. Feature bytes were unchanged.

## Baselines and paired 80 m results

Columns delta_incumbent_composite and delta_coords_composite are paired differences on identical held-out rows. CSVs also include every estate-specific paired difference.

| model | AirHitam | Palong | Serting | composite | delta_incumbent_composite | delta_coords_composite |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | 0.01469 | 0.21931 | 0.13496 | 0.12298 | 0.00000 | 0.07912 |
| coords | 0.01404 | 0.03932 | 0.07822 | 0.04386 | -0.07912 | 0.00000 |
| raw_ndvi | 0.02080 | 0.03942 | 0.09230 | 0.05084 | -0.07214 | 0.00698 |
| s009 | 0.03015 | 0.25267 | 0.19847 | 0.16043 | 0.03745 | 0.11657 |
| s014 | 0.15964 | 0.31152 | 0.39107 | 0.28741 | 0.16443 | 0.24355 |
| budgetfloor25 | 0.15769 | 0.31016 | 0.38489 | 0.28425 | 0.16126 | 0.24038 |

## Hypotheses and controlled experiments

**1. Calibrated, capped upward-only tail protection.** A direct fallback can overwrite already high full-model scores and combine incompatible score scales. We first quantile-calibrated a compact quality-vigor/spatial core to the full-model training score distribution, used an upward-only repair, then limited the fallback near the labelled model's upper-quartile score. The uncapped control has V composite +.180684, including Palong +.235234 (**−.018072** versus incumbent); the capped model reaches +.229541 with Palong +.336414. AirHitam is essentially unchanged at +.07335 versus +.07262, within noise. Thus preserving the body and limiting fallback magnitude is empirically more effective than forcing severe tails to the very top. This supports a calibration/ranking-overhead explanation for the reconstructed fallback loss, not a claim to have recovered the exact missing s015 guard implementation.

On L OOF AirHitam, the uncapped control adds **12 Healthy low-NDVI palms to the top 10%**; the capped model adds **zero** at that budget. Both protect more of that group at 25%, costing some AirHitam recall. On U, AirHitam low-tail trees in the top 10% are 69 uncapped versus 20 capped. Unknown U labels prevent interpreting those as false positives. `guard_rank_overhead.csv` records all estates and inclusive Middle counts. Cap alternatives and transition widths were explored on L/U only; no V cap/threshold micro-grid was run.

**2. Remove extrapolation and shrink contextual suppressors.** Dropping NDVI² and NDVI×coordinate terms alone does not repair AirHitam: U correlation remains negative, and severe-tail ranks remain low. Linear correlated optical/contextual terms can undo physical vigor effects. Clip ±3/C=.02 reproduces documented L gains but AirHitam's tail still averages only about the 40th percentile. Strong nuisance penalties, context shrinkage, physical-only features, and differential physical/context regularization cost substantial Palong/Serting performance. A bounded guard avoids refitting the useful body merely to control tails.

**3. Protected monotone low-dimensional core plus honest ranking residual.** We fit a sign-bounded quality-vigor coefficient with only linear vigor and spatial quadratic terms, froze that core, and trained contextual residuals using original-block OOF offsets. A pairwise logistic ranking loss emphasizes high-core-risk negatives; offset-logistic, physical-only, orthogonalized and damped controls isolate the mechanism. Residuals are capped relative to core scale. Pure protected models have composite L only about .13–.16; a half full-model blend reaches .24639 but insufficiently protects the U tail. These are different from in-sample residual fits and do not leak same-block labels. None was sufficiently competitive to spend V.

**4. Transfer only sign-consistent vegetation indices.** We held the capped guard and target-local context fixed and changed source-prior pooling to only qNDVI, qNDRE, NDVI_contrast35, NDRE1_contrast35, ndvi_dip and ndre_contrast. V becomes +.060659/+.319079/+.277015, composite +.218918. Removing all pooling is also weaker on L. This clean ablation separates source-sign restrictions from heavy shrinkage. Although all estates are equally trusted, marginal sign agreement does not guarantee useful conditional transfer restrictions; no estate-name branch is introduced.

**5. Bounded independent context-change refinements.** New research recipes use already-carried covariates, without new extraction: 2023→2024 and 2024→2025 AlphaEarth point/contrast deltas, or June/October DINO SAR change/short-footprint differences. Eight training-only PCs are residualized against static context without labels. Original-block OOF checkpoint scores serve as fixed offsets; additions are strongly regularized and bounded to ±.25 final score units. Point annual change raises official AirHitam L by .00610 and 80 m-buffer AirHitam by .00986, but five-repeat mean composite delta is **−.00170**. V +.226011 is a noise-scale tie below the simpler checkpoint, with lower AirHitam point estimate. SAR change variants lose local performance. This is an incremental residual mechanism over carried representations, not a rerun of s015's temporal-SAR CNN. No new GPU training was needed; the broker confirms no active VM and zero session GPU usage.

Representative experimental rows follow; all **46** official recipe/comparator records and all **426** estate/protocol records are in `OFFICIAL_SUMMARY.csv`, `EXPERIMENT_SUMMARY.csv` and `EXPERIMENTS.csv`. Source configs are frozen under `candidates/`.

| model | AirHitam | Palong | Serting | composite | delta_incumbent_composite | delta_coords_composite |
| --- | --- | --- | --- | --- | --- | --- |
| recon_clip3_C02 | 0.13492 | 0.28838 | 0.26382 | 0.22904 | 0.10605 | 0.18518 |
| recon_guard25 | 0.10379 | 0.27644 | 0.25448 | 0.21157 | 0.08859 | 0.16771 |
| floor25 | 0.10528 | 0.26708 | 0.26874 | 0.21370 | 0.09071 | 0.16984 |
| budgetfloor25 | 0.15769 | 0.31016 | 0.38489 | 0.28425 | 0.16126 | 0.24038 |
| budgetfloor20 | 0.15731 | 0.31244 | 0.38488 | 0.28488 | 0.16189 | 0.24102 |
| budgetfloor65 | 0.15792 | 0.31028 | 0.38628 | 0.28483 | 0.16184 | 0.24097 |
| no_ndvi_poly | 0.16069 | 0.29404 | 0.34910 | 0.26794 | 0.14496 | 0.22408 |
| shrink_context | 0.06484 | 0.26379 | 0.26018 | 0.19627 | 0.07329 | 0.15241 |
| shrink_context_C05 | 0.04759 | 0.24306 | 0.21353 | 0.16806 | 0.04507 | 0.12419 |
| no_pool_physical | 0.06986 | 0.24859 | 0.29859 | 0.20568 | 0.08269 | 0.16182 |
| rankresidual1 | 0.06645 | 0.26627 | 0.28897 | 0.20723 | 0.08425 | 0.16337 |
| rankresidual2 | 0.15116 | 0.30733 | 0.34489 | 0.26779 | 0.14481 | 0.22393 |
| blend_core50 | 0.09218 | 0.27127 | 0.26420 | 0.20922 | 0.08623 | 0.16535 |
| protected_pair | 0.03303 | 0.24877 | 0.18703 | 0.15627 | 0.03329 | 0.11241 |
| protected_pair_phys | 0.02253 | 0.22606 | 0.17454 | 0.14104 | 0.01806 | 0.09718 |
| protected_pair_orth | 0.03885 | 0.23773 | 0.17424 | 0.15027 | 0.02729 | 0.10641 |
| protected_pair_mix | 0.13245 | 0.30039 | 0.30633 | 0.24639 | 0.12340 | 0.20253 |
| guard_no_pool | 0.14402 | 0.31047 | 0.36399 | 0.27282 | 0.14984 | 0.22896 |
| guard_physical_pool | 0.14600 | 0.30902 | 0.36688 | 0.27397 | 0.15098 | 0.23011 |
| guard_differential | 0.13003 | 0.23351 | 0.27932 | 0.21428 | 0.09130 | 0.17042 |
| guard_physical_shrink | 0.12765 | 0.26860 | 0.31254 | 0.23626 | 0.11328 | 0.19240 |
| annual_change | 0.16379 | 0.30736 | 0.38626 | 0.28580 | 0.16282 | 0.24194 |
| annual_contrast_change | 0.16081 | 0.30435 | 0.38272 | 0.28263 | 0.15964 | 0.23876 |
| dino_change_resid | 0.15538 | 0.30633 | 0.36981 | 0.27717 | 0.15419 | 0.23331 |
| dino_short_change | 0.15579 | 0.30828 | 0.37908 | 0.28105 | 0.15806 | 0.23719 |
| annual_joint_change | 0.16353 | 0.30817 | 0.38388 | 0.28519 | 0.16221 | 0.24133 |

## In-run V evidence

Cached additive Gaussian noise has σ=.005. Small differences among the capped checkpoint and annual residual do not establish improvements. V was used adaptively after local testing, so these are neither independent trials nor calibrated confidence intervals. The failed first query returned no score and is retained in the history.

| model | AirHitam | Palong | Serting | composite |
| --- | --- | --- | --- | --- |
| v02_budgetfloor25 | 0.07262 | 0.33641 | 0.27959 | 0.22954 |
| v03_uncapped_floor | 0.07335 | 0.23523 | 0.23347 | 0.18068 |
| v04_physical_pool | 0.06066 | 0.31908 | 0.27702 | 0.21892 |
| v05_annual_change | 0.06498 | 0.33606 | 0.27699 | 0.22601 |

`verifier/full_history.json` records all five calls and hashes; the four successful per-estate results are in separate JSONs and `VERIFIER_SUMMARY.csv`. Submission source/config/requirements remain byte-identical to the successful query-2 stage. Changes to dataset manifests/layout did not change feature bytes or predictions.

## Stability, uncertainty and far-field limits

Repeated whole-group results versus incumbent:

| estate | assignments | mean_lift | mean_delta | min_delta | positive_deltas |
| --- | --- | --- | --- | --- | --- |
| AirHitam | 10 | 0.11983 | 0.09587 | 0.07604 | 10 |
| Palong | 10 | 0.30124 | 0.05709 | 0.01927 | 10 |
| Serting | 10 | 0.34384 | 0.20889 | 0.15430 | 10 |
| composite | 10 | 0.25497 | 0.12062 | 0.09249 | 10 |

Fixed-OOF paired block bootstrap, 4,000 replicates, seed 161127:

| reference | estate | observed_delta | lower95 | upper95 | fraction_positive |
| --- | --- | --- | --- | --- | --- |
| incumbent | AirHitam | 0.14301 | 0.02730 | 0.27096 | 0.99925 |
| incumbent | Palong | 0.09085 | -0.08606 | 0.25599 | 0.82975 |
| incumbent | Serting | 0.24993 | 0.13627 | 0.30757 | 1.00000 |
| incumbent | composite | 0.16126 | 0.08073 | 0.23036 | 1.00000 |
| coords | AirHitam | 0.14365 | 0.01693 | 0.28016 | 0.99375 |
| coords | Palong | 0.27084 | 0.13941 | 0.39294 | 1.00000 |
| coords | Serting | 0.30666 | 0.13566 | 0.36163 | 0.99975 |
| coords | composite | 0.24038 | 0.15623 | 0.29711 | 1.00000 |

The bootstrap samples original blocks independently within estates and reuses identical count weights across model pairs. It includes held-out block sampling variation, **not refitting, feature/model selection, V adaptation or independent-estate uncertainty**. Palong's interval is wide and crosses zero despite 10/10 repeated assignment wins and positive V gain. The composite evidence is consistent, but does not establish pathogen causality or out-of-estate transportability.

| model | protocol | AirHitam | Palong | Serting | composite |
| --- | --- | --- | --- | --- | --- |
| budgetfloor25 | block160 | 0.11790 | 0.33134 | 0.35863 | 0.26929 |
| coords | block160 | -0.00478 | -0.00283 | 0.07531 | 0.02257 |
| incumbent | block160 | 0.02763 | 0.28175 | 0.10064 | 0.13668 |
| budgetfloor25 | buffer160 | 0.13246 | 0.04967 | 0.03898 | 0.07370 |
| coords | buffer160 | 0.04045 | -0.00604 | 0.02748 | 0.02063 |
| incumbent | buffer160 | 0.00887 | -0.01794 | 0.09658 | 0.02917 |
| budgetfloor25 | buffer80 | 0.10289 | 0.25965 | 0.17540 | 0.17931 |
| coords | buffer80 | -0.00821 | -0.00186 | 0.02739 | 0.00577 |
| incumbent | buffer80 | 0.00608 | 0.14582 | 0.05116 | 0.06769 |
| budgetfloor25 | quad | 0.11081 | 0.32993 | 0.35352 | 0.26475 |
| coords | quad | -0.00709 | 0.00292 | 0.07798 | 0.02460 |
| incumbent | quad | 0.03333 | 0.22625 | 0.11397 | 0.12452 |
| budgetfloor25 | quad160 | 0.04497 | -0.01420 | 0.04271 | 0.02449 |
| coords | quad160 | 0.03025 | 0.01627 | -0.01092 | 0.01187 |
| incumbent | quad160 | 0.02681 | -0.01628 | -0.00795 | 0.00086 |
| budgetfloor25 | quad80 | 0.10371 | 0.18354 | 0.24235 | 0.17653 |
| coords | quad80 | 0.01350 | 0.00024 | 0.02556 | 0.01310 |
| incumbent | quad80 | 0.02539 | 0.17781 | 0.01236 | 0.07186 |

At 80 m buffers and unbuffered/80 m-buffered quadrants, the selected model improves all estates over incumbent. DINO's October patch is approximately 103 m wide, so adjacent 80 m blocks can share image support. A 160 m tree separation exceeds the diagonal overlap bound for these square patches and provides a stronger check. **At buffer160, Serting loses to incumbent** (+.03898 versus +.09658), and one fold has no training positives. For that diagnostic only, the evaluator uses a disclosed common zero-score fallback for all models in degenerate folds; official CV has none. At quad160, Palong is near/no-skill (−.01420). These expose limited distant-region transportability rather than an all-protocol victory. The faithful s009 nested routine fails some very thin 160 m-buffered training folds because fewer than five groups remain; those runs are not reported as valid complete comparisons. The selected model's primary gain is not pure coordinate smoothing, but shared image context and within-estate environmental structure remain contributors. Global optical crown-mixture inversion is transductive and can couple wider observations; even the160m buffer does not prove independence of every derived covariate.

## Feature signs, AirHitam support and operational interpretation

Cohen's d is Unhealthy minus Healthy using pooled SD, descriptive on all binary L. It is not a causal coefficient or another held-out test.

| feature | AirHitam | Palong | Serting |
| --- | --- | --- | --- |
| NDVI | -0.19502 | -0.31057 | -0.53012 |
| qNDVI | -0.19503 | -0.35568 | -0.68702 |
| qNDRE | -0.43171 | -0.60424 | -0.80618 |
| NDVI_contrast35 | -0.18200 | -0.54288 | -0.47732 |
| NDRE1_contrast35 | -0.48468 | -0.77823 | -0.70058 |
| ndvi_dip | 0.08194 | 0.54894 | 0.69822 |
| ndvi_sd20 | -0.25346 | 0.06498 | 0.12258 |
| ndvi_edge | -0.24991 | 0.05790 | 0.11729 |
| height_mean6 | 0.30667 | -0.55824 | -0.25639 |
| log_green_red | 0.57308 | -0.14053 | -0.39028 |
| deblur_GRVI | 0.64350 | -0.22668 | -0.37084 |

NDVI/red-edge and local vigor contrasts have consistent stress directions. Texture SD/edge, historical height, green/red and deblurred GRVI flip signs across estates. Conditional coefficients from a collinear joint fit need not match these marginal signs. `parent_coefficients.csv` records the unchanged full-model coefficients; the final guard is a separate nonlinear score correction. Opaque independently fitted PCA component signs are not compared as physiological effects.

AirHitam support audit confirms 16 L palms below −3 raw-NDVI SD, **all Healthy**; below −2.5 SD there are 19 L palms with two positives. Only two U palms are below the absolute L NDVI minimum, so density shift is broader than simple out-of-range extrapolation. Empty/young crowns, soil mixture, stand structure and registration are unresolved explanations. Guarding these crowns is a requested operational precaution, not proof that the unprotected model's U predictions were wrong.

AirHitam U tail comparison (rank columns are fractions):

| model | spearman | rank2.5 | top25_2.5 | rank3 | top25_3 |
| --- | --- | --- | --- | --- | --- |
| s014 | -0.13257 | 0.20929 | 1 | 0.15960 | 1 |
| s009 | 0.52917 | 0.82734 | 63 | 0.86052 | 43 |
| incumbent | 0.51122 | 0.85044 | 72 | 0.88989 | 46 |
| budgetfloor25 | 0.09200 | 0.80818 | 71 | 0.86822 | 55 |

The final recipe is **not globally monotone** in all correlated optical observations. It explicitly lifts low-raw-NDVI rankings through a compact calibrated core while retaining the high-dimensional body. Its quality-vigor core coefficient is negative in all three full-data fits. This is an observed safety result on the supplied U population, not a universal causal monotonicity theorem.

## Inspection yield and error analysis

Primary AP lift improves every estate, but inspection metrics show material tradeoffs:

| estate | recall25 | inclusive_recall25 | enrichment10 | roc | precision25 | false_positives_per_detection |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | 0.44643 | 0.44643 | 3.03571 | 0.69596 | 0.10373 | 8.64000 |
| Palong | 0.65152 | 0.62121 | 4.84848 | 0.80619 | 0.22338 | 3.47674 |
| Serting | 0.74627 | 0.71642 | 5.37313 | 0.85925 | 0.30120 | 2.32000 |

At top 25%, incumbent recall is 30.4%/78.8%/61.2%; selected is **44.6%/65.2%/74.6%**. Palong's top-25 recall drops even while top-10 enrichment and AP improve. AirHitam still fails majority capture at25%; at30% inspection its L OOF recall is52.1%, and at35% it is60.7%. These are capacity tradeoffs, not deployment guarantees. Middle consumes inspection capacity, reducing selected inclusive recall to **44.6%/62.1%/71.6%**. These are L OOF estimates, not V/U yield estimates. Precision and false-positive overhead are defined only on scored binary rows.

`recall_budgets.csv` and exported PNG/PDF inspection curves show 10–50% capacity tradeoffs. `error_by_distance.csv` records each held-out positive's distance to training positives using only training labels. Small distant-positive samples and weak heavily buffered results remain limitations. All **4,473 U scores** are exported in `unlabelled_ranking.parquet`/CSV with exact keys, percentiles and an illustrative ceil(25%×estate size) inspection flag. No U labels, disease probabilities or U precision claims are included. The downstream polarized-imagery system is out of scope.

## Exact offline recipe and execution contract

The full-model body uses raw NDVI/local coordinates; nine polynomial outputs (three linear and six quadratic/interaction terms), with the latter weighted .3; twelve optical terms ×.5; eight training-only AlphaEarth2025 local-contrast PCs ×.5; three historical height summaries ×.35 with denominator floors 3/1/1 m; nine deblur indices ×.3; eight October103 DINO SAR PCs ×.3; intercept10. C=.15, balanced logistic loss, with equal-weight source-estate optical coefficients as the ridge-prior centre. Contextual and spatial responses stay target-local. All estate names receive exactly the same recipe.

The protected core fits balanced C=.1 logistic regression to the mean of raw NDVI and quality NDVI p75 plus linear/quadratic spatial terms; it excludes vigor² and vigor×coordinate terms. Let f be full-model training scores, s target full scores, b core training scores and t target core scores. Let q(t) be the empirical midrank CDF of b. The guard computes:

```
a = quantile(f, q(t))
cap = quantile(f, .75) + .1 * std(f) * (t - median(b)) / std(b)
desired = min(a, cap)
z = (target raw NDVI - training mean NDVI) / training SD NDVI
w = clip((-z - 2.5 + .35) / .70, 0, 1)
final = s + w * max(desired - s, 0)
```

Thus the full score is never overwritten downward, the core/full scales are matched, and the correction turns on smoothly over z=−2.15 to −2.85. The cap is a **training-score reference near the inspection budget**, not a guarantee of a fixed U percentile or inspection count. No labels from target rows enter this calculation. Training-only transforms are fitted separately in every CV call; there are no target label-neighbor features, Middle targets, pseudo-labels, inference-time network calls or raster accesses.

`agentrsi selfcheck` and final `agentrsi cv` pass. Non-base imports are exactly **threadpoolctl==3.8.0**; all others are standard library or task-base packages. Fresh four-CPU workers use ≤**0.206 s** for the full estate prediction and ≤**277.7 MiB** peak RSS in this container. Outputs match the verified checkpoint exactly, preserve permuted/duplicated target alignment within 4.5e−16, are finite float64, handle empty target inputs, and do not mutate inputs or write scratch files. Source inspection and the isolated verifier support offline execution; `fresh_subprocess_contract.json` records measured checks. Plotting dependencies are research-only and absent from submission requirements.

## Primary literature and provenance

- Roberts et al. (2017), [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881): supports validation that respects dependence. Publisher full text returned403; publisher-supplied Crossref metadata was retrieved. No inaccessible detailed result is asserted.
- Liaghat et al. (2014), [Early detection of basal stem rot disease in oil palms based on hyperspectral reflectance data using pattern recognition algorithms](https://doi.org/10.1080/01431161.2014.903353): optical disease-screening motivation. Hyperspectral designs do not establish Sentinel-2 per-crown diagnostic specificity; only metadata were retrieved here.
- [Sentinel-2 SR Harmonized catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) and [Cloud Score+ catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED): harmonized reflectance and quality provenance. No second baseline04.00 offset is applied.
- Brown et al. (2025), [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291), and [official annual-embedding catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL): provenance of annual geospatial context, not current disease confirmation. Carried manifests record actual queried year availability.
- Oquab et al. (2023), [DINOv2](https://arxiv.org/abs/2304.07193): frozen RGB encoder provenance. Applying it to SAR display channels is an out-of-domain representation; no pathogen specificity is inferred.
- [sklearn average_precision_score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): exact step-function AP definition, distinct from trapezoidal PR area.

The web tool failed with a service500; direct HTTPS requests to the primary pages/metadata succeeded except the restricted publisher full text. Responses and fetch status are archived under `report/literature`. This session adds an empirically tested score-safety policy, not a literature claim that every low-NDVI crown is diseased.

## Artifacts and next evidence

`submission/pipeline.py`, `model_engine.py`, `requirements.txt`, `config.json`: byte-identical verified nominee. `selection.json` records hashes and query. `OFFICIAL_SUMMARY.csv`, `EXPERIMENT_SUMMARY.csv`, `EXPERIMENTS.csv`: actual paired experiments. `verifier/` and `VERIFIER_SUMMARY.csv`: genuine call history. `repeat_summary.csv`, `repeat_comparison.csv`, `bootstrap.csv`: stability and conditional uncertainty. `TAILS.csv`, `tail_audit.txt`, `guard_rank_overhead.csv`, `marginal_signs.csv`, `parent_coefficients.csv`: support, mechanism and signs. `inspection_diagnostics.csv`, `recall_budgets.csv`, `error_by_distance.csv`, `unlabelled_ranking.*`: triage/error outputs. `selfcheck.txt`, `official_cv.txt`, `import_audit.json`, `dataset_audit.json`, `fresh_subprocess_contract.json`: execution evidence. `figures/`: standalone PNG/PDF plots. `research/` and `candidates/`: reproducible engines/configs/cached OOF predictions; no research archive is placed in datasets.

The next high-value evidence is prospective block inspection of AirHitam's protected low-NDVI crowns and model disagreements with random controls, and independent estate/year validation. The current model can be shipped as a balanced ranking upgrade over s001 while explicitly retaining uncertainty about low-vigor disease specificity and distant-region screening yield.
