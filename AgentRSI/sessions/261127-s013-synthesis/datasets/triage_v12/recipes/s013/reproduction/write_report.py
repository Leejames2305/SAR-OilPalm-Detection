"""Generate the report and six-section handoff from frozen empirical artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path('/app');R=ROOT/'report';df=pd.read_csv(R/'EXPERIMENTS.csv');summary=pd.read_csv(R/'EXPERIMENT_SUMMARY.csv');estates=['AirHitam','Palong','Serting'];parents=['incumbent','coords','s009','s010','s011','s012','s012_bag'];frozen=[f'repeat{i}' for i in range(401,421)]
def md(frame):
    columns=list(frame.columns);lines=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        vals=[]
        for value in row:
            if pd.isna(value):s='NA'
            elif isinstance(value,(float,np.floating)):s=f'{value:+.6f}'
            else:s=str(value)
            vals.append(s)
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

official=df[df.protocol=='official'].pivot(index='model',columns='estate',values='lift');official['Composite']=official.mean(axis=1)
official['Delta incumbent']=official.Composite-official.loc['incumbent','Composite'];official['Delta coords']=official.Composite-official.loc['coords','Composite']
history={'incumbent':(.11875051189920606,[.03702768808911442,.2533059847191584,.06591786288934534]),'coords':(.06646184597484037,[.0397917985858249,.1278070936422356,.03178664569646067]),'s009':(.176790,[.065513,.276497,.188360]),'s010':(.216825,[.054553,.289959,.305963]),'s011':(.182472,[.037031,.338494,.171891]),'s012':(.203411,[.049552,.363177,.197506]),'s012_bag':(.213993,[.049482,.366903,.225594])}
vr=[]
for p in sorted((R/'verifier').glob('v[1-8]_*.json')):
    d=json.loads(p.read_text());name=p.stem.split('_',1)[1];vals=[d['estates'][e]['noisy_pr_auc_lift'] for e in estates];comp=d['composite']['noisy_pr_auc_lift'];history[name]=(comp,vals)
    vr.append(dict(Query=p.stem.split('_')[0],Recipe=name,AirHitam=vals[0],Palong=vals[1],Serting=vals[2],Composite=comp,Delta_incumbent=d['composite']['noisy_delta_vs_incumbent'],Delta_coords=d['composite']['noisy_delta_vs_coords']))
pd.DataFrame(vr).to_csv(R/'VERIFIER_SUMMARY.csv',index=False)
parentrows=[]
for model in parents+['pooled_sar']:
    h=history[model];row=dict(Model=model)
    for i,e in enumerate(estates):row['L '+e]=official.loc[model,e];row['V '+e]=h[1][i]
    row['L composite']=official.loc[model,'Composite'];row['V composite']=h[0];parentrows.append(row)
parent_table=md(pd.DataFrame(parentrows))
allofficial=official.copy();allofficial['V composite']=[history.get(m,(np.nan,[]))[0] for m in allofficial.index];allofficial=allofficial.reset_index().rename(columns={'model':'Recipe'})
official_table=md(allofficial)
final=df[(df.model=='pooled_sar')&df.protocol.isin(frozen)].copy()
repeat=final.groupby('estate').agg(Lift=('lift','mean'),Delta_incumbent=('delta_incumbent','mean'),Worst_delta_incumbent=('delta_incumbent','min'),Delta_coords=('delta_coords','mean'),Delta_s010=('delta_s010','mean'),Delta_s012=('delta_s012','mean'),s012_wins=('delta_s012',lambda v:int((v>0).sum())),assignments=('lift','size')).reindex(estates).reset_index()
repeat.to_csv(R/'final_frozen_confirmation.csv',index=False)
comp=final.groupby('protocol')[['delta_incumbent','delta_coords','delta_s010','delta_s012']].mean()
spatial_names=['official','shift40','block120','block160','buffer40','buffer80','blockquad','blockquad80','blockquad160','quadrant','regions80','medianquad','median80']
sp=df[(df.model=='pooled_sar')&df.protocol.isin(spatial_names)].pivot(index='protocol',columns='estate',values='lift').reindex(spatial_names)
sp['Composite']=sp.mean(axis=1)
paired=summary[summary.model=='pooled_sar'].set_index('protocol')
for c in ['delta_incumbent','delta_coords','delta_s010','delta_s012']:sp[c]=paired[c]
spatial_table=md(sp.reset_index())
bud=pd.read_csv(R/'inspection_budget_diagnostics.csv');bud25=bud[bud.budget==.25]
triage=df[(df.model=='pooled_sar')&(df.protocol=='official')][['estate','prevalence','ap','lift','roc','recall25','enrich10']]
boot=pd.read_csv(R/'bootstrap.csv');bootcomp=boot[boot.estate=='composite']
diag=json.loads((R/'dataset_audit.json').read_text());contract=json.loads((R/'contract_verification.json').read_text());equiv=json.loads((R/'package_equivalence_final.json').read_text());sha=hashlib.sha256((ROOT/'submission/pipeline.py').read_bytes()).hexdigest()
sign=pd.read_csv(R/'feature_sign_consistency.csv');sign=sign[sign.feature.isin(['NDVI','qNDVI','qNDRE','qNDMI','ndvi_dip','ndvi_sd20','ndvi_edge','deblur_NDRE','height_mean6','height_sd6','height_contrast6'])]
disagree=pd.read_csv(R/'prediction_disagreement.csv');disagree=disagree[((disagree['first']=='s010')&(disagree.second=='s012'))|((disagree['first']=='s009')&(disagree.second=='s012'))]
ablations=['pooled_sar','uniform_sar','pool_fullopt','pooled_sar_self','pooled_sar_permuted','pooled_sar_nocoords','pooled_sar_noalpha','pooled_sar_no_compactopt','pooled_sar_nostructure']
abl=official.reindex(ablations).reset_index().rename(columns={'model':'Recipe'})
frozen_all=summary[summary.protocol.isin(frozen)&summary.model.isin(parents+['pooled_sar'])].groupby('model').agg(Composite=('lift','mean'),Delta_incumbent=('delta_incumbent','mean'),Delta_coords=('delta_coords','mean'),Assignments=('protocol','size')).reset_index()
readme=f'''# Oil-palm disease triage — cross-session synthesis 261127-s013

## Outcome and scope

The final submission is **pooled_sar**, a single offline CPU recipe combining s011/s012's uniform compact condition/context design with s010's shared-optical/local-deviation architecture and October SAR patch representation. All three estates' labels are equally trusted and estates have equal evaluation weight. No estate-name capacity branch, disease-label neighbor, pseudo-label, or supervised Middle target is used.

Official 80 m composite AP lift is **{official.loc['pooled_sar','Composite']:+.6f}**, versus incumbent {official.loc['incumbent','Composite']:+.6f}, coordinates {official.loc['coords','Composite']:+.6f}, s010 {official.loc['s010','Composite']:+.6f}, and s012 {official.loc['s012','Composite']:+.6f}. Noisy V composite is **+.220382**, delta incumbent **+.101632**, delta coordinates **+.153921**. All three V estate deltas versus both required comparators are positive. These are ranking results, not pathogen diagnosis, calibrated disease probabilities, or prospective field recall.

The best reference V composite is s010 +.216825: the new +.003557 difference is below the .005 single-observation noise scale and is **not evidence of a confirmed V win over s010**. The synthesis trades Palong V against s012 (−.024857) and Serting V against s010 (−.046352). It does not dominate every parent in every estate.

**Selection-policy disclosure:** the provisional exploration plan included a stricter .015 maximum V estate loss relative to the s012 checkpoint. V8 fails that guard in Palong. The final choice explicitly changes that provisional rule after V, choosing the task's incumbent-relative scientific criteria and positive mean L increments in every estate over s012. This is a V-informed, exploratory decision, not a passed preregistered incremental-parent gate. The conservative s012 package remains byte-preserved under datasets/triage_v12/recipes/s013/parents/s012. No ensemble weights or estate gates were fitted to V, and no further V calls followed V8.

At a top-25% binary budget, recall is 48.2% / 74.2% / 76.1%. Counting Middle against capacity gives 48.2% / 71.2% / 74.6%. AirHitam's majority-capture objective remains unmet at this budget. A distant buffered Palong stress test remains weaker than incumbent. These limitations constrain deployment claims despite the large composite gain.

## Data, provenance and validation

L has 2,747 rows: 2,398 binary-scored palms (189 positives) and 349 Middle. Estate binary counts/positives are 964/56, 770/66, 664/67. U has 4,473 rows; every row is returned in exact input order. Middle is excluded from binary training and AP, and included only in capacity diagnostics. Other estates' L labels may enter through ctx.other_labeled, as authorized; target held-out labels never enter training.

Only **10** of the 18 advertised carried feature tables were actually staged. All dependencies of every staged submission are present. Missing advertised assets were autumn, gee_s2_stress, optical_aligned, optical_texture, planting_geometry, sar_native, sar_structure and terrain_canopy. They were not silently reconstructed or re-extracted. No new raster/GEE/GPU feature acquisition was necessary. The original source tables, raster data, prior files and credentials are never accessed by fit_predict.

Official folds exactly match agentrsi cv: local east=(Long−minLong)×111320×cos(meanLat), north=(Lat−minLat)×110540; block=floor(east/80)×100000+floor(north/80). There are50/39/13 occupied L blocks in AirHitam/Palong/Serting, respectively, and each contains binary-scored rows. GroupKFold(5) allocates the entire L table including Middle, followed by binary metric filtering. All supervised scaling, PCA and fitting use current training rows only. The required coordinates baseline is distance-weighted 15NN on training-standardized Long/Lat, fitted to binary labels. AP is sklearn's tie-aware step-function average_precision_score; lift subtracts scored prevalence, then estates receive an unweighted mean. Fractional boundary inclusion handles inspection-score ties.

Exploration reallocates complete 80 m blocks with seeds21–25. First confirmation uses seeds101–110. **Frozen final confirmation uses seeds401–420**, after V8 selection and without further model changes. Repeats reuse the same labels/imagery; they do not create independent estates or trials. The full table contains {df.model.nunique()} models and {len(df)} completed estate/protocol evaluations. Failed runs are not replaced with optimistic scores or partial-estate composites.

Wider protocols: 120/160 m blocks; shifted80 m grid (+40m); 40/80 m nearest-held-out-tree exclusion buffers; population-weighted whole-block quadrants (blockquad), with80/160 m buffers. The older median-coordinate quadrants reproduce prior results but split some original80 m blocks and are secondary stress diagnostics. The unweighted occupied-block-median quadrant (quadrant/regions80) emphasizes sparse tails and can hold out590/815 Serting rows, leaving only8 training positives (7 with80 m buffer). It is a harsh sample-size/extrapolation test, not a balanced region estimate. All fold sizes, positive counts and distances are in spatial_split_audit.csv.

The final SAR view is approximately103 m square, so80 m blocking or buffering does not guarantee independent raw patches. Whole-block160 m region buffers provide a stronger direct-overlap check for the final SAR crops; scores are positive but lower. The optical inverse has a weak global influence tail, so even that buffer does not establish complete image independence or disease causality. No random-row validation or same-block label stacking was used.

## Unified cross-lineage audit

L values below are new paired reruns. V values for reference sessions are the **supplied verifier-verified reward table**, not substituted report maxima. Their reports contain slightly different noisy observations (s010 +.219732, s012 +.205755, etc.); those are historical observations, not repeated evidence here. Incumbent/coords V references are the scorer-returned baseline values. The preserved s012 bag V is the previous handoff's observation. We did not spend queries re-scoring unchanged parents.

{parent_table}

Mechanistic divergence is reproducible in L, but L does not reliably predict V estate ordering. s010's Serting L lift is lower than s012, including repeated assignments, while its V lift is about.108 higher. Palong L is nearly tied between those parents, yet s012 V is about.073 higher. AirHitam s010 L advantage shrinks sharply in V. These patterns support conditional/geographical instability and small-sample uncertainty; they do not prove a specific cause or justify questioning any estate's labels. Even stronger contiguous-region performance does not guarantee the V ordering.

Parent prediction disagreement is substantial enough to motivate controlled crossover, but does not itself establish an ensemble gain:

{md(disagree)}

top25_disagreement is the fraction of binary rows with different fractional selection; Jaccard uses fractional tie membership. These are official OOF predictions from the same folds.

## Controlled mechanisms and attribution

1. **s011/s012 uniform capacity:** replace s010's inherited estate branch with quadratic terms shrunk by.3, holding all other s010 features/pooling fixed (s010_curvature). L composite rises from+.255805 to+.269130; V+.214764 remains close to the parent reward. This removes a name branch without explaining the parent V divergence.
2. **s010 patch families into s012:** October SAR, optical NRG160, and both were added jointly or as s009-style block-OOF bounded residuals. Joint SAR improves repeated Serting most, while bounded offsets change rankings less. V1 SAR+.208825 and V2 offsets+.201727 show that L gains alone did not preserve Palong V. The final retains only October SAR; optical NRG160 did not earn its extra size/footprint.
3. **s010 pooling into s012:** shared physical optical coefficients plus independent estate deviations give pool_fullopt+.279816 official. Shared-only4-feature controls were weaker. Self-duplicate and source-label-permutation controls show that part of the gain is changed regularization, not demonstrated transferred physiology. The final pooled_sar combines that architecture with local SAR PCs; a common aligned SAR basis shared across estates was rejected without V because fresh repeat Palong deteriorated.
4. **Stability/bagging:** the preserved s01232-block probability ensemble and new SAR/both-patch bags were tested under the same protocols. SAR bag gives+.283051 official and positive repeat means in all estates, but V+.206329 does not improve the protected checkpoint balance. The final is one joint logistic fit, with no score averaging.
5. **Feature-gap crossover:** s010_curvature omits the compact moisture/quality, physical-floor height and mixture measurements. Adding them locally, jointly or bounded around the frozen parent, raises L Palong (legacy_complement+.282713 composite) but V6/V7 remain about+.214 and preserve the older Palong weakness. This is direct evidence that simply adding the missing family does not reproduce s012's V gain.
6. **Collinearity and pruning:** raw-vs-standardized AlphaEarth PCA, quality-NDVI blending, quality-family deletion and deblur redundancy pruning were controlled. Small local gains were mixed; no dimension, sensor-weight or V-feature sweep followed. The final uses unchanged s012 optical/height/mixture families and PCA counts, with the s010 architectural crossover.

Final ablations are descriptive after freezing, not a new selection step:

{md(abl)}

No-coordinates still retains imagery that can encode geography, so it is not a pure disease-signature test. Its large Palong loss establishes dependence on spatial context. Real source labels, self-duplication and one fixed permutation distinguish source information from effective penalty changes; one permutation is not a permutation-test p-value.

**Adopted:** uniform differential ridge and compact optical/AlphaEarth from s011; physical-floor historical height and label-free crown-mixture covariates/C=.15 from s012; shared-optical/local-deviation design and frozen October SAR view from s010. **Retired from final:** s009 nested offsets/one-sided vigor and reconstruction branch, s010 estate-name branch and optical-patch group, s012 probability bagging, common latent source pooling, new feature extraction and unvalidated pruning. Sources for every parent and V candidate are preserved; this session does not repeat the older unavailable-source narrative.

## Exact final training recipe

For the target estate and each source estate independently, keep Healthy/Unhealthy and construct49 local predictors:9 standardized NDVI/east/north degree≤2 terms (six degree2 terms multiplied.3),12 standardized compact optical features×.5,8 training PCs of standardized64-dimensional2025 AlphaEarth local contrast×.5,3 historical height features×.35 with sqrt(training_variance+[3m,1m,1m]^2) scaling,9 standardized deblur optical features×.3, and8 training PCs of frozen October SAR64-view PCs×.3. SAR PCA uses saved compressed components directly, then standardizes its8 outputs; AlphaEarth inputs are standardized before PCA. Both use deterministic full SVD.

The12 optical measurements additionally form a shared standardized block×.3. Each estate has its own49-column local block and an intercept indicator×10. With three estates there are162 classifier columns:12+3×49+3. Set each estate's total sample weight to the target binary training size, half on each class. Fit LogisticRegression(C=.15,max_iter=2000,tol=1e-7,fit_intercept=False). Return the target block decision function as finite float64. This shares optical coefficients while retaining local deviations and independent spatial/latent/structural responses. No physical sign constraint is imposed on estate-brittle context.

The two runtime tables are triage_v12/features.parquet and dino_sar/features.parquet. The former consolidates compact optical, prior-year embeddings, historical height and optical inversion; the latter supplies only oct103_ components. Runtime reads them through ctx.datasets_dir, uses ctx.cpus and writes nothing. No network/GPU/raster/library outside the declared pins is needed. Historical height predates2026; annual embeddings encode2025 context; frozen SAR RGB features are out-of-domain representations. None is a measured pathogen biomarker. Existing manifests/scripts specify native raster CRS reprojection, HH/HV/VV bands1/2/4, positive linear-power support and10log10 display preprocessing. No Yamaguchi/T3 extraction was performed this session.

## Repeated blocks and conditional uncertainty

{md(repeat)}

Frozen20 assignments: incumbent wins {int((final.delta_incumbent>0).sum())}/60 estate cells, coords wins {int((final.delta_coords>0).sum())}/60. Composite beats incumbent {int((comp.delta_incumbent>0).sum())}/20 and s012 {int((comp.delta_s012>0).sum())}/20. Means in every estate improve s012, but several individual AirHitam/Palong assignments do not. All parents were also run on these exact assignments:

{md(frozen_all)}

Paired4,000-resample bootstrap uses original80 m blocks within each estate, fixed official OOF predictions and common multiplicities for each comparator. Reported intervals condition on the adaptive campaign and fixed fits; they do not refit, correct model selection, remove adjacent-block dependence, or establish new-estate/year significance.

{md(bootcomp)}

The point official composite delta versus incumbent is+.147568, with95% interval[+.081769,+.206256]; coordinates point delta+.198494, interval[+.119152,+.246979]. Bootstrap means differ from original point estimates because AP is nonlinear. Incremental intervals versus s010 and s012 cross zero. AirHitam's individual incumbent interval also crosses zero; no universal individual-estate significance claim is made.

## Wider spatial holdouts and failures

{spatial_table}

The final exceeds incumbent in all estates on120/160 m blocks, shifted grid and40/80 m random-block buffers. It trails incumbent in **Palong blockquad80** by−.016145; at160 m region buffer training may be as small as3 binary rows in one Palong fold, and the recipe's reduced-PC/fallback behavior matters. Non-monotonic results between80/160 m buffers reflect changed training support and must not be interpreted as a smooth robustness curve. Sparse-tail quadrant Serting lift is close to zero even though its official score is strong. S012 shares this weakness. This is additional information beyond15NN, not proof of spatially invariant disease physiology.

Unmodified s009 fails Serting's legacy median80 inner validation (<5 groups) and Palong blockquad160 (fixed8 PCs with only3 samples). s010 also fails that Palong160 m stress for fixed8 PCs; AirHitam and Serting were attempted separately and their available results are retained. Early generic offset prototypes fail a one-class inner fit under buffer80 and have NA for the unfinished estate/protocol. Those failures are documented in research logs/JSON, not filled with scores. An empty-shared-block harness bug was fixed and pool_no_share rerun; a pre-build invocation of nocoords was rerun after its file existed. No primary parent or final official/frozen-assignment result failed.

![Spatial holdout sensitivity](spatial_robustness.png)

## V observations and final decision

Eight distinct mechanisms/crossovers were queried, all valid;7/15 remain. Gaussian noiseσ=.005 is cached per content hash. V observations are adaptive and correlated in model provenance. No cosmetic re-query, tail clipping, source-label recovery, or V-fitted ensemble weight was used.

{md(pd.DataFrame(vr))}

V8 versus V1 improves composite by+.011557, with increments+.005521/+.002382/+.026769. This is consistent with the combined architecture helping Serting, but small AirHitam/Palong differences are not precise mechanism estimates. The final decision and its explicit post-V checkpoint-guard amendment are in selection.json. The final has all-positive V estate deltas versus incumbent/coords, but cannot be described as all-parent non-degrading or independently V-confirmed superior to s010. s012 remains the conservative Palong-oriented alternative; s010 remains the stronger Serting-V reference.

## Feature signs, conditional suppression and operational errors

{md(sign)}

ndvi_sd20 and ndvi_edge correlate .947/.970/.983 approximately across estates, yet their marginal signs differ from their negative joint coefficients in Palong/Serting. qNDVI/qNDRE/qNDMI, raw/quality/deblur spectra and spatial interactions are also correlated; deblur moisture/NBR correlation reaches.975. s012's Palong qNDVI encoded coefficient is+.064 despite negative marginal d, while the final combined shared/local standardized qNDVI coefficient is−.055. This is a conditional coefficient redistribution, not proof of a corrected biological pathway. Supplied NDVI remains a core predictor. qNIRv/NDMI contrasts/dip still show conditional suppressions in some estates, and pruning controls do not justify blindly dropping them.

Historical mean-height marginal signs are+.307/−.558/−.256; October SAR/latent axes must not be pooled as universal disease signs. Independent PCA axes are not comparable across estates/folds. feature_effects.csv, feature_correlations.csv, model_coefficients.csv and final_optical_coefficients.csv record the marginal and conditional evidence.

{md(triage)}

{md(bud25)}

Middle consumes inspections without being declared Healthy. At25% budget Palong uses94 inspections on Middle; AirHitam costs about7.93 other inspections per detected Unhealthy. Unknown U outcomes preclude measuring actual field precision/recall or labeling low-score unsupported rows false negatives. Palong minimum U score remains about−24.97 under quadratic extrapolation; covariate_support.csv records outside-range features. error_by_positive_distance.csv reports small, descriptive distance bins using one estate-wide top25% allocation, never reranking within each bin.

![Paired lineage performance and inspection capacity](lineage_and_triage.png)

## Integrity, runtime and reproducibility

Final selfcheck and official CLI CV pass (selfcheck.txt, official_cv.txt). The clean single-file package equals the immutable V8 package in **{len(equiv)} paired fits**, maximum absolute difference **{max(v['max_difference'] for v in equiv):.1f}**: full U, every official fold, buffered whole-block regions and one reserved repeat. Packaging was not re-scored for fresh V noise. File SHA256 is `{sha}`; V8 package content hash is `0ebba5e7a34162954e60a5d6d521693f8681df2376f07cd6960cbf3c9623849f`.

Fresh workers use scratch cwd/4 CPUs and guard network, source-directory reads and writes outside scratch. Shape/dtype/finiteness, immutability, deterministic repetition, shuffled/duplicate targets, dropped Middle, renamed ctx estate, empty target and single-class fallback pass. Full-fit runtimes are at most{max(v['seconds'] for v in contract):.3f}s, peak observed RSS{max(v['peak_rss_MiB'] for v in contract):.1f}MiB. Guards are Python-level checks, not a replacement for grading isolation. Only non-base pin is threadpoolctl==3.8.0. Matplotlib3.10.7 is research-only.

All10 feature tables retain7,220 exact keys, zero duplicate keys/NaNs/nonfinite covariates/forbidden feature names/symlinks. Total datasets including preserved sources are about{diag['total_dataset_bytes']/1024**2:.2f}MiB, below128MiB carry-forward and512MiB caps. All4,473 exact-order U scores/ranks are in unlabelled_ranking.csv/.parquet. No predictions are copied into feature tables.

Essential runtime datasets: **triage_v12 and dino_sar**. Preserve their manifests/extraction scripts and recipes/s013. For continued parent auditing, also retain gee_quality, gee_canopy, optical_jitter, alphaearth and dino_optical. compact_triage, highres_canopy and crown_mixture duplicate values already consolidated into triage_v12 and are safe to archive/prune only after confirming no further extraction/provenance audit is needed. Unused SAR views in dino_sar could be compacted into a new exact-key subset in a future packaging step; do not silently mutate the original table because parent recipes use other views. Nothing was deleted here.

Core reproduction (all label use is authorized L):

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s010 s011 s012 s012_bag pooled_sar --protocols official block120 block160 buffer40 buffer80 blockquad blockquad80 blockquad160
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/final_diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/audit.py
python /app/research/write_report.py
```

Every research candidate is preserved under datasets/triage_v12/recipes/s013/candidates; every scored one has its exact portable source. Parents and reproduction scripts are also carried there, avoiding dependence on ephemeral research directories. Restore reproduction into /app/research; evaluate.py falls back to carried parents/candidates if original staging is absent. Cache NPZs include implementation hashes and only authorized OOF predictions/fold indices. Use --force for deliberate reruns; do not mix partial-estate composites. Raw GEE/raster extraction scripts remain in their existing dataset directories; provider reprocessing can alter fresh covariates, so saved table hashes define this reproducible snapshot.

## Next scientific work

1. Prospective80 m-block inspection in all estates, with weak-support/distant Palong, AirHitam misses, Middle costs and a random unranked audit sample. More local model tuning cannot resolve unseen conditional relationships.
2. Independent estates/years, accurately registered finer current crown imagery, and measured outcomes. Native S2 pixels mix palms spaced about9m; frozen embeddings/optical inversion do not create measured crown resolution.
3. Separate source-label information from regularization using additional independent domains. Existing controls establish only limited conditional transfer evidence. Protect the preserved parents and compare future models under the same folds, with footprint-aware buffers.
4. If revisiting the alternative with higher Palong V (s012) or higher Serting V (s010), use new outcomes or a genuinely new mechanism. Do not fit estate-specific ensemble weights to these noisy V observations or rerun hashes for noise.

## Complete official experiment table

Every row uses the same official folds. All per-estate/protocol paired deltas and secondary metrics are in EXPERIMENTS.csv; missing stress cells remain NA. Values below are exploratory, including final ablations after freeze.

{official_table}

## Primary literature and documentation

- [Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193): frozen general-purpose visual representations motivate the carried patch mechanism; no oil-palm pathology validation is inferred.
- [Brown et al., AlphaEarth Foundations](https://arxiv.org/abs/2507.22291): learned geospatial embeddings supply contextual measurements, not physically signed disease indices.
- [Sentinel-2 SR Harmonized catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED): authoritative band/resolution/harmonization provenance; no second+1000 offset is applied.
- [Meta/WRI HighResCanopyHeight](https://github.com/facebookresearch/HighResCanopyHeight): source/method provenance for historical height; historical structure is not current disease confirmation.
- [Izzuddin et al., airborne hyperspectral Ganoderma analysis](https://doi.org/10.21894/jopr.2018.0037): context for canopy-condition screening; no disease-specific accuracy from that paper is transferred to this dataset.
- [Roberts et al., spatial/temporal cross-validation](https://doi.org/10.1111/ecog.02881): blocking/extrapolation rationale. Publisher text was403; Crossref metadata was checked, not an unavailable full-paper analysis.

The web endpoint failed; accessible primary arXiv/catalogue/publisher/repository pages were fetched directly and saved under literature/, with HTTP status provenance. All empirical claims in this report come from the authorized local experiments, not imported paper accuracies.
'''
(R/'REPORT.md').write_text(readme)
keymodels=['incumbent','coords','s009','s010','s011','s012','s012_bag','uniform_sar','both_offset','pool_fullopt','s010_curvature','sar_bag','legacy_complement','legacy_protected','pooled_sar']
short=allofficial.set_index('Recipe').reindex(keymodels).reset_index()
handoff=f'''# Session 261127-s013-synthesis

## State of play

Final frozen submission: **pooled_sar**, one CPU/offline pipeline.py plus threadpoolctl==3.8.0. Uniform s012 compact optical/AlphaEarth/height/mixture design + s010 shared-optical/local-deviation architecture and October SAR8PCs.49 local predictors per estate,12 shared optical,162 total classifier columns, C=.15. No estate-name branch, label-neighbor feature, Middle target or score averaging. All three estates equally trusted/weighted.

Official lift A/P/S **+.075808/+.409951/+.378503**, composite **+.288087**; incumbent+.140519,coords+.089593,s010+.255805,s012+.269549. V **+.220382**, delta incumbent+.101632/coords+.153921; estates+.063217/+.338320/+.259611. Eight queries used,7 remain; no more V tuning after V8. No GPU/GEE work launched.

Selection is exploratory and V-informed. It loses Palong V vs s012 by.024857 and Serting V vs s010 by.046352; +.003557 composite vs s010 is below scorer noise. **It fails the provisional .015 checkpoint-loss guard; selection.json explicitly discloses the post-V policy amendment to the task's incumbent-relative criteria. Do not call this a passed preregistered all-parent non-degradation test.** Conservative s012 and all stronger per-estate references are preserved.

Reserved20 block assignments401–420:60/60 estate wins versus incumbent and coords; mean extra lift over s012 A/P/S+.008926/+.004100/+.036063; s012 estate wins17/20,13/20,20/20. Composite s012 wins{int((comp.delta_s012>0).sum())}/20. Conditional4,000-block bootstrap composite95% delta incumbent[+.081769,+.206256],coords[+.119152,+.246979]; intervals vs s010/s012 cross0. Repeats reuse labels; AirHitam individual incumbent interval crosses0.

Binary Recall25 **48.2/74.2/76.1%**; counting Middle capacity **48.2/71.2/74.6%**. AirHitam majority capture remains unmet at25%. Palong whole-block quadrant+80m buffer loses.016145 to incumbent.160m buffered regions remain positive but can leave3 binary training rows. Sparse-tail Serting region skill approaches0. Unknown U outcomes remain unknown; no field recall/precision guarantee.

Final officialCV/selfcheck/pins/fresh-process contract pass. Clean package matches V8 in45 fits exactly. Runtime≤{max(v['seconds'] for v in contract):.3f}s/estate,RSS≤{max(v['peak_rss_MiB'] for v in contract):.1f}MiB.4473 exact-order rankings saved.10 tables with exact7220 keys, no NaNs/nonfinite/forbidden feature columns/symlinks;~49MiB including preserved sources. Only runtime tables triage_v12 and dino_sar. Pipeline SHA256 `{sha}`.

## What I tried

{md(short)}

All values official80m A/P/S and paired deltas; V column uses supplied parent rewards/new observations. Evidence: report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,VERIFIER_SUMMARY.csv,REPORT.md; immutable authorized OOF predictions research/results/MODEL_PROTOCOL_ESTATE.npz. Main campaign has{df.model.nunique()} models. Staged parent code was available and actually rerun; no unavailable-source reconstruction claim.

- Capacity crossover: exact s010 except uniform.3 quadratic shrinkage; s010_curvature removes name branch, improves L, retains V tradeoff.
- Joint versus s009-style bounded block-OOF patch offsets: small stable local gains, no balanced parent V dominance.
- Compact optical partial pooling with self/permutation/no-share controls; combine with October SAR to form final. Some gains are changed regularization, not established transferred physiology.
- s01232-block probability bag crossed with SAR/optical patches: positive repeat means; V not balanced enough. No model score averaging in final.
- Missing quality/moisture/height/mixture measurements around uniform s010, joint or bounded: L Palong improves, V does not. This does not explain away parent divergence.
- Common aligned SAR basis transfer fails fresh Palong repeat mean and receives no V query. Raw-alpha scaling, quality blend and targeted redundant-feature pruning give mixed small gains.

Parent rank correlations s010/s012 A/P/S=.797/.896/.806; top25 disagreement15.35/9.22/9.04% of binary rows. s012 Palong qNDVI conditional sign is positive despite negative marginal d; final shared+local coefficient is negative. SD/edge correlate.947–.983 and have sign reversals. Coefficients are conditional predictive associations, not pathogen biomarkers; see signs/correlations/weights CSVs.

## Dead ends

- MainChecklist.md closed leads remain closed; do not repeat SAR means/algorithm grids, SMOTE, leaky label neighbors or Middle weak positives.
- No naive V-weighted ensemble/estate gates. Large L gains do not predict the estate V ordering: s010 stronger Serting V, s012 stronger Palong V.
- Optical NRG patch group, common latent SAR pooling, new bags and missing-family offsets do not earn final complexity. All eight exact V packages remain available.
- s009 fails legacy median80 Serting nested5-fold with<5 groups and Palong160m buffer PCA8 with3samples; s010 fails the latter. Attempted other estates are retained. Some generic offset prototypes fail one-class inner folds atbuffer80; these are NA, not zero/perfect scores. Initial empty-shared-block harness failure was corrected and rerun.
- Sparse-tail whole-block quadrant leaves only8 training positives for590/815 held-out Serting rows; near-zero skill reflects serious extrapolation/small-sample limits. Legacy median quadrants cut some80m blocks; do not call them block-disjoint.

## Open leads, ranked

1. Prospective whole-block inspections in all estates; include weak-support/distant Palong, AirHitam misses, Middle costs and random unranked audits. More same-label CV tuning cannot validate U outcomes.
2. Independent estates/years and accurate finer crown registration/current imagery.103m SAR context overlaps80m boundaries;160m region buffers help but do not remove the optical inverse's global tail.
3. Separate source-label information from regularization using independent domains. Do not interpret partial pooling as a universal disease signature.
4. Preserve conservative s012 and high-Serting-V s010. Confirm new mechanisms with new outcomes; no cosmetic V re-queries or estate-specific mixture-weight fit.

Essential datasets: **triage_v12,dino_sar**. Keep gee_quality,gee_canopy,optical_jitter,alphaearth,dino_optical for exact parent audits. compact_triage/highres_canopy/crown_mixture are redundant with consolidated values and safe to archive only after provenance needs are met. Do not mutate dino_sar in place; parent recipes use other saved views. Eight advertised datasets were absent at startup; no unnecessary re-extraction occurred.

## How to resume

Read REPORT.md,selection.json,validation_plan.json and MainChecklist.md. Final recipe is frozen after V8;7 queries remain but another micro-tuning round is not justified. No GPU session was started. Source-preserved portable variants and parents are **carried in datasets/triage_v12/recipes/s013/**, avoiding dependence on research paths surviving handoff. Restore reproduction/ into /app/research if needed; evaluate.py falls back to carried parents/candidates. Source feature files remain unchanged and source hashes audited.

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s010 s011 s012 s012_bag pooled_sar --protocols official block120 block160 buffer40 buffer80 blockquad blockquad80 blockquad160
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/final_diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/audit.py
python /app/research/write_report.py
```

Use --force for deliberate evaluator reruns; caches include implementation hashes. Full composites require3 estates; partial failures are NA. Runtime pin only threadpoolctl3.8.0; research plotting matplotlib3.10.7. Do not touch prior reports/checklist, /tests, sealed splits/labels, banned repo or credentials.

## Artifact index

- submission/pipeline.py,requirements.txt: clean selected offline model.
- datasets/triage_v12/recipes/s013/selected,parents,candidates,reproduction,evidence,README.md,checksums.json: all preserved sources/alternatives and reconstruction instructions. Each V-scored candidate has portable/ source and dependency metadata; no fitted labels or disease weights embedded.
- report/REPORT.md,HANDOFF.md,selection.json,validation_plan.json: conclusions, policy amendment, frozen decisions.
- report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,OFFICIAL_EXPERIMENTS.csv,VERIFIER_SUMMARY.csv,final_frozen_confirmation.csv,repeat_summary.csv: unified paired evidence.
- report/verifier/v1..v8 JSON;bootstrap.csv/.json;spatial_split_audit.csv: query results,conditional uncertainty and spatial support.
- report/model_coefficients.csv,final_optical_coefficients.csv,feature_effects.csv,feature_sign_consistency.csv,feature_correlations.csv,prediction_disagreement.csv: mechanisms/suppressors/signs/rank overlap.
- report/inspection_budget_diagnostics.csv,error_by_positive_distance.csv,covariate_support.csv,unlabelled_ranking.csv/.parquet: inspection costs,errors,support,4473 aligned U scores.
- report/lineage_and_triage.png/.pdf,spatial_robustness.png/.pdf,literature/: standalone figures/primary sources.
- report/selfcheck.txt,official_cv.txt,package_equivalence_final.json,contract_verification.json,dataset_audit.json,environment.json: verification,exact45-fit equality,48.9MiB data audit.
- research/evaluate.py,synthesis.py,legacy_extra.py,build.py,package.py,aggregate.py,diagnose.py,bootstrap.py,equivalence.py,contract.py,final_diagnostics.py,audit.py,plots.py,preserve.py,write_report.py: reproducible research; results/*.npz and logs remain local evidence.
'''
(R/'HANDOFF.md').write_text(handoff)
print('Wrote REPORT.md and six-section HANDOFF.md')
