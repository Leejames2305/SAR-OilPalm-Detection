# Path-Forward Trials: Supervised, Unsupervised, and Healthy-First SAR Learning

Generated from `misc/scripts/path_forward_trials/` using the sampled dataset at `data/Processed/dataset_all_estates.csv`.

## Protocol

- Primary metric: PR-AUC, compared with estate prevalence; secondary metric: enrichment in the top 10% ranked trees.
- Honest local CV: StratifiedGroupKFold on the existing ~220 m blocks, repeated with three seeds.
- Broader holdout: adjacent blocks assigned to four spatial regions, then leave-one-region-out.
- Cross-estate: leave-one-estate-out. Estate-wise z-score is reported as a domain-normalisation variant.
- All labels are used only in supervised stages; Healthy-only detectors are fitted to Healthy training trees only.

## Decision comparison

| protocol | estate | role | model | feature_set | pr_auc_mean | pr_auc_lift_mean | roc_auc_mean | enrichment_at_10pct_mean |
|---|---|---|---|---|---|---|---|---|
| block_5fold | AHCensus_Basemap | RF baseline | rf_balanced | all14 | 0.072 | 0.008 | 0.523 | 1.162 |
| block_5fold | AHCensus_Basemap | Best supervised SAR | logreg_l2 | log_raw4 | 0.092 | 0.028 | 0.569 | 1.681 |
| block_5fold | AHCensus_Basemap | Best Healthy-only | pca_recon_10 | all14 | 0.072 | 0.008 | 0.519 | 1.308 |
| block_5fold | AHCensus_Basemap | Best SAR + spatial | logreg_l2 | raw4_plus_local_plus_coords | 0.121 | 0.057 | 0.620 | 2.180 |
| block_5fold | AHCensus_Basemap | Spatial label context | knn_label_fraction_25 | coords_plus_training_labels | 0.099 | 0.035 | 0.584 | 1.993 |
| broad_region_4fold | AHCensus_Basemap | RF baseline | rf_balanced | all14 | 0.066 | 0.002 | 0.505 | 0.996 |
| broad_region_4fold | AHCensus_Basemap | Best supervised SAR | logreg_l2 | log_raw4 | 0.092 | 0.029 | 0.588 | 1.681 |
| broad_region_4fold | AHCensus_Basemap | Best Healthy-only | pca_recon_10 | all14 | 0.073 | 0.010 | 0.521 | 1.370 |
| broad_region_4fold | AHCensus_Basemap | Best SAR + spatial | logreg_l2 | raw4_plus_local_plus_coords | 0.125 | 0.062 | 0.618 | 2.491 |
| broad_region_4fold | AHCensus_Basemap | Spatial label context | knn_label_fraction_10 | coords_plus_training_labels | 0.083 | 0.019 | 0.496 | 1.246 |
| block_5fold | Palong_Basemap | RF baseline | rf_balanced | all14 | 0.087 | -0.005 | 0.485 | 0.896 |
| block_5fold | Palong_Basemap | Best supervised SAR | logreg_l2 | log_raw4 | 0.114 | 0.022 | 0.551 | 1.444 |
| block_5fold | Palong_Basemap | Best Healthy-only | pca_recon_10 | local14 | 0.111 | 0.019 | 0.576 | 1.295 |
| block_5fold | Palong_Basemap | Best SAR + spatial | logreg_l2 | all14_plus_coords | 0.239 | 0.146 | 0.746 | 2.390 |
| block_5fold | Palong_Basemap | Spatial label context | nearest_u_minus_h | coords_plus_training_labels | 0.156 | 0.063 | 0.663 | 1.593 |
| broad_region_4fold | Palong_Basemap | RF baseline | rf_balanced | all14 | 0.080 | -0.012 | 0.454 | 0.647 |
| broad_region_4fold | Palong_Basemap | Best supervised SAR | logreg_l2 | all14 | 0.111 | 0.019 | 0.558 | 1.295 |
| broad_region_4fold | Palong_Basemap | Best Healthy-only | pca_recon_10 | local14 | 0.112 | 0.020 | 0.582 | 1.344 |
| broad_region_4fold | Palong_Basemap | Best SAR + spatial | logreg_l2 | all14_plus_coords | 0.242 | 0.150 | 0.742 | 2.440 |
| broad_region_4fold | Palong_Basemap | Spatial label context | nearest_u_minus_h | coords_plus_training_labels | 0.100 | 0.008 | 0.504 | 1.344 |
| block_5fold | Serting_Basemap | RF baseline | rf_balanced | all14 | 0.083 | -0.009 | 0.471 | 0.540 |
| block_5fold | Serting_Basemap | Best supervised SAR | svm_rbf | all14 | 0.098 | 0.006 | 0.519 | 1.232 |
| block_5fold | Serting_Basemap | Best Healthy-only | isolation_forest | raw4_plus_local | 0.116 | 0.024 | 0.485 | 1.124 |
| block_5fold | Serting_Basemap | Best SAR + spatial | xgboost_balanced | log_raw4_plus_coords | 0.179 | 0.087 | 0.621 | 2.463 |
| block_5fold | Serting_Basemap | Spatial label context | nearest_u_relative_distance | coords_plus_training_labels | 0.154 | 0.062 | 0.532 | 1.685 |
| broad_region_4fold | Serting_Basemap | RF baseline | rf_balanced | all14 | 0.083 | -0.009 | 0.464 | 0.713 |
| broad_region_4fold | Serting_Basemap | Best supervised SAR | xgboost_balanced | log_raw4 | 0.095 | 0.002 | 0.517 | 0.778 |
| broad_region_4fold | Serting_Basemap | Best Healthy-only | isolation_forest | raw4_plus_local | 0.116 | 0.023 | 0.473 | 1.232 |
| broad_region_4fold | Serting_Basemap | Best SAR + spatial | rf_balanced | log_raw4_plus_coords | 0.164 | 0.071 | 0.597 | 2.463 |
| broad_region_4fold | Serting_Basemap | Spatial label context | nearest_u_relative_distance | coords_plus_training_labels | 0.162 | 0.070 | 0.567 | 1.491 |

## Model race on all 14 SAR features

| estate | model | pr_auc_mean | pr_auc_std | roc_auc_mean | enrichment_at_10pct_mean |
|---|---|---|---|---|---|
| AHCensus_Basemap | knn_25 | 0.091 | 0.000 | 0.544 | 1.619 |
| AHCensus_Basemap | hist_gbdt_balanced | 0.079 | 0.006 | 0.536 | 1.453 |
| AHCensus_Basemap | gaussian_nb | 0.077 | 0.000 | 0.554 | 1.495 |
| AHCensus_Basemap | logreg_l1 | 0.074 | 0.000 | 0.530 | 1.308 |
| AHCensus_Basemap | logreg_l2 | 0.073 | 0.000 | 0.539 | 1.059 |
| AHCensus_Basemap | lda_shrink | 0.073 | 0.000 | 0.538 | 1.183 |
| AHCensus_Basemap | rf_balanced | 0.072 | 0.001 | 0.523 | 1.163 |
| AHCensus_Basemap | extratrees_balanced | 0.071 | 0.001 | 0.520 | 1.038 |
| AHCensus_Basemap | xgboost_balanced | 0.068 | 0.001 | 0.498 | 1.142 |
| AHCensus_Basemap | svm_rbf | 0.068 | 0.000 | 0.516 | 0.996 |
| AHCensus_Basemap | mlp_64_16 | 0.066 | 0.006 | 0.494 | 1.059 |
| Palong_Basemap | logreg_l2 | 0.103 | 0.000 | 0.553 | 1.145 |
| Palong_Basemap | logreg_l1 | 0.103 | 0.000 | 0.540 | 1.095 |
| Palong_Basemap | hist_gbdt_balanced | 0.102 | 0.002 | 0.515 | 1.178 |
| Palong_Basemap | svm_rbf | 0.097 | 0.000 | 0.530 | 0.846 |
| Palong_Basemap | xgboost_balanced | 0.097 | 0.001 | 0.527 | 0.996 |
| Palong_Basemap | gaussian_nb | 0.097 | 0.000 | 0.537 | 1.046 |
| Palong_Basemap | mlp_64_16 | 0.094 | 0.012 | 0.498 | 1.079 |
| Palong_Basemap | rf_balanced | 0.087 | 0.000 | 0.485 | 0.896 |
| Palong_Basemap | knn_25 | 0.087 | 0.000 | 0.478 | 0.797 |
| Palong_Basemap | extratrees_balanced | 0.087 | 0.001 | 0.482 | 0.797 |
| Palong_Basemap | lda_shrink | 0.082 | 0.000 | 0.462 | 0.647 |
| Serting_Basemap | svm_rbf | 0.098 | 0.000 | 0.519 | 1.232 |
| Serting_Basemap | mlp_64_16 | 0.097 | 0.015 | 0.509 | 0.929 |
| Serting_Basemap | knn_25 | 0.094 | 0.000 | 0.519 | 0.907 |
| Serting_Basemap | logreg_l2 | 0.094 | 0.000 | 0.500 | 0.454 |
| Serting_Basemap | logreg_l1 | 0.089 | 0.000 | 0.491 | 0.713 |
| Serting_Basemap | xgboost_balanced | 0.088 | 0.003 | 0.478 | 0.670 |
| Serting_Basemap | lda_shrink | 0.086 | 0.000 | 0.472 | 0.648 |
| Serting_Basemap | gaussian_nb | 0.086 | 0.000 | 0.483 | 0.519 |
| Serting_Basemap | hist_gbdt_balanced | 0.086 | 0.005 | 0.473 | 0.886 |
| Serting_Basemap | extratrees_balanced | 0.084 | 0.002 | 0.482 | 0.562 |
| Serting_Basemap | rf_balanced | 0.083 | 0.001 | 0.471 | 0.540 |

## Feature-family ablation (XGBoost)

| estate | feature_set | pr_auc_mean | pr_auc_lift_mean | enrichment_at_10pct_mean |
|---|---|---|---|---|
| AHCensus_Basemap | raw4 | 0.067 | 0.004 | 1.142 |
| AHCensus_Basemap | log_raw4 | 0.067 | 0.004 | 1.142 |
| AHCensus_Basemap | existing_indices | 0.065 | 0.001 | 1.391 |
| AHCensus_Basemap | extended_indices | 0.068 | 0.004 | 0.913 |
| AHCensus_Basemap | local14 | 0.069 | 0.005 | 1.017 |
| AHCensus_Basemap | all14 | 0.068 | 0.004 | 1.142 |
| AHCensus_Basemap | raw4_plus_ext | 0.067 | 0.003 | 1.246 |
| AHCensus_Basemap | raw4_plus_local | 0.070 | 0.007 | 1.412 |
| AHCensus_Basemap | all14_plus_ext | 0.070 | 0.006 | 1.121 |
| AHCensus_Basemap | all14_plus_local | 0.073 | 0.010 | 1.557 |
| AHCensus_Basemap | coords_only | 0.060 | -0.003 | 0.996 |
| Palong_Basemap | raw4 | 0.088 | -0.004 | 0.846 |
| Palong_Basemap | log_raw4 | 0.088 | -0.004 | 0.846 |
| Palong_Basemap | existing_indices | 0.093 | 0.001 | 1.046 |
| Palong_Basemap | extended_indices | 0.079 | -0.013 | 0.515 |
| Palong_Basemap | local14 | 0.084 | -0.008 | 0.681 |
| Palong_Basemap | all14 | 0.097 | 0.005 | 0.996 |
| Palong_Basemap | raw4_plus_ext | 0.080 | -0.012 | 0.681 |
| Palong_Basemap | raw4_plus_local | 0.093 | 0.001 | 0.996 |
| Palong_Basemap | all14_plus_ext | 0.094 | 0.001 | 0.996 |
| Palong_Basemap | all14_plus_local | 0.102 | 0.009 | 1.212 |
| Palong_Basemap | coords_only | 0.172 | 0.080 | 2.042 |
| Serting_Basemap | raw4 | 0.097 | 0.004 | 0.929 |
| Serting_Basemap | log_raw4 | 0.097 | 0.004 | 0.929 |
| Serting_Basemap | existing_indices | 0.077 | -0.015 | 0.756 |
| Serting_Basemap | extended_indices | 0.087 | -0.005 | 0.756 |
| Serting_Basemap | local14 | 0.095 | 0.003 | 0.972 |
| Serting_Basemap | all14 | 0.088 | -0.005 | 0.670 |
| Serting_Basemap | raw4_plus_ext | 0.091 | -0.002 | 0.972 |
| Serting_Basemap | raw4_plus_local | 0.092 | -0.001 | 0.994 |
| Serting_Basemap | all14_plus_ext | 0.089 | -0.003 | 0.843 |
| Serting_Basemap | all14_plus_local | 0.092 | -0.001 | 1.124 |
| Serting_Basemap | coords_only | 0.163 | 0.071 | 2.182 |

## Cross-estate transfer

| estate | model | feature_set | pr_auc | pr_auc_lift | roc_auc | enrichment_at_10pct |
|---|---|---|---|---|---|---|
| AHCensus_Basemap | rf_balanced | all14 | 0.084 | 0.020 | 0.525 | 1.370 |
| AHCensus_Basemap | rf_balanced | all14_plus_ext | 0.076 | 0.012 | 0.518 | 1.183 |
| AHCensus_Basemap | rf_balanced | all14_estate_z | 0.074 | 0.010 | 0.483 | 1.121 |
| AHCensus_Basemap | xgboost_balanced | local14_estate_z | 0.073 | 0.009 | 0.517 | 1.308 |
| AHCensus_Basemap | logreg_l2 | all14 | 0.073 | 0.009 | 0.537 | 1.308 |
| AHCensus_Basemap | logreg_l2 | all14_plus_local | 0.072 | 0.008 | 0.530 | 1.059 |
| AHCensus_Basemap | logreg_l2 | all14_plus_ext | 0.072 | 0.008 | 0.518 | 1.432 |
| AHCensus_Basemap | rf_balanced | local14 | 0.071 | 0.007 | 0.504 | 1.121 |
| AHCensus_Basemap | xgboost_balanced | local14 | 0.071 | 0.007 | 0.514 | 1.183 |
| AHCensus_Basemap | rf_balanced | local14_estate_z | 0.070 | 0.006 | 0.499 | 1.308 |
| AHCensus_Basemap | xgboost_balanced | all14_plus_ext | 0.068 | 0.004 | 0.492 | 0.872 |
| AHCensus_Basemap | xgboost_balanced | all14_estate_z | 0.067 | 0.003 | 0.482 | 1.059 |
| AHCensus_Basemap | rf_balanced | all14_plus_ext_estate_z | 0.067 | 0.003 | 0.496 | 0.934 |
| AHCensus_Basemap | xgboost_balanced | all14_plus_local | 0.065 | 0.002 | 0.505 | 0.996 |
| AHCensus_Basemap | xgboost_balanced | all14 | 0.065 | 0.002 | 0.495 | 0.872 |
| AHCensus_Basemap | xgboost_balanced | all14_plus_local_estate_z | 0.065 | 0.001 | 0.503 | 0.996 |
| AHCensus_Basemap | rf_balanced | all14_plus_local | 0.065 | 0.001 | 0.521 | 0.872 |
| AHCensus_Basemap | logreg_l2 | local14 | 0.064 | 0.000 | 0.500 | 0.934 |
| AHCensus_Basemap | xgboost_balanced | all14_plus_ext_estate_z | 0.064 | 0.000 | 0.484 | 0.996 |
| AHCensus_Basemap | rf_balanced | all14_plus_local_estate_z | 0.063 | -0.001 | 0.507 | 0.872 |
| AHCensus_Basemap | logreg_l2 | all14_plus_local_estate_z | 0.063 | -0.001 | 0.495 | 0.810 |
| AHCensus_Basemap | logreg_l2 | all14_estate_z | 0.062 | -0.002 | 0.497 | 0.685 |
| AHCensus_Basemap | logreg_l2 | local14_estate_z | 0.062 | -0.002 | 0.492 | 0.747 |
| AHCensus_Basemap | logreg_l2 | all14_plus_ext_estate_z | 0.061 | -0.003 | 0.485 | 0.872 |

## Unsupervised structure

| estate | neighbor_u_rate_if_u | neighbor_u_rate_if_h | difference | permutation_p_one_sided |
|---|---|---|---|---|
| AHCensus_Basemap | 0.119 | 0.060 | 0.059 | 0.002 |
| Palong_Basemap | 0.336 | 0.060 | 0.276 | 0.002 |
| Serting_Basemap | 0.255 | 0.070 | 0.185 | 0.002 |

| estate | feature_set | method | n_clusters | ari | ami | max_cluster_unhealthy_rate | max_cluster_enrichment |
|---|---|---|---|---|---|---|---|
| AHCensus_Basemap | extended_indices | kmeans | 2 | 0.007 | 0.002 | 0.078 | 1.216 |
| Palong_Basemap | all14_plus_local | kmeans | 2 | -0.046 | 0.007 | 0.100 | 1.085 |
| Serting_Basemap | local14 | gmm_diag | 2 | 0.008 | 0.000 | 0.104 | 1.125 |

## Review-budget curve for best spatial fusion

| estate | budget | k | precision | recall | enrichment |
|---|---|---|---|---|---|
| AHCensus_Basemap | 0.010 | 26 | 0.192 | 0.031 | 3.018 |
| AHCensus_Basemap | 0.050 | 126 | 0.198 | 0.156 | 3.114 |
| AHCensus_Basemap | 0.100 | 252 | 0.159 | 0.250 | 2.491 |
| AHCensus_Basemap | 0.150 | 377 | 0.130 | 0.306 | 2.040 |
| AHCensus_Basemap | 0.200 | 503 | 0.113 | 0.356 | 1.778 |
| AHCensus_Basemap | 0.250 | 628 | 0.100 | 0.394 | 1.574 |
| AHCensus_Basemap | 0.300 | 754 | 0.093 | 0.438 | 1.457 |
| AHCensus_Basemap | 0.400 | 1005 | 0.094 | 0.588 | 1.468 |
| AHCensus_Basemap | 0.500 | 1256 | 0.086 | 0.675 | 1.349 |
| Palong_Basemap | 0.010 | 22 | 0.500 | 0.055 | 5.428 |
| Palong_Basemap | 0.050 | 109 | 0.294 | 0.160 | 3.187 |
| Palong_Basemap | 0.100 | 218 | 0.225 | 0.245 | 2.440 |
| Palong_Basemap | 0.150 | 326 | 0.224 | 0.365 | 2.431 |
| Palong_Basemap | 0.200 | 435 | 0.218 | 0.475 | 2.371 |
| Palong_Basemap | 0.250 | 543 | 0.210 | 0.570 | 2.279 |
| Palong_Basemap | 0.300 | 652 | 0.192 | 0.625 | 2.081 |
| Palong_Basemap | 0.400 | 869 | 0.167 | 0.725 | 1.811 |
| Palong_Basemap | 0.500 | 1086 | 0.150 | 0.815 | 1.629 |
| Serting_Basemap | 0.010 | 17 | 0.235 | 0.026 | 2.547 |
| Serting_Basemap | 0.050 | 84 | 0.250 | 0.136 | 2.706 |
| Serting_Basemap | 0.100 | 167 | 0.180 | 0.195 | 1.945 |
| Serting_Basemap | 0.150 | 251 | 0.171 | 0.279 | 1.854 |
| Serting_Basemap | 0.200 | 334 | 0.162 | 0.351 | 1.750 |
| Serting_Basemap | 0.250 | 417 | 0.146 | 0.396 | 1.583 |
| Serting_Basemap | 0.300 | 501 | 0.134 | 0.435 | 1.448 |
| Serting_Basemap | 0.400 | 667 | 0.127 | 0.552 | 1.379 |
| Serting_Basemap | 0.500 | 834 | 0.125 | 0.675 | 1.350 |

## Incremental value of SAR on top of coordinates

| estate | cv | model | feature_set | delta_pr_auc | delta_roc_auc | ci_low | ci_high | prob_positive |
|---|---|---|---|---|---|---|---|---|
| AHCensus_Basemap | leave_one_region_out_4 | logreg_l2 | raw4_plus_local_plus_coords | 0.020 | -0.004 | -0.002 | 0.046 | 0.958 |
| AHCensus_Basemap | spatial_5fold_repeated | logreg_l2 | raw4_plus_local_plus_coords | 0.016 | 0.000 | 0.003 | 0.042 | 0.985 |
| Palong_Basemap | leave_one_region_out_4 | logreg_l2 | all14_plus_coords | 0.042 | 0.017 | -0.031 | 0.088 | 0.898 |
| Palong_Basemap | spatial_5fold_repeated | logreg_l2 | all14_plus_coords | 0.053 | 0.009 | -0.021 | 0.102 | 0.917 |
| Serting_Basemap | leave_one_region_out_4 | xgboost_balanced | log_raw4_plus_coords | 0.036 | 0.029 | -0.042 | 0.074 | 0.849 |
| Serting_Basemap | spatial_5fold_repeated | xgboost_balanced | log_raw4_plus_coords | 0.022 | 0.003 | -0.002 | 0.059 | 0.964 |

## Interpretation

1. Model substitution is not the main lever. Several learners beat RF slightly (kNN on AHCensus, logistic regression on Palong, SVM-RBF on Serting), but the winner changes by estate and none reaches a useful SAR-only operating point under spatial holdout.
2. Unsupervised clustering does not recover health classes: ARI values are near zero. The labels are spatially clustered, but the absolute SAR feature space does not form analogous clusters.
3. Learning Healthy and flagging deviations does not produce a robust improvement. PCA reconstruction, isolation forest, GMM, OCSVM, kNN distance, and an autoencoder score at or near no-skill on most estate/protocol combinations, with the best detector changing by estate.
4. Explicit RVI-adjacent indices do not rescue the model. VSI/CSI/Rc/Rp/RVIHH/RVIVV are deterministic functions of the supplied polarimetric channels and are mostly redundant with them.
5. Adding coordinates is the only large and repeatable effect in these trials. It is a spatial-prior result, not a radar classification result, and is relevant only to in-estate triage where nearby confirmed labels are available. It does not establish transfer to a new estate.
6. Cross-estate transfer remains weak. Estate-wise z-scoring helps some cases but does not create a robust disease signature.
7. Recommended path: deprioritise further algorithm shopping and Healthy-only anomaly detection. If the intended product is in-estate triage, develop a spatial decision-support model that combines confirmed labels, tree geometry, and SAR features, evaluated with broader regional holdouts. If the product must generalise to new estates, change the data: multi-temporal/change SAR, optical fusion, and more estates are higher-value than another classifier.

## Artifacts

- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\decision_rules.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\best_all14_models.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\feature_ablation_xgboost.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\best_cross_estate.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\label_spatial_autocorrelation.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\unsupervised_cluster_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\supervised_repeat_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\healthy_only_repeat_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\broad_spatial_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\spatial_fusion_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\spatial_context_metrics.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\fusion_vs_coords.csv`
- `D:\User_Stuff\Coding\MasterProject\SAR-OilPalm-Detection\misc\POC_Results\PathForwardTrials\budget_curve_best_fusion.csv`