# Exact runtime consolidation, session261127-s012-open

All7220 original exact(estate,Long,Lat)keys, no duplicate/rounded keys/NaNs/
forbidden columns.76carried compact_triage covariates +3historical fine-canopy
height features +27native optical point/deblur/stable index/contrast covariates.
All options retained for reproducible feature ablations; final recipe uses one
optical variant only. No labels or fitted risk scores involved. See source
manifests/extraction scripts for full provenance; source SHA256 checksums included.

compact_triage: CloudScore+ S2 Apr-Aug2026 +canopy local optical contrasts and
registration summaries +AlphaEarth2025 64local-contrast components. highres_canopy:
Meta/WRI Maxar-based historical canopy model(2009-2020 imagery), native~1.194m,
mean6m/SD6m/mean6-minusmean24. crown_mixture: S2 Apr15-Aug15 2026 median,SCL4-7,
CloudScore+>=.65,same-window fallback; EPSG32648 native10/20m information sampled
on10m grid with independently audited pixel-center-to-corner registration.
Fixed Gaussian5m spectral observation model over complete tree coordinate list,
regularized inversion lambda .25(deblur)/1(stable), prior bilinear point spectra.
Index definitions/masks/limitations and scripts in crown_mixture. No claim of
physical superresolution. Native mosaics can be regenerated offline preparation
via EarthEngine, not required or accessed at fit_predict time.

Training-only scaling/PCA and local coordinate origin in pipeline. Historical
height denominator includes uncertainty floor(3m height,1m SD/contrast) to avoid
magnifying sub-accuracy variation in short-canopy stands. All estates receive
the same recipe and equally trusted ground truth. Scores are not probabilities.

`recipes/block_bag.py` and exactrequirements preserve the higher-noisyV experimental alternative for future independent confirmation. It was not selected under the preregistered multi-estate incremental promotion rule. No fitted scores or labels are embedded. Runtime does not import this alternate recipe.
