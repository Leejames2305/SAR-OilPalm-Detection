# Exact runtime covariate subset, session 261127-s011-ablation

This label-free consolidation contains all 7,220 original exact (estate, Long,
Lat) keys, no rounding or duplicate keys, 79 columns (3 keys + 76 covariates),
zero NaNs, and no forbidden column names. It preserves the source-table keys
and values. Source checksums are in source_checksums.json. Reproduce with
`python /app/datasets/compact_triage/consolidate.py` after reproducing the four
carried upstream tables; their original extraction scripts/manifests remain.

Sources and windows:

- gee_quality: COPERNICUS/S2_SR_HARMONIZED with Cloud Score+ cs_cdf>=.65 and
  SCL4, April15–August15 2026; March15–September15 SCL4 fallback. Median per-image
  qNDVI, qNDRE, qNDMI, qNIRv. log_green_red=log(max(B3,1)/max(B4,1)), using
  median harmonized DN. No extra +1000 offset. Native RGB/NIR10m, red-edge/SWIR20m.
- gee_canopy: May15–July15 2026 Sentinel-2 median, SCL4–7; NDVI, NDRE1 and NDMI
  contrast = focal median10m minus focal median35m, sampled10m EPSG32648.
- optical_jitter: cloud-controlled optical source as gee_quality. Uniform mean
  of four registration offsets (+/-5m,+/-5m) around corrected trunk coordinates.
  NDVI ring10–30m mean minus centre (ndvi_dip), NDVI SD20m, central4x4 gradient
  magnitude (ndvi_edge), NDRE median10m minus median40m (ndre_contrast).
- alphaearth: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2025 only, all64 components
  c2025_A00..A63 = point embedding minus circular40m focal mean, EPSG32648 10m.
  Annual latent context is not a physical disease biomarker. Downstream8-PC
  reduction and scaling use binary training rows only.

No new raster/GEE/GPU extraction occurred in this consolidation. No label,
block, identifier, supervised fitted score, selected ranking, or split
assignment is included. Wider source tables are retained for provenance and
future ablations; the final runtime reads only this table. The dropped
GRVI_crown and all DINO patch embeddings are absent. See upstream manifests
for masking, source versions and extraction geometry. Neighborhood features
can share imagery across nearby trees; 120/160m and buffered spatial validation
is reported alongside80m validation.
