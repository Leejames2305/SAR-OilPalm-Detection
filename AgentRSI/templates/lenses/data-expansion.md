## Focus Lens: Data Expansion (Rasters & GEE)
Prioritize extracting new leak-free tree-level covariates for all `7,220` `(estate, Long, Lat)` rows into `/app/datasets/<name>/features.parquet`:
- Local ALOS-2 GeoTIFFs in `/app/rasters/` (Yamaguchi `260610`/`261014` after proper linear/dB handling, temporal `261014 - 260610` backscatter & H-Alpha deltas, texture/percentile window stats `winstats`, AirHitam T3 quad-pol).
- Google Earth Engine (`agentrsi gcp-init`): Sentinel-2 red-edge/SWIR (`B5`, `B6`, `B7`, `B8A`, `B11`, `B12`), topography/moisture/canopy products from `/app/prior/Plan-GEECatalogSurvey.md`.
Remember: extraction scripts belong alongside `MANIFEST.md` in `/app/datasets/<name>/`, while `fit_predict()` reads only the pre-extracted parquet during grading.
