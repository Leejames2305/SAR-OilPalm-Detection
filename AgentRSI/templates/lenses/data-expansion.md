## Focus Lens: Data Expansion (Rasters & GEE)

Prioritize extracting compact, leak-free tree-level covariates for all `7,220` `(estate, Long, Lat)` rows into `/app/datasets/<name>/features.parquet` (total `/app/datasets/` under `128 MB` for automatic session carry-over):
- Local ALOS-2 GeoTIFFs in `/app/rasters/`: Yamaguchi 4-component (`Pd, Pv, Ps, Pc` in dB — cross-mask ROI support from `Cal_Spk_TC`, test both dB and linear power), multi-date `261014 - 260610` backscatter/H-Alpha/Yamaguchi deltas and ratios, multi-scale window statistics (`winstats`: min/max/quantiles/std/texture), and AirHitam `260610` T3 coherency invariants (`EPSG:4326`).
- Google Earth Engine (`agentrsi gcp-init`): Sentinel-2 red-edge/SWIR indices (`B5, B6, B7, B8A, B11, B12`, NDRE, NDMI, NBR) uniformly across all three estates (account for baseline 04.00 `+1000` offset), plus terrain/topography (SRTM/FABDEM elevation, slope, TWI/drainage), canopy height, or moisture products from `/app/prior/Plan-GEECatalogSurvey.md`.

For every new feature family, verify cross-estate Cohen's $d$ sign consistency and paired 80 m spatial-block CV lift before including it in `/app/submission/pipeline.py`, and document exact provenance and extraction scripts inside `/app/datasets/<name>/`.
