# October-aligned Sentinel-2 canopy physiology

Source: Google Earth Engine COPERNICUS/S2_SR_HARMONIZED, 2026-09-14 inclusive to 2026-11-14 exclusive, covering all three estates. CLOUDY_PIXEL_PERCENTAGE < 80 and SCL 4 through 7 retained, matching supplied data conventions. Harmonized collection is used directly with no additional 1000 DN offset. Median SR bands B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12 remain in raw DN units. NDVI=(B8-B4)/(B8+B4); NDRE1=(B8A-B5)/(B8A+B5); NDRE2=(B8A-B6)/(B8A+B6); NDMI=(B8-B11)/(B8+B11); NBR=(B8-B12)/(B8+B12). Bandwise median composite, then indices, not median of scene indices.

Analysis grid EPSG:32648 at 10 m; native 20 m red-edge/SWIR bands retain their original information limit. Nearest-neighbour resampling is Earth Engine's default; no spatial super-resolution is claimed. NDVI/NDRE1/NDMI also have square radius-10 m median and maximum, plus centre minus radius-30 m median. Grid projection is explicit before focal filtering. Clear observation count is exported for quality auditing.

All 7,220 exact keys from public L/unlabelled CSV union are preserved in their original order. No supervised extraction or imputation is performed. Missing values and per-estate quality summaries are in audit.json; training-only imputation is required if needed. Extraction uses a service-account path from the environment but never prints or writes credentials. Run extract.py in the development environment with earthengine-api, pandas and pyarrow; cached API response chunks are optional and contain only covariates and temporary sequence numbers, absent from the final table.

Audited output for this checkpoint: 7,220 rows × 28 columns; zero NaNs and zero duplicate exact keys. The extraction script also supports rerunning directly from this dataset directory.
