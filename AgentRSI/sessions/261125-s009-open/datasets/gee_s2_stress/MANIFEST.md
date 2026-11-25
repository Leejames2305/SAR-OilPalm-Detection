# Sentinel-2 stress covariates

Source: Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED`, sampled at tree coordinates for all 7,220 rows in `/app/data/L_labelled.csv` union `/app/data/unlabelled.csv`. Current-season window is 2026-05-15 through 2026-07-15; matched prior-season window is 2025-05-15 through 2025-07-15. Each estate is reduced uniformly in chunks with `ee.Image.reduceRegions`, scale 20 m.

Images are filtered to `CLOUDY_PIXEL_PERCENTAGE < 80` and masked to SCL classes 4–7. Median composites use B2–B8, B8A, B11, B12. Derived indices are NDRE1=(B8A-B5)/(B8A+B5), NDRE2=(B8A-B6)/(B8A+B6), NDMI=(B8-B11)/(B8+B11), NBR=(B8-B12)/(B8+B12), CI_rededge=B8A/B5-1, MSI=B11/B8, and EVI=2.5(B8-B4)/(B8+6B4-7.5B2+10000). `delta_*` columns are current minus prior-season values.

`features.parquet` has 7,220 unique exact `(estate, Long, Lat)` keys, 54 columns including keys, zero NaNs, and is ~1 MB. `extract.py` is the reproducible extraction script.
