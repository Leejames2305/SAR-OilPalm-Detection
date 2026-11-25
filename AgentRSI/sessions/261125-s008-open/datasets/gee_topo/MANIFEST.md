# Topography and drainage covariates

Sources: Google Earth Engine `USGS/SRTMGL1_003` elevation and `MERIT/Hydro/v1_0_1` hydrologic layers, sampled at all 7,220 exact tree coordinates with `reduceRegions`, scale 30 m. Features are elevation, terrain slope, aspect, MERIT HAND (`hnd`), log(1+upstream area), and elevation minus focal median at 150 m and 500 m radii. `features.parquet` has 7,220 unique exact keys, 11 columns including keys, and zero NaNs. `extract.py` is the reproducible extraction script.

Session 261125-s007 correction: the actual table has 11 columns including the three keys, seven terrain variables, and a temporary `row` sequence from extraction. The `row` column is not a scientific covariate and must never be modeled. This table is unused by the final submission. Exact key coverage and zero NaNs were rechecked; see `/app/report/dataset_audit.json`.
