# Topography and drainage covariates

Sources: Google Earth Engine `USGS/SRTMGL1_003` elevation and `MERIT/Hydro/v1_0_1` hydrologic layers, sampled at all 7,220 exact tree coordinates with `reduceRegions`, scale 30 m. Features are elevation, terrain slope, aspect, MERIT HAND (`hnd`), log(1+upstream area), and elevation minus focal median at 150 m and 500 m radii. `features.parquet` has 7,220 unique exact keys, 9 columns including keys, and zero NaNs. `extract.py` is the reproducible extraction script.
