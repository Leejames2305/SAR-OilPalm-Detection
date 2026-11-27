# Canopy proxy and local-background optical covariates

Source: COPERNICUS/S2_SR_HARMONIZED (Google Earth Engine), May15–July15 2026 and matched 2025 dates (exclusive end). SCL4–7 retained, CLOUDY_PIXEL_PERCENTAGE<80; median raw surface-reflectance DN. No extra harmonisation offset. At 10m EPSG:32648 grid, calculate NDVI=(B8-B4)/(B8+B4), NDRE1=(B8A-B5)/(B8A+B5), NDMI=(B8-B11)/(B8+B11), GRVI=(B3-B4)/(B3+B4).

Per index: original pixel, 10m circular focal median, focal-median10 minus focal-median35 (local background contrast), 20m circular spatial SD. A 3x3 10m highest-NDVI pixel proxy supplies crown B3/B4/B8 and corresponding indices. This proxy is an optical mixed-pixel maximum, not an observed canopy centroid. Native red-edge/SWIR are 20m; sampling at10m adds no native information. Median composite uses setDefaultProjection EPSG:32648 10m; no label involvement. Points sampled with reduceRegions(mean), scale10m and crs32648.

Features cover all 7,220 exact source keys, 49 columns, zero duplicates, zero NaNs. extract.py is the extraction script. Covariates include current and previous season; no labels, identifiers or blocks. The 35m background contrast aims to distinguish relative canopy condition from stand vigor. It may still encode local stand structure/spatial context and needs coordinates-only and broad-region controls.
