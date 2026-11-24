# ALOS window statistics

Extracted from `/app/rasters` using rasterio, reprojecting EPSG:4326 tree coordinates to each native raster CRS. For each raster band and radius 1 (3x3) and 2 (5x5), stored mean, std, min, max, p10, p90, and valid count. Power/HAlpha/T3 masks use finite and >0 (T3 T11 support); Yamaguchi dB values remain signed and finite. Also stored two-date differences and ratios for matched bands. Coordinates are copied exactly from the two source CSVs.
