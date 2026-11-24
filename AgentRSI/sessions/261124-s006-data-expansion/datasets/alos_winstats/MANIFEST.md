# ALOS window statistics

Extracted from `/app/rasters` with explicit filename mapping: `spk`=Cal_Spk_TC linear backscatter, `ha`=Cal_mat_HAlpha_Spk_TC, `yam`=Cal_mat_Yamaguchi_Spk_TC signed dB, and `t3`=AirHitam Cal_mat_Spk_TC coherency. Coordinates are reprojected from EPSG:4326 to each native raster CRS before radius-1/radius-2 sampling. Power/HAlpha use finite positive masks; T3 uses finite positive T11 support; Yamaguchi uses finite signed values. Each band stores mean, std, min, max, p10, p90, and valid count. Matched-date deltas and ratios are stored for spk, ha, and yam.
