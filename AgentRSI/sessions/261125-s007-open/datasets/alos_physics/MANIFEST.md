# Physically scaled ALOS covariates

Source: the 18 supplied backscatter, HAlpha, and Yamaguchi GeoTIFFs for 2026-06-10 and 2026-10-14. Exact WGS84 coordinates from the union of the two public CSVs are reprojected to each raster CRS before sampling. All 7,220 unique `(estate, Long, Lat)` keys are preserved; labels and row identifiers are absent.

Backscatter is finite positive linear power transformed to 10*log10(power) per pixel. HAlpha retains original units and requires finite positive support. Yamaguchi accepts negative dB and is cross-masked by finite positive Cal_Spk_TC support reprojected with nearest-neighbor resampling. Pd/Pv/Ps/Pc fractions are computed per pixel as 10**(db/10) divided by their sum, before averaging. No invalid zero power is logged as a valid observation.

Radius 1, 2, and 4 square windows (3, 5, 9 pixels at 6.43 m) supply mean, standard deviation, 90th minus 10th percentile, and median, requiring at least five valid pixels. Matched-date changes are October minus June. Crown-neighborhood contrast is radius-1 mean minus radius-4 mean, without changing the physical sign. Features contain no labels and no fitted supervised transformations. `audit.json` records shape, NaN counts, and masking support; any missing values must be imputed using training data only. Reproduce with `python /app/datasets/alos_physics/extract.py` in the development environment (numpy, pandas, rasterio).

Audited output for this checkpoint: 7,220 rows × 611 columns; zero NaNs and zero duplicate exact keys. The extraction script also supports rerunning directly from this dataset directory.
