# Fine-resolution historical canopy structure

Source public Meta/WRI ImageCollection projects/meta-forest-monitoring-okw37/assets/CanopyHeight,
band cover_code (canopy-height estimate in metres), accessed2026-11-27. Native
EPSG3857 pixel1.1943285669558747m (nominal1m map), retained for focal operations.
Source Maxar imagery2009-2020,80%2018-2020 according catalogue; not current disease
or current palm height. Model predicted canopy heights, trained on aerial LiDAR,
may have stand age/domain/registration errors. Tolan etal2024,
https://doi.org/10.1016/j.rse.2023.113888 ; official model repository
https://github.com/facebookresearch/HighResCanopyHeight . Attribution Meta/WRI;
source imageryMaxar; dataset CC-BY4.0 per publiccatalogue.

Features pointheight; circular3/6/12/24m mean/SD/p90; fraction exceeding2/5/10m
within6/12/24m; mean6/12minusmean24. Zero is valid ground/low height, retained.
Exact7220 originalkeys,27columns,noNaNs/duplicates. Extraction extract_highres.py,
GEEreduceRegions(mean) atnative resolution, no labels used. Metadata saved in
extraction.json. This historical independent structural context may encode
planting age/management; any gain must survive buffered/coarserholdouts.
