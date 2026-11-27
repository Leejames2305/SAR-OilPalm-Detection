# Independent terrain and historical canopy structure

Source: Google Earth Engine COPERNICUS/DEM/GLO30 (30m digital surface model) and users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1 (10m 2020 canopy-height estimate). The elevation mosaic is projected to EPSG:32648 at30m before terrain derivatives. Slope in degrees, aspect as sine/cosine, topographic position elevation minus circular focal mean60/120m, relief as circular standard deviation60/120m. Terrain is a surface model and includes vegetation effects; it is not bare-earth terrain or a measured disease indicator. Height first source band in metres, raw/contrast to40m circular mean/SD40m. No labels used. reduceRegions(mean) at10m, EPSG32648; terrain remains native30m information. No offset/harmonisation. 2020 height predates current screening and can encode stand age/management.

Exact source keys for all7220 trees; zero duplicate keys. Extraction script extract_structure.py; missing summary in extraction log and report audit.
