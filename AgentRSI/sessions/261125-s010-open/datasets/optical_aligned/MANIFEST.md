# Corrected optical registration features

Corrected trunk-containing pixel, fixed subtraction of half a pixel from sampleRectangle centre origin.

Source and preprocessing identical to optical_texture: cloud-controlled S2 April15–August15 2026, fallback March15–September15, SCL4 and cs_cdf>=.65; native10m EPSG32648 with 20m red-edge not sharpened. Four covariates: NDVI ring10–30m mean minus centre, NDVI SD20, central4x4 gradient magnitude, NDRE med10-med40. 7220x7, exact original coordinate keys, zero missing/duplicate keys. No labels used. registration_texture.py builds from label-free chip mosaic rasters; extract_optical_chips.py reproduces source mosaics.
