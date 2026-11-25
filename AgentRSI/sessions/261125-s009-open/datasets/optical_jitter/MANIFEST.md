# Corrected optical registration features

Uniform mean of texture values computed after offsets (+/-5m,+/-5m) in pixel indexing about the corrected trunk; fixed registration uncertainty integration, no label-dependent weights.

Source and preprocessing identical to optical_texture: cloud-controlled S2 April15–August15 2026, fallback March15–September15, SCL4 and cs_cdf>=.65; native10m EPSG32648 with 20m red-edge not sharpened. Four covariates: NDVI ring10–30m mean minus centre, NDVI SD20, central4x4 gradient magnitude, NDRE med10-med40. 7220x7, exact original coordinate keys, zero missing/duplicate keys. No labels used. registration_texture.py builds from label-free chip mosaic rasters; extract_optical_chips.py reproduces source mosaics.
