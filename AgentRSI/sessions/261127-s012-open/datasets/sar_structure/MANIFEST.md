# Native two-date SAR spatial descriptors

ALOS PALSAR-2 June10 and October14 2026, EPSG:32648, 6.43m pixels.
Exact WGS84 keys reprojected before trunk-containing pixel indexing. All within-date raster geometries checked identical. October differs in width/origin from June and is bilinearly reprojected to the June grid in linear physical power (including Yamaguchi dB-to-power-to-dB) before temporal differences and stacked patches. HAlpha is interpolated in native scalar units. Power bands HH/HV/VH/VV linear
positive finite ->10log10 dB. HAlpha H/A/alpha finite positive. Yamaguchi
Pd/Pv/Ps/Pc already dB, all finite negatives retained; cross-mask power ROI.
Broad physical clipping power[-55,15], decomposition[-65,15]dB.
3/7/11-pixel windows (~19/45/71m) median/SD/10th/90th percentile.
October-minus-June same-pixel median and SD for3/7 windows. No labels used.
All7220 exact unique keys, zeroNaNs. extract_sar.py reproduces features and
22-channel77m patch input for separate label-free self-supervised experiment.
Overlap across adjacent blocks is possible: require coarser/buffered checks.
