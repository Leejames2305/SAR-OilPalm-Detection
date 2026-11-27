# Native Sentinel-2 crown-mixture experiment

Source COPERNICUS/S2_SR_HARMONIZED Apr15-Aug15 2026, SCL4-7 and CloudScore+
cs_cdf>=.65; same-window SCL4-7 fallback. Median B2/B3/B4/B8/B5/B8A/B11/B12,
DN/10000, no extra offset. Native10m/20m information, reprojectEPSG32648 10m.
sampleRectangle pixel-center coordinates (.5,.5) converted to GeoTIFF corners;
boundary-coordinate audit included. Exact coordinate keys reprojected before
bilinear point sampling. No labels used. All7220unique keys/zeroNaNs.

Gaussian5m crown observation model from ALL supplied tree coordinates perestate,
up to8crowns within15m per10m pixel, normalized weights; pixels nearest crown<8m.
Solve (W.T W+lambda I)(f-f0)=W.T(p-Wf0), f0=bilinear pixel spectrum. Two fixed
lambda .25/1 variants. No learned disease scores or selected label-dependent
hyperparameters. Clip reconstructed spectra .0001..1. Compute NDVI,NDRE,NDMI,
NBR,NIRv,GRVI for point/deblur/stable spectra and local contrast to9nearesttrees.
20m bands remain20m information. This underdetermined inverse problem does NOT
create native per-crown resolution, may amplify geolocation/mosaic noise, and
assumes catalogue completeness/Gaussian mixing. Footprint overlaps: coarser and
buffered validation required. Reproduction scripts and inversion diagnostics
retained. Source mosaic rasters remain research assets, not runtime dependencies.
