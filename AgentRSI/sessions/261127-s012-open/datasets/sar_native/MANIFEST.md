# Native multitemporal SAR self-supervision

All7220 exact original keys, zero duplicate keys/NaNs,323columns. Source/extraction
is sar_structure/extract_sar.py: June10/October14 2026 ALOS2, native6.43m, reprojection
of coordinates and October-to-June pixel grids in physical power,11polarimetric
channels perdate (power4,HAlpha3,Yamaguchi4), stacked22channels,12x12(~77m) pixels.
All trees100% support. Source Yamaguchi negative dB retained and cross-masked with
positive power ROI. Global channel scaling/clipping[-6,6] is label-free over all
patches; no labels/IDs/coordinates/split assignments uploaded to the GPU.
Two random seeds21/42 masked denoising conv autoencoders with64latent dimensions,
100epochs each,35%2x2 pixel masking,10%channel dropout,.1Gaussian noise,
SmoothL1 reconstruction; random D4 transforms. Shared11channel paired-date
VICReg encoder,100epochs; emits64dimensions June/October/difference. Model
weights, training logs, exact extraction/training/compression scripts retained.
Frozen features only at runtime, no torch/GPU/network. Downstream PCA is fit
only on each training fold. 77m footprints overlap across adjacent80m blocks:
120/160m and buffered/far-field validation mandatory. These are representations
of SAR context, not proven disease biomarkers. Citations He2022 arxiv2111.06377;
Bardes2021 arxiv2105.04906. T4 job job-0001-14ba2d,~109s training.
