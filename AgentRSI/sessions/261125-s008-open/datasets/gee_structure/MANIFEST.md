# Independent historical canopy structure

ETH Global Canopy Height 2020 at 10m, Earth Engine asset users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1 and its SD counterpart. Height is a learned optical/GEDI estimate, not direct per-palm height; 2020 is six years before current imagery/labels and may precede replanting. Label-free covariates: height, uncertainty, height minus 50m focal median, 30m spatial height SD. Source: Lang et al., A high-resolution canopy height model of the Earth, Nature Ecology & Evolution (2023), https://doi.org/10.1038/s41559-023-02206-6 .

Sampled via reduceRegions(mean), EPSG:32648, scale10m at all 7,220 original tree coordinates. Exact keys preserved. 7 columns, no NaNs or duplicate keys. extract.py reproduces extraction. Intended as stand-structure confounding/interaction covariates, not 2026 direct disease measurements.
