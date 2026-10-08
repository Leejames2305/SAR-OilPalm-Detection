# GEE catalogue survey - 12-month time-series stack (AirHitam)

**Date:** 8 October 2026

**Scope:** AirHitam only. Window: 12 months ending at the field survey date (~mid-2025 to mid-2026), with a 6-month variant as robustness check. Planned for `TimeSeries.md` POC.

**Goal**: test whether multi-temporal SAR/optical change signal adds information beyond NDVI-only LogReg (PR 0.133, NDVI+red-edge 0.145 under spatial-block CV, see `RVI_NDVI.md`).

## Tier 1 - core per-tree stack (10-20 m, usable at tree scale)

| Catalogue ID | Sensor / band | Res. | Revisit / cadence | Use |
|---|---|---|---|---|
| `COPERNICUS/S1_GRD` | C-band VV, VH (or HH, HV); asc/desc via `orbitProperties_pass` | 10 m (25/40 m variants exist) | ~6 d | Main multi-temporal SAR. dB, calibrated, terrain-corrected. |
| `COPERNICUS/S2_SR_HARMONIZED` | B2-B4, B8 (10 m); B5-B7, B8A, B11, B12 (20 m); SCL for cloud mask | 10-20 m | 5 d | Optical NDVI and red-edge time series. Harmonisation offset (+1000 after 2022-01-25, baseline 04.00+) must be applied. |

## Tier 2 - regional / coarse covariates

| Catalogue ID | Variables | Res. | Cadence | Use |
|---|---|---|---|---|
| `NASA_SMAP_SPL3SMP_E_006` | `soil_moisture_am/pm`, `vegetation_water_content_am/pm`, QA flags `retrieval_qual_flag_*`, `tb_qual_flag_*` | 9 km | daily composite (am/pm) | Seasonal soil-water covariate. Filter on QA flags. Anomaly bands experimental. |
| `NASA_SMAP_SPL4SMGP_008` | `sm_rootzone`, `sm_surface`, `sm_rootzone_wetness`, `sm_rootzone_pctl`, `land_evapotranspiration_flux`, `vegetation_greenness_fraction`, `leaf_area_index` | 11 km band table (description says 9 km EASE - reconcile) | 3-hourly | Model-based root-zone moisture; ET and LAI as covariates. Land-model only during outages (2019-06/07, 2022-08/09). |
| `NOAA_CDR_VIIRS_NDVI_V1` | `NDVI` (scale 0.0001), `QA` bitmask | 5566 m (0.05 deg) | daily | Regional NDVI cross-check. Too coarse for per-tree use. `TIMEOFDAY` has known +1-day error. |
| `LANDSAT_COMPOSITES_C02_T1_L2_8DAY_NDVI` | `NDVI` (-1 to 1) | 30 m | 8 d | Gap-filler / independent optical check. Landsat 7 excluded after 2017; Landsat 9 inclusion not stated on page. |

## Tier 3 - masks and climate forcing

| Catalogue ID | Use |
|---|---|
| `GOOGLE_DYNAMICWORLD_V1` (10 m) | Mask non-palm pixels (built-up, bare, water, forest) before sampling 3x3 windows. Agreed. |
| `ESA_WorldCover_v200` (10 m) | Secondary mask / sanity check. |
| `UCSB-CHC_CHIRPS_V3_DAILY_SAT` | Daily rainfall; seasonal covariate. Agreed. |
| `ECMWF_ERA5_LAND_DAILY_AGGR` / `_MONTHLY_AGGR` | Soil water, temperature, evapotranspiration covariates. Band names to verify in console before use. Agreed. |

## Tier 4 - not used for per-tree signal

- `JAXA_ALOS_PALSAR_YEARLY_SAR_EPOCH` (25 m, annual): prior-year baseline only.
- `LARSE_GEDI_GEDI02_A_002` (25 m footprints, sparse): canopy structure sanity check only.
- `Earth_Big_Data_GLOBAL_SEASONAL_S1_V2019_COHERENCE` / `_BACKSCATTER`: candidate change signal; resolution and years not confirmed. Verify before use.
- MODIS VI products (250 m-1 km): too coarse.


## Proposed design

- Per tree, 3x3 window at each date; S1 VV/VH and ratio, asc and desc separately; S2 NDVI and NDRE; cloud/QA masked.
- Features: per-date anomaly against estate median (removes shared seasonal moisture), seasonal delta, temporal CV, and 12-month slope.
- Covariates: SMAP root-zone, CHIRPS rainfall, ERA5 soil water; regressed out before testing disease signal.
- Evaluation: same spatial-block and corner-deployment protocol as `RVI_NDVI.md`; compare against NDVI-only (0.133) and NDVI+red-edge (0.145).
- Acceptance: a change-feature set must beat NDVI+red-edge under spatial-block CV, and the gain must survive the 6-month variant.
