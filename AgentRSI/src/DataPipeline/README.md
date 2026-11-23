# DataPipeline

Builds the sealed labels, the frozen split and the agent view, checks them for leaks, and calibrates the gate.
Run with `AgentRSI/.venv/bin/python` from any directory; paths resolve from the script location.

| Step | What it does | Needs |
|---|---|---|
| `001-BuildData.py` | Builds `.sealed/labels_master.csv`, the frozen split (from `config/split.yaml`), the agent view (`L_labelled.csv` + merged `unlabelled.csv`) and the verifier staging. `--extract-sar` rebuilds multi-date SAR locally (keeping NDVI); `--extract` rebuilds SAR + Earth Engine NDVI | raw labels; `--extract-sar` needs SAR scenes; `--extract` also needs the Earth Engine key |
| `002-CheckLeaks.py` | Sealed-label and split-leak check on the agent view, `tasks/` and `jobs/` (skips binary rasters/images for text scans; checks filenames on all files), plus mode and git-ignore checks. `--control` runs four seeded leaks that must all be caught | after 001 |
| `003-Calibration.py` | `feasibility`, `leakage` (pooled OOF, includes 80 m) and `gate` (composite T gate over resampled 80 m splits; never touches frozen T) | after 001 |

Helpers (not steps): `common.py` (paths, `config/split.yaml` loader, SAR date discovery, spatial blocks) and `features.py` (labels, multi-date SAR and S2 NDVI extraction).

## When data changes

- New SAR scene dates: drop complete `ALOS2-Subset_<estate>_<date>_Cal_Spk_TC.tif` and `..._HAlpha_Spk_TC.tif` sets into `data/SAR-scenes/`, then run `001-BuildData.py --extract-sar` and `002-CheckLeaks.py`. Dates are auto-discovered and suffixed (`HH_<date>`, `HAlpha_H_<date>`, etc.).
- New labels or optical refresh: run `001-BuildData.py --extract`, then `002-CheckLeaks.py`.
- Every run rewrites the master from the raw labels.
- The split is frozen in `config/split.yaml` (single source of truth). `001` refuses to write `config/split_assignment.csv` unless the sha256 and counts still match `config/split.yaml`. Changing that file is an epoch decision recorded in the plan, not a fix.

## Layout

- `.sealed/` (git-ignored, mode 700, files 600): master, split report, verifier staging. Nothing from here goes into `tasks/` or `jobs/`.
- `.work/` (git-ignored): `features.csv` (all discovered SAR dates + S2 NDVI), `agent_view/{L_labelled.csv,unlabelled.csv}`, `gate_results.json`, `gate_run.log`.
- `config/split.yaml`: single source of truth for epoch, block size, seed, tolerances, counts and frozen sha256.
- `config/split_assignment.csv`: frozen split with raw ids, block and part, no labels.

## Frozen values (Epoch 1)

- Split sha256 (`estate, id, part`): `26f34d4d8e0159e53985fce735adf410e0cae392d7f1cb8aded8a7490cd0e2ed`
- Block size 80 m, seed 20261119.
- Agent-visible rows have no `id` and no V/T partition: `L_labelled.csv` holds L, `unlabelled.csv` merges V + T. `(estate, Long, Lat)` identifies a tree uniquely.
