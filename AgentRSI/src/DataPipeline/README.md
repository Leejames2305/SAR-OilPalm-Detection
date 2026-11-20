# DataPipeline

Builds the sealed labels, the frozen split and the agent view, checks them for leaks, and calibrates the gate.
Run with `AgentRSI/.venv/bin/python` from any directory; paths resolve from the script location.

| Step | What it does | Needs |
|---|---|---|
| `001-BuildData.py` | Builds `.sealed/labels_master.csv`, the frozen split, the agent view and the verifier staging. `--extract` first rebuilds the features | raw labels; `--extract` also needs the SAR scenes and the Earth Engine key |
| `002-CheckLeaks.py` | Sealed-ID check on the agent view, `tasks/` and `jobs/`, plus mode and git-ignore checks. `--control` runs a seeded leak that must be caught | after 001 |
| `003-Calibration.py` | `feasibility`, `leakage` and `gate` (composite T gate calibration) | after 001 |

Helpers (not steps): `common.py` (paths, constants, spatial blocks) and `features.py` (labels, SAR and NDVI extraction).

## When data changes

- New scene files or labels: run `001-BuildData.py --extract`, then `002-CheckLeaks.py`.
- `--extract` replaces `.work/features.csv`. The agent view is rebuilt from it in the same run.
- Every run rewrites the master from the raw labels.
- The split is frozen. `001` refuses to write it unless the sha256 still matches `SPLIT_SHA256` in `common.py`. Changing that value is a decision to record in the plan, not a fix.

## Layout

- `.sealed/` (git-ignored, mode 700, files 600): master, split report, verifier staging. Nothing from here goes into `tasks/` or `jobs/`.
- `.work/` (git-ignored): features, agent views, gate results.
- `config/split_assignment.csv`: frozen split with raw ids and part, no labels.

## Frozen values

- Split sha256 (estate, id, part): `26f34d4d8e0159e53985fce735adf410e0cae392d7f1cb8aded8a7490cd0e2ed`
- Block size 80 m, seed 20261119.
- Agent-visible rows have no id. Estate and coordinates identify a tree (unique in the master).
