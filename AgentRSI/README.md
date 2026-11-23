# AgentRSI

Autonomous-agent modelling harness for oil-palm *Ganoderma* triage across three Malaysian estates (AirHitam, Palong, Serting), inspired by RSI4EO and built on **Harbor `0.27.0`** + **rootful Podman**.

Each session gives a coding agent (`codex` or `claude-code`) a CPU-only sandbox with the multi-date ALOS-2 SAR + Sentinel-2 NDVI base tables, raw ROI-masked SAR rasters, prior research notes, and read-only Google Earth Engine access. When the session finishes, a **separate sealed verifier container** scores the agent's submitted `pipeline.py` on held-out spatial blocks (80 m) and reports:

- **V (Validation — search signal):** visible to the next session's handoff.
- **T (Certification — host-only gate):** never exposed to agents; used by `collect.py` to decide promotion (`>= +0.03` composite PR-AUC lift over incumbent, no estate degrading below `-0.03`, 20-attempt budget per epoch).

For the full design, gate calibration, and milestone status, see [`docs/AgentPlan/Plan-AgentRSI.md`](../docs/AgentPlan/Plan-AgentRSI.md).

---

## What You Need Before Running (Gitignored Files & Prerequisites)

Several files are intentionally `.gitignore`d (secrets, raw labels, heavy rasters, local tool binaries) and must exist on your machine before first run:

### 1. Host Prerequisites
- **WSL2 Linux** with **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/) installed.
- **Rootful Podman** running with socket exposed at:
  `/mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock`
- **Linux `podman` and `docker-compose` binaries** placed in `AgentRSI/.tools/bin/` (gitignored; placed first on `PATH` by the runner scripts so `podman compose` uses Linux `docker-compose`).

### 2. Required User-Provided Files (Not in Git)

| Path | What it is | Notes |
|---|---|---|
| `data/Labels/` *(repo root)* | Ground-truth estate label spreadsheets | Used once by `001-BuildData.py` to build `.sealed/labels_master.csv`. |
| `data/SAR-scenes/` *(repo root)* | ROI-masked ALOS-2 GeoTIFF scenes | Auto-discovered by date (`*_Cal_Spk_TC.tif`, `*_HAlpha_Spk_TC.tif`, plus raw rasters exposed to agents). |
| `AgentRSI/.env` | Runtime secrets & gateway settings | Copy from `AgentRSI/.env.example` and fill in `BASE_URL`, `API_KEY`, `GCP_PROJECT`, `GCP_SA_KEY_PATH`, and `DAILY_SPEND_CAP_USD`. |
| `AgentRSI/.secrets/Keys_SAEarthEngine.json` | Least-privilege **GEE-only** GCP service-account key | Must have Earth Engine viewer access **only** — no GCS bucket read access (protects truth labels in `sar-oilpalm`). Mounted read-only into agent containers at `/run/secrets/gcp-sa.json`. |

> **Never commit or paste** `.env`, `.secrets/`, or `.sealed/` contents into git, task folders, or chat logs.

### 3. Auto-Generated Local Directories (Gitignored)
These are created automatically by the setup and run scripts:
- `.venv/`, `.uv-cache/` — Python virtual environment (`harbor==0.27.0` + pinned stack).
- `.sealed/` *(mode `700`, files `600`)* — sealed master labels and verifier staging (never copied into `tasks/` or `jobs/`).
- `.work/` — extracted `features.csv` and `agent_view/{L_labelled.csv,unlabelled.csv}`.
- `tasks/`, `jobs/`, `dashboard/` — generated Harbor tasks, raw trial logs, and local dashboard output.

---

## How to Use

Run commands from the **repository root** (`/home/lee/GitRepo/SAR-OilPalm-Detection` or your local clone).

### First-Time Setup (or After Data / Verifier Changes)

```bash
# 1. Sync the pinned Python environment (harbor==0.27.0)
uv --directory AgentRSI sync

# 2. Build sealed labels, frozen 80 m split, and agent view (see src/DataPipeline/README.md)
#    Use --extract on a fresh machine (needs SAR scenes + GEE key),
#    --extract-sar to rebuild SAR locally, or omit flags if .work/features.csv exists.
AgentRSI/.venv/bin/python AgentRSI/src/DataPipeline/001-BuildData.py --extract
AgentRSI/.venv/bin/python AgentRSI/src/DataPipeline/002-CheckLeaks.py

# 3. Build the two local Podman base images (rootful Podman must be running)
AgentRSI/.venv/bin/python AgentRSI/scripts/build_verifier_base.py   # localhost/agentrsi-sealed:epoch1
AgentRSI/.venv/bin/python AgentRSI/scripts/build_agent_base.py      # localhost/agentrsi-agent-base:latest
```

### Running a Session (Daily Loop)

```bash
# 1. Build a Harbor task folder (runs allowlist copy + leak check automatically)
#    Available --focus lenses (templates/lenses/): open | literature | data-expansion | ablation
AgentRSI/.venv/bin/python AgentRSI/scripts/build_task.py \
  --session-id s001 \
  --focus open \
  --parent s000-dry

# 2. (Optional) Static lint + dry run before spending API budget
AgentRSI/.venv/bin/harbor lint AgentRSI/tasks/s001
AgentRSI/.venv/bin/python AgentRSI/scripts/run_session.py --session-id s001 --dry-run

# 3. Run the session (agent sandbox + separate sealed verifier)
#    Defaults come from config/session.yaml; flags below are optional overrides.
AgentRSI/.venv/bin/python AgentRSI/scripts/run_session.py \
  --session-id s001 \
  --agent codex \
  --model gpt-6.1-terra

# 4. Collect outputs, run secret/smell scans, evaluate T gate, and update lineage
#    Add --dry-session when testing to skip spending a T certification attempt or promoting.
AgentRSI/.venv/bin/python AgentRSI/scripts/collect.py \
  --session-id s001 \
  --parent starter
```

### Regrading & Verification Utilities

```bash
# Re-score a recorded Harbor job against a (possibly updated) verifier task
AgentRSI/.venv/bin/python AgentRSI/scripts/regrade.py \
  AgentRSI/jobs/<session_id>/<timestamp> \
  --task-dir AgentRSI/tasks/<session_id>

# Inspect raw Harbor trial logs & agent trajectories in Harbor's viewer
AgentRSI/.venv/bin/harbor view AgentRSI/jobs/<session_id>

# Run all host unit tests (verifier, builder, collector, spend guard)
AgentRSI/.venv/bin/python -m unittest discover -s AgentRSI/tests -v

# Standalone leak audit across generated trees
AgentRSI/.venv/bin/python AgentRSI/src/DataPipeline/002-CheckLeaks.py \
  --scan AgentRSI/tasks AgentRSI/jobs AgentRSI/sessions
```

---

## Key Folder Structure

```text
AgentRSI/
├── config/                     # Versioned configuration (edit here to tune runs)
│   ├── session.yaml            #   Default agent, model, timeouts, CPU/RAM limits, focus lens
│   ├── verifier.yaml           #   Sealed image tag, estate weights, T gate thresholds
│   ├── split.yaml              #   FROZEN 80 m spatial split contract (change = new epoch)
│   └── split_assignment.csv    #   Host-only tree-to-block/split map (no labels)
│
├── sessions/<session_id>/      # ★ START HERE TO READ RESULTS (committed summary per session)
│   ├── REPORT.md               #   Agent's full experiment write-up
│   ├── HANDOFF.md              #   Agent's handoff notes for the next session
│   ├── reward.json             #   Verifier flat numeric scores (composite V lift, runtime)
│   ├── v_report.json           #   Per-estate V breakdown vs incumbent & coords-only baseline
│   ├── certification.json      #   HOST-ONLY T gate decision & paired bootstrap CI
│   ├── meta.json               #   Session metadata, cost, smell scan, promotion outcome
│   └── submission/             #   Submitted pipeline.py + pinned requirements.txt
│
├── state/                      # ★ TRACK OVERALL PROGRESS & GUARDS
│   ├── lineage.json            #   Current incumbent, epoch, T attempts used, daily spend, kill_switch
│   └── events.jsonl            #   Append-only audit log of starts, ends, promotions, cap blocks
│
├── scripts/                    # CLI entry points (build_*, run_session, collect, regrade)
├── src/
│   ├── DataPipeline/           # Sealed split, feature extraction, leak checker, gate calibration
│   └── libbuilder/             # Task builder, Harbor runner, secret stager, host collector
├── verifier/                   # Sealed grading library baked into localhost/agentrsi-sealed:epoch1
├── starter/                    # Baseline incumbent pipeline.py & requirements.txt
├── templates/                  # Instruction base, lenses/, task.toml/Dockerfiles, in-container CLI
├── tests/                      # Unit tests for verifier, builder, and collector
│
├── tasks/<session_id>/         # (gitignored) Generated Harbor task directories
├── jobs/<session_id>/          # (gitignored) Raw Harbor job outputs, logs, and trajectories
├── .sealed/                    # (gitignored, 0700) Master labels & verifier staging
├── .work/                      # (gitignored) Extracted features & agent_view CSVs
├── .secrets/                   # (gitignored) GEE-only service-account JSON
├── .tools/bin/                 # (gitignored) Linux podman & docker-compose binaries
└── .env                        # (gitignored) Gateway URL, API key, GCP project, daily spend cap
```

---

## Operational Notes

- **Kill switch & spend cap:** Set `"kill_switch": true` in `state/lineage.json` to block new sessions immediately, or lower `DAILY_SPEND_CAP_USD` in `.env`.
- **Agent gateway differences:** `run_session.py` automatically normalizes `BASE_URL` and flags per agent — `codex` uses `.../v1` (`openai_responses`, `OPENAI_API_KEY`) and does not accept `max_turns`/`max_budget_usd`; `claude-code` strips `/v1` (`anthropic_messages`, `ANTHROPIC_AUTH_TOKEN`) and honors both.
- **Harbor `harbor check` on Podman:** Upstream Harbor (`0.25.0`–`0.27.0`) hits a `crun: chdir to /app: No such file or directory` error on `harbor check -e podman`. Use `harbor lint` + `002-CheckLeaks.py` to validate tasks instead.
- **Adding new SAR dates:** Drop complete ALOS-2 scenes into `data/SAR-scenes/`, re-run `001-BuildData.py --extract-sar` + `002-CheckLeaks.py`, and rebuild `localhost/agentrsi-sealed:epoch1` via `scripts/build_verifier_base.py`.
