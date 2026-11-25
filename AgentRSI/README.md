# AgentRSI

Autonomous-agent modelling harness for oil-palm *Ganoderma* triage across three Malaysian estates (AirHitam, Palong, Serting), inspired by RSI4EO and built on **Harbor `0.27.0`** + **rootful Podman**.

Each default session runs for **4 hours** (`14400 s`, `$10` cost/daily spend cap) and gives a coding agent (`codex` or `claude-code`) a CPU-only sandbox with the multi-date ALOS-2 SAR + Sentinel-2 NDVI base tables, raw ROI-masked SAR rasters, prior research notes, carried-over derived datasets, read-only Google Earth Engine access, and two MCP extensions enabled by default:

- **M7 (`agentrsi-verifier`):** a multi-container sealed sidecar exposing a query-budgeted, noisy validation-split (`V`) scoring tool during the run.
- **M8 (`agentrsi-colab`):** a hardened host-side WSL MCP broker wrapping `google-colab-cli` for on-demand Colab GPU offload (`T4`/`L4` allowed; `A100` restricted unless explicitly opted in) without exposing host Google OAuth/ADC tokens to containers.

When the session finishes, a **separate sealed verifier container** scores the agent's submitted `pipeline.py` (CPU-only, offline) on held-out 80 m spatial blocks against the current promoted incumbent and reports:

- **V (Validation — search signal):** visible to the next session's handoff.
- **T (Certification — host-only gate):** never exposed to agents; used by `collect.py` to decide promotion (paired block-bootstrap 5th percentile `> 0`, composite PR-AUC lift `>= +0.03` over incumbent, at least **2 of 3 estates** improved, no estate degrading below `-0.03`, 20-attempt budget per epoch).

For the full design, gate calibration, and milestone status, see [`docs/AgentPlan/Plan-AgentRSI.md`](../docs/AgentPlan/Plan-AgentRSI.md).

---

## What You Need Before Running (Gitignored Files & Prerequisites)

Several files are intentionally `.gitignore`d (secrets, raw labels, heavy rasters, local tool binaries) and must exist on your machine before first run:

### 1. Host Prerequisites
- **WSL2 Linux** with **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/) installed.
- **Rootful Podman** running with socket exposed at:
  `/mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock`
- **Linux `podman` and `docker-compose` binaries** placed in `AgentRSI/.tools/bin/` (gitignored; placed first on `PATH` by the runner scripts so `podman compose` uses Linux `docker-compose`).
- *(Required for default sessions with M8 Colab GPU offload; or build tasks with `--disable-colab`)* **`google-colab-cli`** installed and authenticated on the **WSL host only**:
  ```bash
  uv tool install google-colab-cli
  colab auth
  AgentRSI/.venv/bin/python AgentRSI/scripts/colab_broker.py --check-auth
  ```
  > Keep `google-colab-cli` OAuth/ADC credentials strictly on the WSL host — never mount `~/.config/colab-cli` or gcloud ADC into any container.

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
- `.venv/`, `.uv-cache/` — Python virtual environment (`harbor==0.27.0`, `mcp==2.3.0`, pinned scientific stack).
- `.sealed/` *(mode `700`, files `600`)* — sealed master labels and verifier staging (never copied into `tasks/` or `jobs/`).
- `.work/` — extracted `features.csv` and `agent_view/{L_labelled.csv,unlabelled.csv}`.
- `tasks/`, `jobs/`, `dashboard/` — generated Harbor tasks, raw trial logs, and local dashboard output.

---

## How to Use

Run commands from the **repository root** (`/home/lee/GitRepo/SAR-OilPalm-Detection` or your local clone).

### First-Time Setup (or After Data / Verifier / Image Changes)

```bash
# 1. Sync the pinned Python environment (harbor==0.27.0)
uv --directory AgentRSI sync

# 2. Build sealed labels, frozen 80 m split, and agent view (see src/DataPipeline/README.md)
#    Use --extract on a fresh machine (needs SAR scenes + GEE key),
#    --extract-sar to rebuild SAR locally, or omit flags if .work/features.csv exists.
AgentRSI/.venv/bin/python AgentRSI/src/DataPipeline/001-BuildData.py --extract
AgentRSI/.venv/bin/python AgentRSI/src/DataPipeline/002-CheckLeaks.py

# 3. Build the two local Podman base images (rootful Podman must be running)
#    - agentrsi-sealed:epoch1 bakes root-only /sealed, unprivileged `runner`, and isolated
#      /opt/agentrsi/sidecar-venv (for M7) while keeping /usr/local/bin/python clean for v1 grading.
#    - agentrsi-agent-base:latest prebakes geo/ML libs, Node 22, codex, claude, ripgrep,
#      `agentrsi` CLI, and `/usr/local/bin/agentrsi_mcp_relay.py` (for M8).
AgentRSI/.venv/bin/python AgentRSI/scripts/build_verifier_base.py
AgentRSI/.venv/bin/python AgentRSI/scripts/build_agent_base.py
```

### Running a Session (Manual Loop)

Session IDs use **`YYMMDD-sXXX-<lens>`** (e.g., `261124-s001-open`, `261124-s006-data-expansion`) so directories sort chronologically across epochs. `build_task.py` infers `--focus` automatically from a matching `-<lens>` suffix and supports `--next` for auto-numbering.

```bash
# 1. Build a Harbor task folder (runs allowlist copy + placeholder/T-leak canary + leak check)
#    Defaults from config/session.yaml: 4 h timeout (14400 s), M7 in-run verifier ON,
#    M8 Colab GPU broker ON (T4/L4 allowed, A100 restricted), $10 cost/daily spend cap.
#    Available lenses (templates/lenses/): open | literature | data-expansion | ablation
#    Optional flags:
#      --datasets-from <session_id>         carry derived datasets from a specific prior session
#      --timeout <sec>                      override agent wall-clock timeout (default: 14400)
#      --agent-cpus / --agent-memory-mb / --verifier-cpus / --verifier-memory-mb
#      --disable-inrun-verifier             disable M7 noisy-V verifier sidecar MCP
#      --disable-colab                      disable M8 host Colab GPU MCP broker
#      --colab-allow-a100                   opt into A100 GPU selection (restricted by default)
AgentRSI/.venv/bin/python AgentRSI/scripts/build_task.py \
  --next \
  --focus open \
  --parent 261124-s006-data-expansion

# 2. (Optional) Dry run before spending API budget (add --colab-mock to skip live Colab auth)
AgentRSI/.venv/bin/python AgentRSI/scripts/run_session.py \
  --session-id <session_id> \
  --dry-run

# 3. Run the session (agent sandbox + M7 sidecar & M8 host broker + separate sealed verifier)
#    Defaults come from config/session.yaml; flags below are optional overrides.
AgentRSI/.venv/bin/python AgentRSI/scripts/run_session.py \
  --session-id <session_id> \
  --agent codex \
  --model gpt-6.1-terra

# 4. Collect outputs, run secret/smell scans, evaluate T gate, archive datasets/queries, refresh dashboard
#    Add --dry-session when testing to skip spending a T certification attempt or promoting.
#    Note: collect concurrent sessions sequentially (lineage.json is not file-locked).
AgentRSI/.venv/bin/python AgentRSI/scripts/collect.py \
  --session-id <session_id> \
  --parent 261124-s006-data-expansion
```

### MCP Extensions (M7 & M8 — Enabled by Default)

Both MCP servers are **enabled by default** in `config/session.yaml` (alongside the 4 h timeout and `$10` cost cap) and can be disabled per task via `--disable-inrun-verifier` / `--disable-colab` (or toggled in `config/session.yaml`):

- **M7 — In-Run Verifier v2 (`agentrsi-verifier`):**
  - Runs as a Compose sidecar (`verifier-sidecar`, built from `localhost/agentrsi-sealed:epoch1` + current promoted incumbent) at `http://verifier-sidecar:8765/mcp` (Streamable HTTP) authenticated with a per-session ephemeral `AGENTRSI_VERIFIER_MCP_TOKEN` held only in process memory and scrubbed from trial logs.
  - Exposes `score_v`, `verifier_budget`, and `verifier_health`. Loads `V`-only labels (strips all `T` rows immediately), enforces a query cap (`15` default in `config/verifier.yaml`), adds content-hash-cached CSPRNG Gaussian noise ($\sigma = 0.005$ default, preventing noise-averaging by resubmission), and accepts inline `pipeline_code` or larger staged submissions/datasets under the Compose `/shared` volume.
  - Harbor downloads `/var/log/agentrsi-verifier/queries.jsonl` from the sidecar; `collect.py` archives it as `sessions/<session_id>/sidecar_queries.jsonl` and records `inrun_queries_used` in `meta.json`.
- **M8 — Host Colab GPU Offload (`agentrsi-colab`):**
  - `run_session.py` starts `ColabBrokerServer` on the WSL host over a per-session Unix domain socket (`/mnt/wsl/agentrsi/colab-<session_id>.sock` -> `/run/agentrsi/colab-mcp.sock`) plus a jailed exchange directory (`/mnt/wsl/agentrsi/colab-io-<session_id>` -> `/run/agentrsi/colab-io/{in,out}`). Inside the container, `/usr/local/bin/agentrsi_mcp_relay.py` bridges MCP `stdio` to the socket.
  - Exposes only `colab_run`, `colab_status`, `colab_fetch`, and `colab_stop`. Blocks `google.colab.auth`, `drive.mount`, `gcloud`/`gsutil`, `sar-oilpalm`, repo identifiers, metadata IPs, unsafe `pip` flags, path traversal, and symlinks; enforces GPU allowlist (`T4`, `L4`; `A100` opt-in), max 1 active VM, per-job & per-session GPU wall-clock caps, idle auto-teardown, and unconditional `colab stop` + socket/IO cleanup on exit (`finally:`).
  - Final `/app/submission/pipeline.py` still grades **CPU-only and offline** — GPU outputs must be saved as reproducible artifacts/features under `/app/datasets/<name>/`.

### Dashboard, Regrading & Verification Utilities

```bash
# Regenerate and exact-diff verify the offline dashboard (also auto-run by collect.py)
AgentRSI/.venv/bin/python AgentRSI/scripts/dashboard.py --verify

# Check host google-colab-cli readiness for M8 (or run standalone broker in mock mode)
AgentRSI/.venv/bin/python AgentRSI/scripts/colab_broker.py --check-auth

# Re-score a recorded Harbor job against a (possibly updated) verifier task
AgentRSI/.venv/bin/python AgentRSI/scripts/regrade.py \
  AgentRSI/jobs/<session_id>/<timestamp> \
  --task-dir AgentRSI/tasks/<session_id>

# Inspect raw Harbor trial logs & agent trajectories in Harbor's viewer
AgentRSI/.venv/bin/harbor view AgentRSI/jobs/<session_id>

# Run all 55 host unit tests (verifier, M7 sidecar, M8 broker, builder/collector, MCP wiring, dashboard)
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
│   ├── session.yaml            #   Default agent, model, timeouts, CPU/RAM limits, lens, M7/M8 defaults
│   ├── verifier.yaml           #   Sealed image tag, estate weights, T gate thresholds, M7 inrun settings
│   ├── split.yaml              #   FROZEN 80 m spatial split contract (change = new epoch)
│   └── split_assignment.csv    #   Host-only tree-to-block/split map (no labels)
│
├── sessions/<session_id>/      # ★ START HERE TO READ RESULTS (committed summary per session)
│   ├── REPORT.md               #   Agent's full experiment write-up
│   ├── HANDOFF.md              #   Agent's handoff notes for the next session
│   ├── reward.json             #   Verifier flat numeric scores (composite V lift, runtime)
│   ├── v_report.json           #   Per-estate V breakdown vs incumbent & coords-only baseline
│   ├── certification.json      #   HOST-ONLY T gate decision & paired bootstrap CI
│   ├── meta.json               #   Session metadata, cost, smell scan, inrun_queries_used, promotion
│   ├── sidecar_queries.jsonl   #   (M7 runs) V-only in-run query audit log collected from sidecar
│   ├── datasets/               #   (optional) Validated derived covariate stacks (features.parquet)
│   └── submission/             #   Submitted pipeline.py + pinned requirements.txt
│
├── state/                      # ★ TRACK OVERALL PROGRESS & GUARDS
│   ├── lineage.json            #   Current incumbent, epoch, T attempts used, daily spend, kill_switch
│   └── events.jsonl            #   Append-only audit log (sessions, promotions, caps, Colab VM lifecycle)
│
├── dashboard/index.html        # ★ OFFLINE DASHBOARD (gitignored, auto-refreshed by collect.py)
├── scripts/                    # Thin CLIs (build_*, run_session, collect, colab_broker, dashboard, regrade)
├── src/
│   ├── DataPipeline/           # Sealed split, feature extraction, leak checker, gate calibration
│   ├── libbuilder/             # Task builder, Harbor runner, secret stager, collector, dashboard
│   └── colab_broker/           # M8 host-side Colab GPU MCP broker (config, guardrails, backend, server)
├── verifier/                   # Sealed grading library + M7 sidecar (`inrun_server.py`)
├── starter/                    # Session-0 baseline pipeline.py & requirements.txt
├── templates/                  # Instruction base, lenses/, task.toml/Dockerfiles/compose, agentrsi CLI & MCP relay
├── tests/                      # 55 host unit tests (verifier, M7, M8, builder/collector, MCP wiring, dashboard)
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

- **Kill switch & spend cap:** Default cost limits are `$10` (`max_budget_usd: 10.0` in `config/session.yaml` for `claude-code`, and `DAILY_SPEND_CAP_USD=10` in `.env` enforced before launch by `run_session.py`). Set `"kill_switch": true` in `state/lineage.json` to block new sessions immediately, or lower `DAILY_SPEND_CAP_USD` in `.env`.
- **Agent gateway differences:** `run_session.py` automatically normalizes `BASE_URL` and flags per agent — `codex` uses `.../v1` (`openai_responses`, `OPENAI_API_KEY`) and does not accept `max_turns`/`max_budget_usd`; `claude-code` strips `/v1` (`anthropic_messages`, `ANTHROPIC_AUTH_TOKEN`) and honors both. `run_session.py` also passes `--yes` so Harbor does not pause on interactive MCP env-header prompts.
- **Harbor linting on Podman-only hosts:**
  - Default tasks include `environment/docker-compose.yaml` (M7 sidecar); because `harbor lint` invokes the `docker compose` CLI plugin (absent on Podman-only hosts), validate Compose overlays via `AgentRSI/.tools/bin/docker-compose config` plus `002-CheckLeaks.py`. `harbor lint` passes directly on tasks built with `--disable-inrun-verifier`.
  - Upstream Harbor (`0.25.0`–`0.27.0`) still hits `crun: chdir to /app` on `harbor check -e podman`; rely on `harbor lint` (non-Compose) / `docker-compose config` + `002-CheckLeaks.py`.
- **Concurrent sessions:** Use per-task resource overrides (`--agent-cpus`, `--agent-memory-mb`, `--verifier-cpus`, `--verifier-memory-mb`) to stay within Podman's 12-CPU / 15.2 GiB envelope (note each M7 `verifier-sidecar` adds `2` CPUs / `2048 MB`; disable with `--disable-inrun-verifier` / `--disable-colab` if packing several parallel runs), and run `collect.py` sequentially after sessions finish.
- **Adding new SAR dates:** Drop complete ALOS-2 scenes into `data/SAR-scenes/`, re-run `001-BuildData.py --extract-sar` + `002-CheckLeaks.py`, and rebuild `localhost/agentrsi-sealed:epoch1` via `scripts/build_verifier_base.py`.
