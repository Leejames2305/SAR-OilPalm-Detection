# AgentRSI Plan

**Version:** 1.3 (17 Nov 2026) | **Phase:** M0 nearly closed, M1 next | **Paper:** `docs/Reference/...RSI4EO_2610.18845v2.pdf`

**Goal.** Turn this project into an RSI4EO-style loop. A coding agent builds the oil-palm triage model in a sandbox, a sealed verifier scores it, and each day's session starts from what the last one found. Humans design the harness and verifier. Agents build the models.

**Where we are.** M0 passes every check except the two agent trials, which wait on your gateway key (see Progress). The design decisions are made. What is left open is calibration work for M1 (block size, gate values) and the GCP placeholders (see Decisions).

Evidence tags: **[V]** read in Harbor docs or source | **[T]** tested on this machine | **[S]** simulation on the real AirHitam tables | **[?]** unverified.

---

## Overview

| RSI4EO concept | Here |
|---|---|
| Harness H (data, libs, budget) | Per-tree SAR tables and ROI-masked ALOS-2 rasters for 3 estates, prior POC reports, the incumbent pipeline, GEE/GCS access, a Python geo/ML stack. **CPU only.** |
| Verifier V (sealed, scores only) | Sealed held-out blocks for all three estates, each with a validation part (V) and a certification part (T). Returns scores, never labels. |
| Pipeline pi | `submission/`: `fit_predict(labeled, unlabeled, ctx) -> risk scores`, a **training recipe** (the product trains on about 1000 labels in a new field and predicts the rest). |
| Few-label task | Same shape: a labelled slice per estate, the rest is the "unlabeled pool". |
| Recursion inside a task | Agent-driven within the 8 h session. Pseudo-labelling on the unlabeled pool is allowed, nothing is prescribed. |
| **Not in the paper:** recursion across tasks | Our addition. Session N+1 is generated from session N's handoff and the incumbent. |
| 12 h, 4 x GB200, offline | 8 h, CPU-only, **public internet**, GEE/GCS access. |

CPU-only is fine: earlier POCs show nonlinear and deep models lose to linear ones at n=160 positives. Ideas and verification are the bottleneck, not compute.

---

## Decisions

### Made

| Area | Decision | Basis |
|---|---|---|
| **Label trust** | All three estates are treated identically: same split rule, same reward weight, same certification. No provenance field. | Your call (17 Nov): same screening methodology and tools at every site. |
| **Reward** | Mean over the three estates of `lift = pr_auc - prevalence` on V. Equal weights, kept in `verifier.yaml`. The agent sees the per-estate breakdown. | Follows from label trust: no estate is discounted. |
| **Promotion** | One **composite T test** on the same equal weights, plus a per-estate non-degradation check. One nominee per session. Effect floor `d >= 0.03`, T-attempt budget 20. | Your call (17 Nov). Floor and budget came from the AirHitam-only simulation, so M1 re-checks them on all three estates. |
| **Dashboard** | A static HTML page inside `AgentRSI/`: `AgentRSI/scripts/dashboard.py` writes `AgentRSI/dashboard/index.html`. Numbers come from verifier and collector files only. The report supplies event context only. | Your call (17 Nov). Design in Design › Dashboard. |
| **Verifier phases** | v1 grades once at session end. v2 adds an in-run sealed query tool with a query budget. | |
| **Agents** | Claude Code and Codex through a third-party gateway. Agent, model, base URL and key name are run-time parameters. Current default: Codex, `gpt-6.1-terra`, OpenAI Responses. | |
| **Budget and network** | 8 h agent timeout. Agent has public internet. **Verifier network is fully public** (throwaway container). Label protection is the unprivileged subprocess plus root-only label files, not the network. | Your call (11 Nov). |
| **Packages** | The agent ships a pinned `requirements.txt` and the verifier installs it from PyPI. No allowlist, no approval flow. | Your call (11 Nov). |
| **Runtime** | Rootful Podman from WSL, Docker Desktop as fallback. Harbor pinned at 0.25.0. | Rootless cannot enforce CPU or memory limits. |
| **Focus** | Open-ended research, explicitly including data expansion. Optional lenses (`literature`, `data-expansion`, `ablation`) bias the instruction. | |
| **Data egress** | ROI-masked ALOS-2, tree coordinates and labels go to your inference provider. The GCS/GEE service-account key is passed to the agent. | Your call (11 Nov). |
| **Layout** | Everything AgentRSI builds, runs or generates lives under `AgentRSI/`, the dashboard included. Generated tasks are in `AgentRSI/tasks/<id>/`. Harbor's file is `instruction.md` (singular) **[V]**. | |
| **Split sizes** | Proportional: each estate's scored trees go to L / V / T at about 40 / 30 / 30 %, so L is about 1004 / 868 / 667 trees (AirHitam / Palong / Serting). It scales to any estate size. | Your call (17 Nov). Accepts an L below the "about 1000 labels" product shape for the two smaller estates. |
| **Middle class** | **Visible but unscored.** Middle trees go through the same block split as every other tree. In L the agent sees them labelled `Middle`. In V and T they sit unmarked in the unlabeled pool, as they would in a new field. They are never in the positive class and never in any metric. The starter pipeline and the baselines train without them (the POC convention). | Your call (17 Nov): "exclude first", then "visible but unscored". Whether Middle helps is still open (below). Evidence in Observations. |
| **Gateway URL** | The path suffix depends on the API format: `BASE_URL` ends in `/v1` for OpenAI Responses (Codex) and has no `/v1` for Anthropic Messages (Claude Code). `run_session.py` must therefore pick the URL by agent. | Your gateway (17 Nov). The Codex trial confirms Harbor passes it through correctly. |

### Open

| Item | Current default | Resolved by |
|---|---|---|
| **Block size** for the bootstrap | Not chosen. 220 m is too coarse for Palong and Serting (see Observations). | M1 calibration |
| **Non-degradation criterion** (what counts as "worse" on one estate) | Not chosen | M1 calibration |
| **Gate floor and budget on three estates** | 0.03 and 20 carry over | M1 calibration |
| **Keep or drop `Middle`** (does it help a model at all?) | Unverified, mirrored from `MainChecklist.md` §D. Agents may use the `Middle` rows in L however they like. Their reports show whether it helped. | Session evidence (M5 onward) |
| **`GCP_PROJECT` and service-account key path** | Placeholders in `.env.example`. Real values go only in the gitignored `.env` and `.secrets/`. | You, before the first session |

---

## Design

### Session lifecycle

Paths in this section and the next are relative to `AgentRSI/`.

```
state/lineage.json + sessions/s(N-1)/HANDOFF.md
        |  build_task.py   (templates + split + prior knowledge)
        v
tasks/sNNN/  --run_session.py-->  stage secrets -> harbor run -p ... -a <agent> -m <model> --base-url ... -e podman
        |                          agent (8 h, public internet, GEE) writes /app/{submission,report,datasets}
        |                          agent sandbox ends -> declared artifacts copied out -> separate verifier grades
        v
jobs/sNNN/  --collect.py-->  secret scan -> sessions/sNNN/ , lineage.json (certify T, promote or not),
        |                    state/events.jsonl
        v   next day: build_task.py reads it all again
scripts/dashboard.py --> dashboard/index.html   (the page you open; regenerated after every collect)
```

- `collect.py` runs on the host and **only copies, scans and parses**. It never imports or executes agent-written code.
- **Handoff numbers come from the verifier, not the agent.** `render_handoff` injects `reward.json` values into the next instruction and flags any figure in the agent's HANDOFF that disagrees.
- The next session learns whether its predecessor's nominee was promoted (through the incumbent). It never sees T numbers.

### Repository layout

```
AgentRSI/
  README.md                  how to run a session, how to customise
  pyproject.toml             uv project, harbor==0.25.0
  .env.example               AGENT, MODEL, BASE_URL, API_FORMAT, API_KEY_NAME, API_KEY, ENV,
                             CONTAINER_HOST, GCP_PROJECT, GCP_SA_KEY_PATH, DAILY_SPEND_CAP_USD
  .secrets/                  GITIGNORED. gcp-sa.json lives here.
  config/
    session.yaml             focus lens, agent, model, budget, cpus/mem, effort, max_budget_usd
    split.yaml               FROZEN: seeds, block size, L/V/T fractions (change = new epoch)
    verifier.yaml            reward weights, gate thresholds, T-attempt budget, bootstrap B, caps
  templates/                 instruction (base + focus lenses), task.toml, Dockerfile.agent / .verifier
  src/agentrsi/              builder library (build_task, build_data, stage_secrets, render_handoff,
                             collect, lineage, events)
  verifier/                  scoring + gate library baked into every tests/ image (unit-tested)
  starter/                   session-0 pipeline (NDVI-LogReg incumbent from the POCs)
  scripts/                   thin CLIs: build_task  run_session  collect  regrade  dashboard  daily
  state/lineage.json         incumbent, scores, epochs, spend ledger, T-attempt counter  (committed)
  state/events.jsonl         append-only event log  (committed)
  sessions/sNNN/             COLLECTED: REPORT.md HANDOFF.md submission/ datasets/ reward.json
                             certification.json meta.json
  dashboard/index.html       GENERATED by scripts/dashboard.py (gitignored). Open it in a browser.
  tasks/  jobs/              GENERATED (gitignored)
```

The builder uses an **allowlist copy**. Nothing reaches `environment/` unless a rule names it, because `data/Labels/*.csv`, the SHPs and the KMLs contain *all* labels.

### Task anatomy

Sketch. Validate with `harbor lint` and `--dry-run` **[?]**.

```toml
schema_version = "1.4"
artifacts = ["/app/submission", "/app/report", "/app/datasets"]

[task]            name = "agentrsi/s003-open"
[agent]           timeout_sec = 28800.0            network_mode = "public"
[environment]     network_mode = "public"          cpus = 6   memory_mb = 6144   storage_mb = 20480
[verifier]        environment_mode = "separate"    timeout_sec = 3600
[verifier.environment]
  cpus = 4   memory_mb = 4096   network_mode = "public"
```

- **Base image:** `python:3.12-slim-bookworm` (needs bash). `tests/Dockerfile` must `COPY test.sh /tests/` (Observations › Harbor on this machine).
- **Agent-visible `environment/`:**
  - `data/`: per estate, the **L** trees with labels (Healthy, Unhealthy or Middle), and the **V and T trees without labels** (coordinates and features, since a deployed model knows where every tree is). V and T Middle trees are in that pool, unmarked.
  - `rasters/`: the ROI-masked ALOS-2 scenes.
  - `prior/`: `docs/POCReport/*`, MainChecklist (settled facts and dead ends), `GEECatalogSurvey.md`, the last `HANDOFF.md`, the incumbent `submission/`, earlier derived datasets (size-capped).
  - Python geo/ML stack (numpy, scipy, pandas, sklearn, xgboost, lightgbm, torch CPU, rasterio, geopandas, `earthengine-api`, `google-cloud-storage`) plus a helper CLI `agentrsi`: `cv` (spatial-block CV on L), `selfcheck` (runs the verifier contract offline on L only), `gcp-init`.
- **Google credentials.** The key is never baked into an image layer.
  - Preferred: `stage_secrets.py` copies `.secrets/gcp-sa.json` to a mode-0600 file under `/mnt/wsl/agentrsi/` (tmpfs the engine can see, wiped on WSL restart, restaged each run). The task mounts it **read-only** at `/run/secrets/gcp-sa.json` and sets `GOOGLE_APPLICATION_CREDENTIALS`. **[T]**
  - Fallback: an agent-phase env var (`--ae`). **[T]**
  - Only the placeholders `GCP_PROJECT` and `GCP_SA_KEY_PATH` live in `.env`. The instruction says where the key is, to initialise `ee` with it, and never to print or copy it.
- **`instruction.md` sections:** Mission | Harness map | Pipeline contract | Deliverables with absolute paths | Freedom (internet, papers, GEE/GCS, pseudo-labelling, data expansion) | Constraints (CPU only, 28800 s, never touch `/tests`, never try to recover sealed labels, never print credentials) | Checkpoint rule | Nominee (name the one pipeline to certify) | Handoff from the previous session (auto-injected). The paper's "do not use online solutions" line is dropped on purpose.
- **Checkpoint rule:** refresh `HANDOFF.md` at least hourly, because a hard timeout can kill the agent mid-write.
- **`HANDOFF.md` schema** (the builder validates it): State of play | What I tried (idea / result / verdict / evidence path) | Dead ends | Open leads, ranked | How to resume | Artifact index. `REPORT.md` is the long report with literature notes.

### Data and splits

One rule for all three estates, frozen in `split.yaml`:

- The **scored set** is Healthy + Unhealthy. The positive class is Unhealthy.
- Whole **spatial blocks** are assigned to **L / V / T at about 40 / 30 / 30 %** (seeded), over all trees including Middle. The draw is repeated until every part clears the simulation's minimums on scored trees: at least 35 positives in V, 35 in T and 50 in L **[S]**.
- **Middle** trees travel with their block. They are visible (labelled in L, unmarked in the V/T pool) and **unscored**: the verifier removes them before every metric, and the counts below leave them out. Keeping them in the V/T pool matters because a deployed model sees the whole field, and neighbour-based features, the one repeatable POC lever, include them.

Expected sizes of the scored trees before block clustering (proportional, so actual draws vary; L also carries about 40% of each estate's Middle trees, roughly 197 for Palong and 151 for Serting):

| Estate | Scored trees | Unhealthy | L (trees / pos) | V (trees / pos) | T (trees / pos) |
|---|---|---|---|---|---|
| AirHitam | 2511 | 160 | ~1004 / ~64 | ~753 / ~48 | ~753 / ~48 |
| Palong | 2171 | 200 | ~868 / ~80 | ~651 / ~60 | ~651 / ~60 |
| Serting | 1667 | 154 | ~667 / ~62 | ~500 / ~46 | ~500 / ~46 |

AirHitam's L of about 1000 matches the finding that NDVI-LogReg saturates by about 1000 labels. Palong and Serting get less than that. The alternative, a fixed L of about 1000, would leave Serting about 31 positives each in V and T, under the 35 minimum.

- **V** is the agent's search signal. Its score is returned to the agent (session-end `reward.json` in v1, in-run queries in v2). It is **expected to become optimistic**.
- **T** is never reported to the agent. It is spent only on certification: one evaluation per session, for the single declared nominee. Each is one **T attempt** in the ledger.
- **Block size is open.** See Observations › Block counts. M1 picks it by calibration, and re-counts blocks on the grid that includes Middle trees (the Observations counts use scored trees only).

**Data expansion.** New covariates ship as `datasets/<name>/features.parquet` keyed by `(estate, tree_id)` for **all** trees, plus `MANIFEST.md` (source, date, resolution, licence) and the extraction script. The agent has no V or T labels, so features at those coordinates cannot leak labels. The script exists for the product, since a new field needs the same features for its own coordinates. The verifier checks id coverage, schema and a size cap.

### Verifier

**Contract.** Per estate the verifier calls `fit_predict(labeled=L, unlabeled=V+T, ctx)`. `labeled` includes the `Middle` rows, and `unlabeled` includes unmarked Middle trees. The verifier drops every Middle tree before scoring. `ctx` holds the estate name, the agent's `datasets/` path, the other estates' L data, a scratch dir and the CPU count (Observations: `cpus` is a quota). The pipeline runs in an **unprivileged subprocess that cannot read the labels** (label files root-owned, mode 0700). A separate trusted process scores afterwards. The verifier also runs the **incumbent** and a **coordinates-only** baseline in the same run, so every result is a **paired delta**. A paired delta of lift equals a paired delta of PR-AUC, because both models are scored on the same trees.

**Packages.** The verifier runs `pip install` of the agent's pinned `requirements.txt` into an isolated venv, as the unprivileged user, before any label is readable by that user. Same base image and Python in both sandboxes. Exact pins and a recorded `pip freeze` keep reruns comparable. Residual risk: pip can run sdist build code, which the unprivileged user and unreadable labels contain.

**Network at grading (fully public).**
- Label protection has no network layer. A pipeline could send labels out only after breaking the isolation above (privilege escalation or a container escape, which matters more on a rootful engine).
- Self-containment is not enforced. A pipeline could download weights or data at grading time and score well while being unreproducible. Mitigations: the instruction requires shipping everything, `selfcheck` runs the contract in the agent sandbox, and `collect.py` greps the pipeline for URLs and network calls and lists them in the session report. This is a smell test, not a guarantee.
- If it ever matters: set `verifier.environment.network_mode` to `allowlist` or `no-network` (one line, needs the nftables sidecar). A nominee about to be promoted could also be re-graded that way with `regrade`.

**Outputs.**
- `reward.json` (V, visible to the agent at session end): composite `score`; per estate `pr_auc_lift`, `pr_auc`, `prevalence`, `recall_at_25`, `enrichment_top10`, `roc_auc`, V CI bounds; paired delta vs the incumbent; delta vs coordinates-only; runtime; `pip freeze`.
- `certification.json` (T, host-only, never injected into any instruction or `tasks/` directory): composite T result with its CI, per-estate T deltas, the non-degradation values, the outcome and the T-attempt count.

**Checks before trusting it.** The oracle (starter) reproduces a known score exactly. `nop` scores neutral on all three estates. Adversarial pipelines (read `/tests`, write the reward, import the labels, run forever, malicious package build code) fail safely. A pipeline that fetches data at grading time is flagged. Gate calibration is re-run on the real verifier, on all three estates, at the chosen block size.

### Certification gate

Promotion is host-side, in `collect.py`. The nominee is promoted only if, on T:
1. the **composite** paired block-bootstrap 5th percentile of dPR-AUC against the incumbent is **> 0**,
2. the composite dPR-AUC is **>= 0.03**, and
3. **no estate is worse than the non-degradation criterion** (set in M1; it can only block adoptions, so it cannot add false adopts).

Thresholds live in `verifier.yaml`. The composite uses the reward's equal weights, so the gate certifies what the agent optimises, and one composite test per nominee keeps one T attempt per session.

**What it cannot do.**
- It reliably certifies gains of roughly **+0.04 PR-AUC or more** (Observations › Simulation). A real +0.012 gain, like NDVI + red-edge (0.133 to 0.145), would almost never pass on its own. The verifier detects big wins, not increments. Small plausible gains accumulate in `REPORT.md` and are **certified in batches** (stack them, nominate the combination).
- A false adopt costs little when it happens. An adopted null idea has a true effect around -0.005 to +0.005. The real cost is a T score that is optimistic for the incumbent and a spent T. Incumbent T scores are never headline numbers, only paired deltas.
- At about 5% false-adopt per attempt, 20 attempts give about a 64% chance of at least one false adopt. When the budget is exhausted, or new labels arrive, the split is refreshed as a new epoch.
- Branching (several agents or models from one parent) speeds up T consumption, because each branch's certification spends an attempt.

### Dashboard

A simple, interactive timeline of RSI progress. It is part of the `AgentRSI/` project like everything else here, not a doc and not a repo-root script.

| | Path |
|---|---|
| Generator | `AgentRSI/scripts/dashboard.py` (thin CLI, logic in `AgentRSI/src/agentrsi/`) |
| Output | `AgentRSI/dashboard/index.html` (gitignored, regenerated) |
| Inputs, read-only | `AgentRSI/sessions/*/reward.json`, `AgentRSI/sessions/*/certification.json`, `AgentRSI/state/events.jsonl`, and `AgentRSI/state/lineage.json` (header only) |
| Context text, read-only | `AgentRSI/sessions/*/REPORT.md` and `HANDOFF.md`, for event context only |

- The output is one self-contained file (no server, no CDN) that opens in a browser with the network off. `collect` and `daily.sh` regenerate it. It is never edited by hand and never written into a `tasks/` directory.
- **Numbers** come from verifier and collector files only (the table above). The agent's report supplies event context only, because handoff numbers must come from the verifier.
- **Series:**
  - Composite V and per-estate V per session, labelled "search signal, optimistic".
  - Certified T deltas per nominee with CI bands and the **+0.03 line**, labelled "certified". V and T are never merged.
  - Epoch shading. Series are not joined across epoch boundaries.
  - A header with the incumbent, T attempts used out of 20 and the current epoch (from `lineage.json`).
- **Events** (`state/events.jsonl`, one JSON object per line, append-only, written once per run so re-collecting never duplicates): `ts`, `session`, `epoch`, `type`, `summary`, `ref`, optional `values`.
  - `type` is one of `session_start`, `session_end`, `nominee_declared`, `certification_pass`, `certification_fail`, `promoted`, `review_flag`, `split_refresh`, `budget_alert`, `spend_cap`, `run_error`.
  - Hover shows `summary`. Click opens a panel with the full record and a link to the `REPORT.md` or `HANDOFF.md` section in `ref`.

### Iteration, epochs and resume

- State is **files, not conversation**. Each session starts fresh with curated context: ledger, HANDOFF, incumbent code, dead-end list. No trajectory replay, because it restores no sandbox files and bloats context **[V]**.
- Dead ends from `MainChecklist.md` section B go into every `prior/`.
- Several agents or models can attempt from the same parent. They are branches in `lineage.json`, and the gate picks.
- A verifier, split or reward-weight change starts a new **epoch**. The incumbent is re-scored with `regrade` or a rerun. Scores are never compared across epochs. Before S1 runs, changes are free.

### Runtime and cost

- **Sizing.** The engine has 15.2 GiB and 12 CPUs, shared with WSL and Windows. A session uses 6 + 4 GiB (agent, verifier), one session at a time. Limits are enforced only on the **rootful** engine.
- **Runner environment.** `run_session.py` sets `PATH` (Linux binaries first), `XDG_RUNTIME_DIR` (a writable dir), `CONTAINER_HOST` and `DOCKER_HOST`, and exports `API_KEY` into the process environment under the name in `API_KEY_NAME`.
- **Gateway.** `--base-url`, `--api-format`, and `--api-key-name` (the host variable that holds the key). Harbor injects the key into the container under the agent's own variable name (Observations › Harbor design).
- **Spend guards.** `max_budget_usd`, `max_turns`, a dedicated spend-capped key, and a ledger with a hard daily cap and kill switch. `DAILY_SPEND_CAP_USD=1` is **not enforced until the ledger exists (M3)**.

---

## Milestones

Each has an exit test.

- **M0 — Harbor + Podman spike** · *nearly closed.* Everything passes except the two agent trials. *Exit:* the full list in Progress.
- **M1 — Data and sealed split** · *next.* `build_data.py`, frozen `split.yaml` with L/V/T for all three estates (proportional, Middle visible but unscored), the allowlist copier. **Calibrate the block size** (try 220 m, 110 m and finer) on all three estates, then calibrate the **composite gate**, the non-degradation criterion, and re-check the 0.03 floor and the 20-attempt budget. *Exit:* canary leak test green for all three estates; stable split hash; every part clears 35 V / 35 T / 50 L positives; block size chosen; composite gate calibrated.
- **M2 — Verifier v1.** `fit_predict` contract; per-estate and composite scoring; T scoring; gate library with unit tests; pip-install at grading; starter pipeline (NDVI-LogReg from `misc/POC/RVI_NDVI/Results/airhitam_ndvi.csv` plus SAR); coordinates-only baseline per estate. *Exit:* oracle reproduces a fixed composite and per-estate score; `nop` neutral on all three; adversarial pipelines fail safely; runtime inside caps.
- **M3 — Builder, runner, collector.** `build_task`, `run_session`, `collect`, `lineage.json`, the T-attempt counter per nominee, the spend ledger, `events.jsonl` and `certification.json`. *Exit:* a 20-minute dry session on a cheap model flows end to end; `harbor lint` and `harbor check` pass; re-collecting the same job does not duplicate events.
- **M4 — Dashboard.** `AgentRSI/scripts/dashboard.py` writes `AgentRSI/dashboard/index.html`, as specified in Design. Built on fixtures and the M3 dry session. *Exit:* page numbers match `reward.json`, `certification.json` and `lineage.json` exactly (scripted diff); every event shows context on hover and on click; epoch boundaries are not joined; CI bands and the +0.03 line are shown; the page works with the network off; no T number appears in any `tasks/` directory (canary).
- **M5 — First real session (S1, 8 h, spend-capped).** *Exit:* you review the report and the dashboard; we tune the templates.
- **M6 — Iteration and schedule.** S2 is generated from S1, the gate works, resume is frictionless. Then Windows Task Scheduler or cron runs `daily.sh` (collect, dashboard, ledger, kill switch).
- **M7 — In-run verifier (v2).** Multi-container sidecar, MCP score tool, a query budget covering all three estates, noise added to returned V scores. The gate is already in place.
- **Later:** more focus lenses, split-refresh epochs, extra estates as folds, remote sandboxes.

---

## Progress

### M0 exit checks (11 Nov 2026)

Tooling lives in `AgentRSI/` (gitignored `.venv/`, `.tools/`). Nothing was installed globally, because the Claude session's sandbox blocked writes to `~/.local/bin`.

| Exit check | Result | Evidence |
|---|---|---|
| Pinned Harbor installs | **Pass.** `harbor==0.25.0` (published 25 Oct 2026, Python 3.12) | `pyproject.toml`, `uv.lock` |
| Linux `podman` and `docker-compose` | **Pass.** podman remote v6.1.4 (matches the engine), docker-compose v5.7.0, both SHA-256 verified | `.tools/bin/` |
| Harbor Podman preflight | **Pass** with the Linux binaries first on `PATH` | `PodmanEnvironment.preflight()` |
| Oracle hello-world on `-e podman` | **Pass.** reward 1.0 | `jobs/m0-oracle3/` |
| Artifacts and verifier after agent timeout | **Pass.** `AgentTimeoutError` recorded, artifact collected, verifier reward 1 | `jobs/m0-timeout/` |
| No egress sidecar for all-`public` | **Pass.** No sidecar text in any job output; source gates it on non-public policies | `docker.py:307` |
| Secret via `--ae` env | **Pass** (marker, not a real key). Reached the container; 0 of 14 job files and the run log contain it | `jobs/m0-secret2/` |
| Secret via read-only `--mounts` | **Pass** (marker). Readable in the container, write fails, marker in 0 job files and the run log | `jobs/m0-mount/` |
| `regrade` after editing `tests/` | **Pass.** Same output, edited verifier, reward 1.0 to 0.5; source job unchanged | `jobs/m0-regrade/` |
| Resource limits bite (rootful) | **Pass.** 256 MB limit killed an 800 MB allocation (exit 137); `--cpus=1` held 4 workers to 0.98 CPU-per-wall (unlimited 3.25) | direct `podman run` |
| Claude Code trial through the gateway | **Blocked.** No gateway key in `.env` | pending |
| Codex trial | **Blocked.** No gateway key in `.env` | pending |

### Next actions

1. **You:** put the gateway key in `AgentRSI/.env` (never in chat), with `BASE_URL` ending in `/v1` for the Codex trial.
2. Run the Codex trial, then the Claude Code trial. Harbor installs both CLIs inside the container (`npm install -g @openai/codex`, and npm or a bootstrap script for Claude Code), so WSL needs neither. **[V]**
3. Start M1.

### Follow-ups

- Delete `AgentRSI/.tools/m0-marker.txt` (a dummy marker) once M0 closes. `AgentRSI/jobs/m0-secret/` was removed on 17 Nov. `jobs/m0-secret2/` stays because the table above cites it.
- The POC reports still describe Palong/Serting labels as derived or estimated. `MainChecklist.md` now tells readers that this is outdated and that the latest `AGENTS.md` wins, so the reports stay as dated records.

---

## Observations

### Data

| Estate | Healthy | Unhealthy | Middle | Prevalence (H+U) | 220 m blocks | 110 m blocks |
|---|---|---|---|---|---|---|
| AirHitam | 2351 | 160 | none | 6.4% | 15 | 42 |
| Palong | 1971 | 200 | 493 | 9.2% | 6 | 24 |
| Serting | 1513 | 154 | 378 | 9.2% | 4 | 15 |

Block counts use the simulation's grid on the scored trees (the AirHitam counts match the simulation's 15 and 42). **[S]**

- **Block counts.** T holds about 30% of the blocks. At 220 m that is 4 to 5 blocks for AirHitam and 1 to 2 for the other two, so a block bootstrap there is not credible. At 110 m it is roughly 13, 7 and 4 to 5, and Serting is still thin. Even AirHitam at 220 m (15 blocks) gave a nominal-5% test (CI > 0 alone) that passed 7 to 16% of null ideas. M1 therefore has to find a block size that still respects spatial correlation but leaves enough blocks in T for all three estates. Going finer than 110 m for Serting is likely.
- **Spatial clustering.** The chance that an Unhealthy tree's ten nearest neighbours are Unhealthy is 0.119 / 0.336 / 0.255 (AirHitam / Palong / Serting), against 0.060 / 0.060 / 0.070 for Healthy trees (`PathForwardTrials.md`). A coordinates-only model is therefore strong, which is why the baseline is reported per estate. Location is a property of the data. It is not a label-quality problem.
- **POC lift** (best SAR + spatial fusion, broad holdout): +0.062 / +0.150 / +0.071 over prevalence; PR-AUC 0.125 / 0.242 / 0.164. Adding SAR to coordinates gave only +0.016 to +0.053 PR-AUC, with several confidence intervals crossing zero.
- **Middle** (`MeanSampling.md` §7b): it is the disease front. Its nearest Unhealthy tree is 14.6 m away, against 26.5 m (Palong) and 20.7 m (Serting) for Healthy trees. Folding Middle into the positive class dilutes the signal, and training with Middle as weak positives lowered ROC in the POCs (Palong HGB 0.606 to 0.562-0.590; Serting 0.500 to 0.462-0.492). Counting Middle as positive would also move prevalence to about 26%. That is why Middle stays out of the positive class and out of the metrics. Showing it to the agent in L is a bet that the label is still useful as context. Nothing here verifies that (Decisions › Open).

### Simulation [S]

Source: `misc/temp/sim_winners_curse.py` and `sim_certification_gate.py` (scratch, gitignored). "Agent ideas" were LogReg on random subsets of the 26 NDVI/spectral/SAR features, split at 220 m blocks into about 40/30/30 L/V/T (about 47 positives in V and in T), compared with the NDVI-LogReg incumbent. This is a rough stand-in for agent behaviour, not a model of it. **It covers AirHitam only. The three-estate composite is not simulated yet.**

- A typical idea is slightly **worse** than the incumbent (mean dPR-AUC -0.005, matching the POCs). The spread between candidates (SD about 0.026) is as large as any real gain.
- **Selecting on V inflates V.** Best-of-k apparent gain on V versus the same candidate on fresh T: k=1 -0.008 / -0.006; k=3 +0.009 / 0.000; k=10 +0.023 / +0.002; k=30 +0.033 / +0.005; k=60 +0.038 / +0.007. 110 m blocks gave the same pattern.
- In v1 the agent touches V once per session, so selection pressure is roughly the number of sessions (best-of-3 optimism about +0.01, best-of-10 about +0.02). In v2, k reaches 30 to 60 inside one session (+0.03 to +0.04). The gate is cheap, so it is in place from day one rather than added with v2.
- The old gate (V CI-separated from zero and T non-degrading) passed a null idea **21 to 25%** of the time at k=10 to 30.

Gate calibration (T-only, one nominee, paired block bootstrap, 220 m blocks, 15 blocks, 120 repetitions):

| Gate | Null, V-winner of k=10 | Null, k=30 | Real +0.02 | Real +0.04 | Real +0.06 |
|---|---|---|---|---|---|
| CI > 0 | 15.8% | 14.2% | 100% | 100% | 99% |
| CI > 0 and d >= 0.02 | 7.5% | 7.5% | 51% | 100% | 99% |
| **CI > 0 and d >= 0.03 (chosen)** | **5.8%** | **4.2%** | **12%** | **77%** | **99%** |
| CI > 0 and d >= 0.04 | 1.7% | 2.5% | 0% | 50% | 93% |

The "real gain" columns are a **ceiling**. The synthetic candidates carry no candidate-specific noise, while real candidates differ from the incumbent with SD about 0.026. Lowering the floor to 0.02 would roughly double false adopts.

### Harbor design **[V]**

- A task is a directory: `instruction.md`, `task.toml`, `environment/`, `tests/`, optional `solution/`.
- **Separate verifier:** grading runs in a fresh sandbox built from `tests/Dockerfile`. Data baked there never reaches the agent. Only declared `artifacts` cross over (sanitized: no setuid, no outward symlinks, no special files). Separate verifiers do not inherit runtime mounts, so the service-account key cannot reach grading.
- **Reward** goes to `/logs/verifier/reward.txt` or `reward.json` (richer, and wins if both exist). terminal-bench-science mandates 0/1. We use graded metrics on purpose.
- **Claude Code:** gateway via `ANTHROPIC_BASE_URL` plus bearer `ANTHROPIC_AUTH_TOKEN`; options `max_turns`, `max_budget_usd`, `reasoning_effort`, `append_system_prompt`, `disable_web_search`; defaults to `--permission-mode bypassPermissions`; Anthropic Messages only.
- **Codex:** Responses API only. `codex.py` injects the key as `OPENAI_API_KEY` and the URL as `OPENAI_BASE_URL`.
- **`--api-key-name`** names the **host** environment variable that holds the key (`agent_auth_resolution.py` reads it, then maps it to the agent's own variable). Any valid variable name works, so `API_KEY_NAME=ANTHROPIC_AUTH_TOKEN` should also work for Codex and is only misleading (read from source, not yet run; the Codex trial confirms it).
- **Network modes** per phase: `public`, `no-network`, `allowlist`. Restricted modes need an nftables egress sidecar. We use `public` everywhere.
- **Flags:** `--base-url`, `--api-key-name`, `--api-format`, `--ae`/`--ak`, `--mounts`, `-e`, `--override-cpus`/`--override-memory-mb`, `--dry-run`, `--print-config`. `harbor lint` is a static check; `harbor check` is an LLM rubric (not yet run). `regrade` re-runs a newer verifier on recorded outputs.
- **`-e podman`** exists upstream (merged Sep 2026), but harbor#398 is open. With `podman-compose` only, Harbor mis-detects the backend. A working setup in that thread used `docker-compose` as the compose provider, which is what we do.

### Harbor and Podman on this machine **[T]** (11 Nov 2026)

- **Engine.** Podman 6.1.4, API 1.44, crun, netavark, overlay, cgroups v2. Rootless socket `podman-user.sock`, rootful socket `/mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock`. 15.2 GiB and 12 CPUs, shared with WSL and Windows (about 11.5 GiB free at test time). The "2 GiB / 6 CPU" in `podman machine list` is nominal.
- **Containers have internet.** An alpine container on the rootful engine saw 12 CPUs and fetched `pypi.org`.
- **Rootless has no cgroup controllers delegated**, so it cannot enforce CPU or memory limits. Rootful has `cpuset, cpu, io, memory, pids`. Mitigations for the weaker containment: no `privileged`, no extra capabilities, no host mounts except one read-only secret, `harbor lint`.
- **The engine runs in a different WSL distro.** Bind-mounting `/home/lee/...` fails, but the shared `/mnt/wsl` tmpfs is visible. Data and code reach the sandbox through the build context. Secrets use `/mnt/wsl/...` or env injection.
- **Kernel.** `CONFIG_NF_TABLES=y`, so a restricted verifier network stays possible later.
- **The Claude session's Bash sandbox** blocks Unix sockets and Windows interop, so engine tests ran outside it. That is a property of that session, not of your setup.

**Findings that change how we build:**
1. **Base image needs bash.** `alpine` fails because Harbor runs `bash -c`. Use `python:3.12-slim-bookworm`.
2. **Separate verifiers need the tests baked in.** If `tests/` has its own `Dockerfile`, Harbor treats that image as owning `/tests` and does not upload `test.sh`. `tests/Dockerfile` must `COPY test.sh /tests/`.
3. **Artifacts must be declared** in `task.toml`, or the verifier cannot see the agent's output.
4. **Regrade needs `-e podman`** and task names that match the source job's trial names.
5. **`cpus` is a CFS quota, not a cpuset.** `os.cpu_count()` still reports 12 inside a `--cpus=1` container, so worker counts must come from `ctx`, and the agent image must be told the real limit.
6. **Harbor uses the Linux `docker-compose` only when `.tools/bin` is first on `PATH`.** Harbor shells out to `podman compose`, which hands off to whichever `docker-compose` it finds.
7. **The WSL VM stalled and restarted** (15:44 to 15:50, 11 Nov). The kernel log shows repeated order-7 page allocation failures from a `vmbus` worker, then memory-pressure messages. You saw disk at 99% and memory very high. The stalled step was the Podman client download. **The cause is not confirmed**, and I only measured the engine and Harbor after the reboot. Keep a memory guard on long downloads.
8. **The runner must set `PATH`, `XDG_RUNTIME_DIR`, `CONTAINER_HOST` and `DOCKER_HOST`.** Without them `podman` fails on `/run/user/1000/libpod` or on the socket.

---

## Risks

| Risk | Mitigation |
|---|---|
| Label leakage into the agent image | Allowlist copy and an automated canary test: no sealed id has a label in the image or build context, for all three estates. Runs in M1 and on every build. |
| Agent code reads sealed labels at grading | Separate sandbox, unprivileged subprocess, root-only label files, scoring in a trusted process. The network is open (decided), so an escaped pipeline could send labels out. Accepted for a throwaway container. |
| Pipeline depends on the internet at grading | Instruction rule, URL/network-call grep in `collect.py`, optional offline re-grade of a nominee. |
| **Holdout overuse and false adoption** | T-only certification of one nominee, effect floor, T-attempt budget, epoch refresh, paired deltas only. |
| Gate cannot see small gains | Stated openly. Batch certification, and the report ledger preserves them. |
| **Block bootstrap unreliable with few blocks** | Block size chosen by calibration on all three estates in M1. |
| **Spatial location mistaken for model skill** | Coordinates-only baseline reported per estate. Location is a real signal in the data. |
| **Dashboard shows optimistic V as certified** | V labelled "search signal"; T shown separately with its CI and the +0.03 line; never merged. |
| Dashboard numbers drift from the verifier | Reads only the verifier and collector files listed in Design › Dashboard. M4 diffs the page against them. |
| **Credential theft:** an agent on the open web holds a service-account key | Least-privilege service account (Object Viewer plus Earth Engine read), read-only mount (not env, not image), instruction forbids printing it, rotate on a schedule and after any anomaly. |
| Key leaks into artifacts, logs or git | Read-only mount outside the build context, a `collect.py` secret scan (`private_key`, `BEGIN PRIVATE KEY`, key id) that **fails collection**, and `jobs/`, `tasks/`, `dashboard/`, `.secrets/`, `.env` gitignored. M0 confirmed Harbor never records the value. |
| GEE quota or cost from an unattended agent | Instruction asks for point sampling at tree coordinates. Check quota and billing, and the ledger logs usage. |
| Package install runs untrusted code in the verifier | Unprivileged user, labels unreadable by it. |
| PyPI drift makes a rerun differ | Exact pins required, `pip freeze` recorded. |
| Rootful engine weakens containment | No privileged mode or extra capabilities, no host mounts but the secret, `harbor lint`. |
| Cost runaway on a third-party provider | `max_budget_usd`, `max_turns`, a spend-capped key, a ledger with a daily cap and kill switch (the cap is unenforced until M3). |
| Provider key exposure | Gitignored `.env`, passed with `--api-key-name` or `--ae`, never in `task.toml` or the task dir. |
| Podman/Harbor instability (harbor#398) | M0 spike, pinned Harbor version, Docker Desktop fallback (only the `-e` flag changes). |
| 8 h/day on a 12-core, 15 GiB WSL box shared with Windows | One session at a time, enforced limits, schedule only after M6, remote sandboxes remain a Harbor option. |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.3 | 17 Nov 2026 | Middle is visible but unscored (was dropped). Dashboard paths pinned inside `AgentRSI/`, inputs listed. `MainChecklist.md` marks old POC label assumptions as outdated. |
| 1.2 | 17 Nov 2026 | Restructured by topic. All three estates trusted equally (equal reward weights, composite T gate). Proportional splits, Middle dropped, `/v1` rule per API format. Dashboard added (M4). Fixed v1.1 errors: POC lifts, Serting split sizes, Palong/Serting block counts, `--api-key-name`. |
| 1.1 | 16 Nov 2026 | Trusted-labels and dashboard findings added as proposals. |
| 1.0 | 11 Nov 2026 | Verifier network fully public. Gate floor 0.03 and T-attempt budget 20 locked. M0 results added. |
| 0.3 | 11 Nov 2026 | AirHitam-dominant weights, T-only gate, rootful engine. |
