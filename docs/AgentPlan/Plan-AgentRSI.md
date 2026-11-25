# AgentRSI Plan

**Version:** 3.1 (25 Nov 2026) | **Phase:** M0–M8 closed (all planned core & MCP milestones live-verified end-to-end) | **Paper:** `docs/Reference/...RSI4EO_2610.18845v2.pdf`

**Goal.** Turn this project into an RSI4EO-style loop. A coding agent builds the oil-palm triage model in a sandbox, a sealed verifier scores it, and each day's session starts from what the last one found. Humans design the harness and verifier. Agents build the models.

**Where we are.** **M0–M8 are closed.** All four focus lenses (`open`, `data-expansion`, `literature`, `ablation`) have been validated end-to-end (including concurrent 3-lens runs with per-task CPU/RAM overrides), archiving three clean covariate stacks (`alos_winstats`, `gee_s2_stress`, `gee_topo`; `15.99 MB` total under `sessions/261124-s006-data-expansion/datasets/`). **M7 (In-run verifier v2)** and **M8 (Host Colab GPU offload MCP broker)** are implemented, unit-tested (**55/55 tests OK**), and live-verified together inside a real Harbor `0.27.0` multi-container Podman session (`claude-haiku-5-5`, `$0.0327`, `0` exceptions, `0` leak findings) hitting the `verifier-sidecar` Streamable HTTP MCP server (`score_v`, noisy V lift, query budget `15 → 14`, `sidecar_queries.jsonl` archived) and a real Google Colab **Tesla T4** GPU via host Unix-socket broker + `agentrsi_mcp_relay.py` (`cuda=True`, `sar-oilpalm` blocked, `colab sessions` verified `0` active VMs after auto-teardown). Default session settings (`config/session.yaml`, `.env.example`) are now **4 h (`14400 s`)**, **`$10` cost/daily spend cap**, and **M7 in-run verifier + M8 Colab (`T4`/`L4` allowed, `A100` restricted) enabled by default**, ready for modeling campaigns over all 3 cached covariate stacks.

Evidence tags: **[V]** read in Harbor docs or source | **[T]** tested on this machine | **[S]** simulation on the real estate tables | **[?]** unverified.

---

## Overview

| RSI4EO concept | Here |
|---|---|
| Harness H (data, libs, budget) | Combined multi-date SAR + S2 NDVI tables and all ROI-masked ALOS-2 rasters for 3 estates, prior POC reports, the incumbent pipeline, GEE read access (no GCS), a Python geo/ML stack. **CPU only** (optional Colab GPU offload via host MCP broker in M8). |
| Verifier V (sealed, scores only) | Sealed held-out blocks for all three estates, each with a validation part (V) and a certification part (T). Returns scores, never labels. |
| Pipeline pi | `submission/`: `fit_predict(labeled, unlabeled, ctx) -> risk scores`, a **training recipe** (the product trains on about 1000 labels in a new field and predicts the rest). |
| Few-label task | Same shape: a labelled slice (L) per estate, the rest (V + T) merged into one unlabeled pool. |
| Recursion inside a task | Agent-driven within the 4 h session. Pseudo-labelling on the unlabeled pool is allowed; nothing is prescribed. |
| **Not in the paper:** recursion across tasks | Our addition. Session N+1 is generated from session N's handoff and the incumbent. |
| 12 h, 4 x GB200, offline | 4 h (`$10` cap), CPU-only sandbox, **public internet**, GEE read access, M7 noisy-V verifier sidecar & M8 Colab GPU offload (`T4`/`L4`, `A100` restricted) enabled by default. |

CPU-only is fine: earlier POCs show nonlinear and deep models lose to linear ones at n=160 positives. Ideas and verification are the bottleneck, not compute.

---

## Decisions

### Made

| Area | Decision | Basis |
|---|---|---|
| **Label trust** | All three estates are treated identically: same split rule, same reward weight, same certification. No provenance field. | Your call (17 Nov): same screening methodology and tools at every site. |
| **Reward** | Mean over the three estates of `lift = pr_auc - prevalence` on V. Equal weights, kept in `verifier.yaml`. The agent sees the per-estate breakdown. | Follows from label trust: no estate is discounted. |
| **Promotion** | One **composite T test** on the same equal weights, plus a per-estate non-degradation check. One nominee per session. Effect floor `d >= 0.03`, T-attempt budget 20. | Confirmed 20 Nov; re-verified over 120 resampled 80 m splits (23 Nov). |
| **Non-degradation** | An estate fails if its T ΔPR-AUC against the incumbent is below `-0.03` (pre-registered 19 Nov, confirmed 20 Nov). Blocks adoption only. Tested at `-0.02`, `-0.03`, `-0.04` on resampled splits: adoption is identical on synthetic arms. | Confirmed at `-0.03`. Open: real-candidate cost. |
| **Dashboard** | Static self-contained HTML inside `AgentRSI/`: `AgentRSI/scripts/dashboard.py` writes `AgentRSI/dashboard/index.html`. Numbers come from verifier and collector files only; agent reports supply event context only. | Your call (17 Nov). Design in Design › Dashboard. |
| **Verifier phases** | v1 grades once at session end. v2 (`inrun_verifier: true` by default in `config/session.yaml`; opt out via `--disable-inrun-verifier`; **M7**, closed 24 Nov **[T]**) adds a multi-container `verifier-sidecar` Streamable HTTP MCP tool (`score_v`, `verifier_budget`, `verifier_health`) with V-only stripped labels, query budget (`15`), and content-hash-cached CSPRNG Gaussian noise (`σ = 0.005`). | |
| **Agents** | Claude Code and Codex through a third-party gateway. Agent, model, base URL and key name are run-time parameters. Current default: Codex, `gpt-6.1-terra`, OpenAI Responses. | |
| **Budget and network** | **4 h (`14400 s`)** default agent timeout and **`$10`** default cost/daily spend cap (`max_budget_usd: 10.0`, `DAILY_SPEND_CAP_USD=10`). Agent has public internet, with one blocked site: this research repo, `https://github.com/Leejames2305/SAR-OilPalm-Detection`. **Verifier network is fully public** (throwaway container). Label protection is the unprivileged subprocess plus root-only label files, not the network. | Your calls (11 Nov, 20 Nov, 25 Nov). Block covers the agent only; verifier stays unblocked. |
| **Packages** | The agent ships a pinned `requirements.txt` and the verifier installs it from PyPI. No allowlist, no approval flow. | Your call (11 Nov). |
| **Runtime** | Rootful Podman from WSL, Docker Desktop as fallback. Harbor pinned at `0.27.0`. | Rootless cannot enforce CPU or memory limits. |
| **Focus** | Open-ended research, explicitly including data expansion. Optional lenses (`literature`, `data-expansion`, `ablation`) bias the instruction. | |
| **Data egress & GEE SA** | ROI-masked ALOS-2, tree coordinates and labels go to the inference provider. A **GEE-only** service-account key (`Keys_SAEarthEngine.json`, Earth Engine viewer only — **no GCS bucket access**) is mounted read-only into the agent sandbox. Verified live (23 Nov): GEE Sentinel-2 queries succeed while GCS `sar-oilpalm` (metadata/list/read) and IAM all return HTTP 403. **[T]** | Your calls (11 Nov, 23 Nov): least-privilege GEE-only SA protects the `sar-oilpalm` bucket holding ground-truth labels. |
| **Colab GPU offload** | **Host-side WSL MCP broker** (`src/colab_broker/`, **M8**, closed 24 Nov **[T]**, **enabled by default** in `config/session.yaml` with `T4`/`L4` allowed and `A100` restricted; opt out via `--disable-colab`) wrapping `google-colab-cli` over a per-session Unix socket + container `stdio` relay (`agentrsi_mcp_relay.py`), never passing Google OAuth/ADC tokens into containers. | `google-colab-cli` requires `https://www.googleapis.com/auth/cloud-platform` scope, which would grant full personal GCP access (including `sar-oilpalm` truth labels) if passed into the container. |
| **Layout & naming** | Everything AgentRSI builds, runs or generates lives under `AgentRSI/`, the dashboard included. Sessions use **`YYMMDD-sXXX-<lens>`** (e.g., `261123-s001-open`, `261123-s002-data-expansion`; verified end-to-end on Harbor `0.27.0` + Podman **[T]**; historical dry run stays `s000-dry`). Generated tasks live in `AgentRSI/tasks/<session_id>/`. Harbor's file is `instruction.md` (singular) **[V]**. | Chronological sorting across long runs; `build_task.py` infers `--focus` from `-<lens>` and avoids doubling it in `task.toml`. |
| **Split sizes** | Proportional: each estate's trees go to L / V / T at about 40 / 30 / 30 % (±10 pt tolerance, minimums 50 L / 35 V / 35 T Unhealthy). Scales to any estate size. | Your call (17 Nov). Accepts L < 1000 for the two smaller estates so Serting clears 35 V / 35 T positives. |
| **Block size** | **80 m** spatial blocks, frozen in `config/split.yaml` (epoch 1, single source of truth read by `common.py`). | 220 m is infeasible for Serting; 165 m fails Serting's ±10 pt tree-share cap (~13 expected redraws). 80 m gives 70 / 47 / 29 blocks (2.0 / 2.0 / 6.2 expected draws). |
| **Middle class** | **Visible but unscored.** Middle trees travel with their block. In L the agent sees them labelled `Middle`; in `unlabelled.csv` (V + T) they sit unmarked, as in a new field. Never in the positive class and never in any metric. Starter pipeline trains without them. | Your call (17 Nov). Whether Middle helps as context stays open (below). |
| **Gateway URL** | Path suffix depends on API format: `BASE_URL` ends in `/v1` for OpenAI Responses (Codex) and has no `/v1` for Anthropic Messages (Claude Code). `run_session.py` picks the URL by agent. | Confirmed on gateway (17–19 Nov). |
| **Sealed labels & agent view** | Master labels live in `AgentRSI/.sealed/` (git-ignored, dir `700`, files `600`). Agent view is **two combined CSVs** with an `estate` column (`L_labelled.csv` and merged `unlabelled.csv` = V + T), no `id`, no `block`, and no V/T split exposure. `(estate, Long, Lat)` identifies every tree uniquely (7,220/7,220). `config/split_assignment.csv` stays host-only. | Your calls (20 Nov, 23 Nov): no pseudonymous keys, merged unlabelled pool, combined tables with `estate` column. |
| **Base table features** | Multi-date ALOS-2 SAR (`Spk`: `HH, HV, VH, VV`; `HAlpha`: `HAlpha_H, HAlpha_A, HAlpha_alpha`) across all complete dates auto-discovered in `data/SAR-scenes/` (currently `260610`, `261014`, date-suffixed) plus Sentinel-2 optical (`B4, B8, NDVI, clear_count`). All raw rasters (including Yamaguchi and AirHitam T3) are also exposed in `environment/rasters/`. | Your call (23 Nov): keep the base table expanding automatically as new dates arrive, and expose all raw rasters to agents. |

### Open

| Item | Current default | Resolved by |
|---|---|---|
| **Non-degradation cost on real candidates** | Value `-0.03` (confirmed 20 Nov). Synthetic arms adopt identically at `-0.02`, `-0.03` and `-0.04` and never fire on synthetic gains. A rough unsimulated estimate (independent per-estate ΔPR-AUC, SD `0.026`, null effect) puts the chance of some estate falling below `-0.02 / -0.03 / -0.04` at ~`0.53 / 0.33 / 0.18`. Real per-estate noise is unmeasured. | Measure per-estate noise on real candidates before M5. |
| **Keep or drop `Middle`** (does it help a model at all?) | Unverified, mirrored from `MainChecklist.md` §D. Agents may use `Middle` rows in L however they like; reports will show whether it helps. | Session evidence (M5 onward) |
| **`GCP_PROJECT` and service-account key path** | Placeholders in `.env.example`. Real values live only in gitignored `.env` and `.secrets/`. | You, before the first real session |

---

## Design

### Session lifecycle

Paths in this section and the next are relative to `AgentRSI/`.

```
state/lineage.json + sessions/<prev_session>/HANDOFF.md
        |  build_task.py   (templates + agent_view + rasters + prior knowledge)
        v
tasks/YYMMDD-sXXX-lens/  --run_session.py-->  stage secrets -> harbor run -p ... -a <agent> -m <model> --base-url ... -e podman
        |                                     agent (4 h, public internet, GEE, M7 sidecar, M8 Colab T4/L4) writes /app/{submission,report,datasets}
        |                                     agent sandbox ends -> declared artifacts copied out -> separate verifier grades
        v
jobs/YYMMDD-sXXX-lens/   --collect.py-->      secret scan -> sessions/YYMMDD-sXXX-lens/ , lineage.json (certify T, promote or not),
        |                                     state/events.jsonl, dashboard/index.html
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
  pyproject.toml             uv project, pinned deps (harbor==0.27.0, pyyaml==6.0.3, ...)
  .env.example               AGENT, MODEL, BASE_URL, API_FORMAT, API_KEY_NAME, API_KEY, ENV,
                             CONTAINER_HOST, GCP_PROJECT, GCP_SA_KEY_PATH, DAILY_SPEND_CAP_USD
  .secrets/                  GITIGNORED. Service-account JSON lives here.
  .sealed/                   GITIGNORED (mode 700, files 600). labels_master.csv, split_report.json,
                             verifier_context/. Never copied into tasks/ or jobs/.
  .work/                     GITIGNORED. features.csv, agent_view/{L_labelled.csv,unlabelled.csv},
                             gate_results.json, gate_run.log.
  config/
    split.yaml               FROZEN single source of truth: epoch, block size, seed, fractions,
                             minimums, counts, sha256 (change = new epoch)
    split_assignment.csv     frozen split with raw ids, block and part, no labels (host-only)
    session.yaml             focus lens, agent, model, budget, cpus/mem, effort, max_budget_usd (M3)
    verifier.yaml            reward weights, gate thresholds, T-attempt budget, bootstrap B, caps (M2)
  templates/                 instruction (base + focus lenses), task.toml, Dockerfile.agent / .verifier
  src/DataPipeline/          M1 data steps: 001-BuildData, 002-CheckLeaks, 003-Calibration, common, features
  src/libbuilder/            builder library (build_task, stage_secrets, render_handoff, collect,
                             lineage, events, dashboard) (M3–M4)
  verifier/                  scoring + gate library baked into every tests/ image (unit-tested) (M2)
  starter/                   session-0 pipeline (NDVI-only balanced LogReg incumbent) (M2)
  scripts/                   thin CLIs: build_task  run_session  collect  regrade  dashboard
  state/lineage.json         incumbent, scores, epochs, spend ledger, T-attempt counter (committed)
  state/events.jsonl         append-only event log (committed)
  sessions/<session_id>/     COLLECTED (YYMMDD-sXXX-lens): REPORT.md HANDOFF.md submission/ datasets/ reward.json
                             certification.json meta.json
  dashboard/index.html       GENERATED by scripts/dashboard.py (gitignored). Open it in a browser.
  tasks/  jobs/              GENERATED (gitignored)
```

The builder uses an **allowlist copy**. Nothing reaches `environment/` unless a rule names it, because `data/Labels/*.csv`, `data/SHPs/*`, `config/split_assignment.csv`, and `.sealed/*` contain raw labels or the V/T partition.

### Task anatomy

Sketch. Validate with `harbor lint` and `--dry-run` **[?]**.

```toml
schema_version = "1.4"
artifacts = ["/app/submission", "/app/report", "/app/datasets"]

[task]            name = "agentrsi/261123-s001-open"
[agent]           timeout_sec = 14400.0            network_mode = "public"
[environment]     network_mode = "public"          cpus = 6   memory_mb = 6144   storage_mb = 20480
[verifier]        environment_mode = "separate"    timeout_sec = 3600
[verifier.environment]
  cpus = 4   memory_mb = 4096   network_mode = "public"
```

- **Base image:** `python:3.12-slim-bookworm` (needs bash). `tests/Dockerfile` must `COPY test.sh /tests/` (Observations › Harbor on this machine).
- **Agent-visible `environment/`:**
  - `data/`: two combined CSVs covering all three estates (with an `estate` column, no `id`, no V/T partition):
    - `L_labelled.csv`: `estate, Long, Lat`, multi-date SAR (`HH_<date>`, `HV_<date>`, `VH_<date>`, `VV_<date>`, `HAlpha_H_<date>`, `HAlpha_A_<date>`, `HAlpha_alpha_<date>`), S2 (`B4, B8, NDVI, clear_count`), and `label` (`Healthy`, `Unhealthy`, `Middle`).
    - `unlabelled.csv`: merged V + T trees with the same feature columns and no `label`. V and T `Middle` trees sit unmarked in this pool, as they would in a new field.
  - `rasters/`: all ROI-masked ALOS-2 scenes from `data/SAR-scenes/` (both `260610` and `261014`, including `Spk`, `HAlpha`, `Yamaguchi`, and AirHitam `Cal_mat_Spk_TC`).
  - `prior/`: `docs/POCReport/*`, `MainChecklist.md` (settled facts and dead ends), `Plan-GEECatalogSurvey.md`, the last `HANDOFF.md`, the incumbent `submission/`, earlier derived datasets (size-capped).
  - Python geo/ML stack (`numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `torch` CPU, `rasterio`, `geopandas`, `earthengine-api`, `google-cloud-storage`) plus a helper CLI `agentrsi`: `cv` (spatial-block CV on L), `selfcheck` (runs the verifier contract offline on L only), `gcp-init`.
- **Google credentials.** The key is never baked into an image layer.
  - Preferred: `stage_secrets.py` copies the service-account JSON to a mode-0600 file under `/mnt/wsl/agentrsi/` (tmpfs visible to the engine, wiped on WSL restart, restaged each run). The task mounts it **read-only** at `/run/secrets/gcp-sa.json` and sets `GOOGLE_APPLICATION_CREDENTIALS`. **[T]**
  - Fallback: an agent-phase env var (`--ae`). **[T]**
  - Only the placeholders `GCP_PROJECT` and `GCP_SA_KEY_PATH` live in `.env.example`. The instruction says where the key is, to initialise `ee` with it, and never to print or copy it.
- **`instruction.md` sections (overhauled 24 Nov to Terminal-Bench-Science rubric + `AGENTS.md` standards):** Objective (downstream polarized-imagery triage context + estate label parity warning against older POC reports) | Data Specification (exact tabular provenance/units + authoritative 19-GeoTIFF band order, CRS, linear vs dB Yamaguchi scale, T3 coherency, and masking rules) | Evaluation Metrics & Validation Discipline (exact tie-aware AP lift, secondary triage diagnostics, coords-only 15-NN comparison, per-estate non-degradation, 80 m block spatial-leakage warning) | Submission & Execution Contract (explicit agent-vs-verifier package divergence, read-only/offline/no-symlink rules, dynamic resource/timeout placeholders from `session.yaml` + `verifier.yaml`, dataset schema & forbidden columns, `REPORT.md`/`HANDOFF.md`) | Research Latitude & Operational Discipline | Constraints | Focus Lens | Handoff (auto-injected) | Canonical wall-clock timeout closing line (`You have <N> seconds...`, keeping literature/PyPI/GEE open while forbidding repo/sealed-label/credential access). `build_task.py` also enforces an unreplaced-`{{PLACEHOLDER}}` guard alongside the T-leak canary.
- **Checkpoint rule:** refresh `HANDOFF.md` at least hourly, because a hard timeout can kill the agent mid-write.
- **`HANDOFF.md` schema** (validated by the builder): State of play | What I tried (idea / result / verdict / evidence path) | Dead ends | Open leads, ranked | How to resume | Artifact index. `REPORT.md` is the long report with literature notes.

### Data and splits

One rule for all three estates, frozen in `config/split.yaml` (epoch 1):

- The **scored set** is `Healthy` + `Unhealthy`. The positive class is `Unhealthy`.
- Whole **80 m spatial blocks** are assigned to **L / V / T at 40 / 30 / 30 %** (seed `20261119`), over all trees including `Middle`. The draw is repeated until every part clears the minimums on scored trees (`>= 50` Unhealthy in L, `>= 35` in V, `>= 35` in T) and stays within ±10 percentage points of target tree share (all trees).
- **`Middle` trees** travel with their block. They are visible (`label=Middle` in `L_labelled.csv`, unmarked in `unlabelled.csv`) and **unscored**: the verifier removes them before every metric. Keeping them in the unlabeled pool matters because a deployed model sees the whole field, and neighbour-based features include them.
- **V** is the agent's search signal (`reward.json` at session end in v1, in-run queries in v2). It is **expected to become optimistic**.
- **T** is never reported to the agent and never exposed separately in `environment/data/`. It is spent only on certification: one evaluation per session for the declared nominee (one **T attempt** in the ledger).

Frozen Epoch-1 split counts (`config/split.yaml`, sha256 `26f34d4d...`):

| Estate | L: trees · U (M) | V: trees · U (M) | T: trees · U (M) | Blocks | Tree share L / V / T (%) |
|---|---|---|---|---|---|
| AirHitam | 964 · 56 (0) | 730 · 38 (0) | 817 · 66 (0) | 70 | 38.4 / 29.1 / 32.5 |
| Palong | 968 · 66 (198) | 841 · 54 (130) | 855 · 80 (165) | 47 | 36.3 / 31.6 / 32.1 |
| Serting | 815 · 67 (151) | 552 · 40 (103) | 678 · 47 (124) | 29 | 39.9 / 27.0 / 33.2 |

U = Unhealthy, M = Middle (unscored, unmarked in `unlabelled.csv`). Total: 7,220 trees (6,349 scored, 871 Middle); agent view: `L_labelled.csv` 2,747 rows, `unlabelled.csv` 4,473 rows.

**Data expansion.** New covariates ship as `datasets/<name>/features.parquet` keyed by **`(estate, Long, Lat)`** for **all** trees, plus `MANIFEST.md` (source, date, resolution, licence) and the extraction script. The agent has no V or T labels, so features at those coordinates cannot leak labels. The script exists for the product, since a new field needs the same features for its own coordinates. The verifier checks coordinate-key coverage, schema and a size cap.

### Verifier

**Contract.** Per estate the verifier calls `fit_predict(labeled=L, unlabeled=V+T, ctx)` returning risk scores aligned to `unlabeled` (verified by row order / `(estate, Long, Lat)`). `labeled` includes the `Middle` rows, and `unlabeled` includes unmarked `Middle` trees; the verifier drops every `Middle` tree before scoring. `ctx` holds the estate name, the agent's `datasets/` path, the other estates' L data, a scratch dir and the CPU count (Observations: `cpus` is a quota). The pipeline runs in an **unprivileged subprocess that cannot read the labels** (label files root-owned, mode `0700`). A separate trusted process scores afterwards. The verifier also runs the **incumbent** and a **coordinates-only** baseline in the same run, so every result is a **paired delta** (a paired delta of lift equals a paired delta of PR-AUC because both models are scored on the same trees).

**Packages.** The verifier runs `pip install` of the agent's pinned `requirements.txt` into an isolated venv, as the unprivileged user, before any label is readable by that user. Same base image and Python in both sandboxes. Exact pins and a recorded `pip freeze` keep reruns comparable.

**Network at grading (fully public).**
- Label protection relies on the unprivileged subprocess + root-only label files, not a network wall.
- Self-containment is encouraged by the instruction, checked locally by `selfcheck`, and audited by a URL/network-call grep in `collect.py`. If ever needed, `verifier.environment.network_mode` can be switched to `no-network` (or used selectively via `regrade` on a nominee about to be promoted).

**Outputs** (downloaded from `/logs/verifier/` by Harbor **[V]/[T]**):
- `reward.json` (flat dict of finite `float | int` ONLY — enforced by Harbor `_parse_reward_json()` **[V]**): `score` (V composite lift), `v_lift_AirHitam`, `v_lift_Palong`, `v_lift_Serting`, `v_delta_vs_incumbent`, `v_delta_vs_coords`, `submission_valid`, `runtime_sec`.
- `v_report.json` (V-only rich report, zero T numbers): composite & per-estate V metrics (`pr_auc_lift`, `pr_auc`, `prevalence`, `recall_at_25`, `enrichment_top10`, `roc_auc`), paired block-bootstrap V CIs vs incumbent and coordinates-only, `datasets_audit`, `pip_freeze`, `status`/`error`.
- `certification.json` (T, host-only, never injected into any instruction or `tasks/` directory, no tree IDs or coordinates): composite T result with its paired block-bootstrap CI, per-estate T metrics and deltas, `gate` verdict (`passed`, `worst_estate`, `failed_reasons`).

**Verified trust checks (M2 exit, 23 Nov).** Starter incumbent reproduces fixed V and T scores (`v_delta_vs_incumbent = 0.0`, `t_comp_delta = 0.0`, `gate_passed = False`); `nop` scores `0.0` lift on all three estates; adversarial pipelines (read `/sealed`, overwrite `/logs/verifier/reward.json`, infinite timeout loop, symlink into `/sealed`, forbidden `requirements.txt` paths/options) all fail safely (`submission_valid = 0`, `score = 0.0`); pinned PyPI install (`tabulate==0.9.0`) succeeds as `runner` and records `pip_freeze`. Note: runtime URL/network-call grep in `collect.py` remains an M3 deliverable.

### Certification gate

Promotion is host-side, in `collect.py`. The nominee is promoted only if, on T:
1. the **composite** paired block-bootstrap 5th percentile of ΔPR-AUC against the incumbent is **> 0**,
2. the composite ΔPR-AUC is **>= 0.03**, and
3. **no estate's T ΔPR-AUC is below `-0.03`** (non-degradation; blocks adoption only, so it cannot add false adopts).

Thresholds live in `config/verifier.yaml`. Equal estate weights match the reward, and one composite test per nominee keeps one T attempt per session.

**What it cannot do.**
- It reliably certifies gains of roughly **+0.04 PR-AUC or more** (`96.7%` synthetic power at `+0.04`, `0%` at `+0.02`; Observations › Gate calibration). Small plausible gains (e.g., `+0.012`) accumulate in `REPORT.md` and are **certified in batches** (stack them, nominate the combination).
- A false adopt costs little in immediate quality (null ideas cluster near 0), but makes the incumbent's T score optimistic and spends a T attempt. Incumbent T scores are never headline numbers—only paired deltas.
- At a 20-attempt budget, when exhausted (or when new labels arrive), the split is refreshed as a new epoch. Branching spends one T attempt per certified branch.

### Dashboard

A simple, interactive timeline of RSI progress inside `AgentRSI/`.

| | Path |
|---|---|
| Generator | `AgentRSI/scripts/dashboard.py` (thin CLI, logic in `AgentRSI/src/libbuilder/`) |
| Output | `AgentRSI/dashboard/index.html` (gitignored, regenerated after every `collect`) |
| Numeric inputs (read-only) | `sessions/*/reward.json`, `sessions/*/certification.json`, `state/events.jsonl`, `state/lineage.json` (header only) |
| Context text (read-only) | `sessions/*/REPORT.md` and `HANDOFF.md` (event context only, never numbers) |

- One self-contained HTML file (no server, no CDN, works offline), never edited by hand and never written into `tasks/`.
- **Series:** composite V and per-estate V per session (labelled "search signal, optimistic"); certified T deltas per nominee with CI bands and the **+0.03 line** (labelled "certified", never merged with V); epoch shading with series broken across epoch boundaries; header with incumbent, T attempts used out of 20, and current epoch.
- **Events** (`state/events.jsonl`, append-only, idempotent per run): `ts`, `session`, `epoch`, `type`, `summary`, `ref`, optional `values`. Types: `session_start`, `session_end`, `nominee_declared`, `certification_pass`, `certification_fail`, `promoted`, `review_flag`, `split_refresh`, `budget_alert`, `spend_cap`, `run_error`. Hover shows `summary`; click opens details + link to `REPORT.md`/`HANDOFF.md`.

### Iteration, epochs and resume

- State is **files, not conversation**. Each session starts fresh with curated context: ledger, HANDOFF, incumbent code, dead-end list (`MainChecklist.md` §B). No trajectory replay **[V]**.
- Several agents or models can attempt from the same parent as branches in `lineage.json`; the gate picks.
- Any verifier, split, or reward-weight change starts a new **epoch** (incumbent re-scored with `regrade` or rerun; scores are never compared across epochs). Before S1 runs, changes are free.

### Optional Colab GPU offload (M8)

On-demand GPU VMs via `google-colab-cli` can accelerate offline feature/embedding extraction (e.g., geospatial foundation models over ROI rasters) and heavy exploration, while keeping the agent container and verifier CPU-only.

- **Why tokens never enter the container:** `google-colab-cli` requires `openid`, `userinfo.email`, `colaboratory`, **and `https://www.googleapis.com/auth/cloud-platform`** (both `--auth=oauth2` and `--auth=adc`). Even though it does not expose Gmail or full Drive, `cloud-platform` grants full personal GCP access — including the `sar-oilpalm` bucket holding sealed ground-truth labels and unrestricted GCP billing. Passing user OAuth/ADC credentials into the agent sandbox would defeat the least-privilege GEE-only SA.
- **Architecture (WSL host MCP broker, verified live 24 Nov **[T]**):**
  1. **Host-only auth:** authenticate `google-colab-cli` once on the WSL host (`colab auth`). No user Google token ever touches `/mnt/wsl/` or any container.
  2. **Hardened MCP broker (`AgentRSI/src/colab_broker/`):** runs on the WSL host during GPU-enabled sessions (`run_session.py`) and listens on a per-session Unix domain socket `/mnt/wsl/agentrsi/colab-<session_id>.sock` (mounted at `/run/agentrsi/colab-mcp.sock`) alongside a jailed exchange dir `/mnt/wsl/agentrsi/colab-io-<session_id>` (`/run/agentrsi/colab-io/{in,out}`), reusing the `/mnt/wsl/` tmpfs bridge proven in M0 **[T]**.
  3. **Harbor native MCP wiring **[V]/[T]**:** registered in `task.toml` via `[[environment.mcp_servers]]` (`transport = "stdio"`, `command = "python3"`, `args = ["/usr/local/bin/agentrsi_mcp_relay.py", "--socket", "/run/agentrsi/colab-mcp.sock"]`), natively supported by both Claude Code and Codex in Harbor `0.27.0`.
- **Mandatory broker guardrails:**
  - **Narrow tool surface only:** `colab_run(script, gpu, timeout_min, staged_inputs, output_paths, pip_packages)`, `colab_status(job_id)`, `colab_fetch(job_id)`, `colab_stop(job_id)`. **Never** expose raw `colab` CLI, `colab auth`, `google.colab.auth`/`authenticate_user`, `drive.mount`, `gcloud`/`gsutil`, `sar-oilpalm`, repo identifiers, metadata IPs, or unsafe `pip` flags.
  - **Credit & lifecycle guards:** GPU tier allowlist (`T4`/`L4` default, `A100` opt-in), max 1 active VM per session (`VM_BUSY`), hard per-job timeout (`colab exec --timeout`) + per-session GPU wall-clock cap, automatic `colab stop` on idle or session exit (`finally:` in `run_session.py`), stripped host secrets in `sanitized_colab_env()`, and audit entries (`colab_vm_start`, `colab_job_run`, `colab_vm_stop`, `colab_blocked`) in `state/events.jsonl`.
- **Verifier contract boundary:** the verifier stays **CPU-only** (`4 CPU / 4 GiB / 1 h`) with no Colab access. Colab GPU outputs must land as reproducible `datasets/<name>/features.parquet` (plus extraction script) or distilled artifacts that a fast CPU `fit_predict()` consumes at grading time.

### Runtime and cost

- **Sizing.** The engine has 15.2 GiB and 12 CPUs, shared with WSL and Windows. One default session at a time (`6 + 2 GiB` / `6 + 2 CPUs` for agent + M7 `verifier-sidecar` during the agent phase, then `4 GiB` / `4 CPUs` for the separate verifier). Limits are enforced only on the **rootful** engine.
- **Runner environment.** `run_session.py` sets `PATH` (Linux binaries first), `XDG_RUNTIME_DIR`, `CONTAINER_HOST`, `DOCKER_HOST`, `PODMAN_COMPOSE_WARNING_LOGS=false`, picks `BASE_URL` by agent, strips inline `# ...` comments from `.env`, and exports `API_KEY` under `API_KEY_NAME`.
- **Spend guards.** Default `$10` cost cap: `max_budget_usd: 10.0` / `max_turns` (for `claude-code`; Codex rejects those kwargs **[V]/[T]**), a dedicated spend-capped key, and `state/lineage.json` enforcing `DAILY_SPEND_CAP_USD=10`, `t_attempt_budget`, and `kill_switch` inside `run_session.py`.

---

## Milestones

Each has an exit test.

- **M0 — Harbor + Podman spike** · *closed 19 Nov 2026.* *Exit:* full list in Progress.
- **M1 — Data and sealed split** · *closed 20 Nov 2026, hardened 23 Nov 2026.* `src/DataPipeline/` reading `config/split.yaml` (single source of truth), frozen 80 m split with proportional L/V/T (Middle visible but unscored), multi-date SAR (`260610`, `261014`, auto-discovered, `HAlpha_H/A/alpha`) + S2 NDVI, combined `L_labelled.csv` + merged `unlabelled.csv`, hardened leak checker (verified clean on `prior/` docs and GeoTIFF rasters), and resampled-split composite gate calibration. *Exit:* all checks green in Progress.
- **M2 — Verifier v1** · *closed 23 Nov 2026.* `verifier/` scoring, baselines (`incumbent_fit_predict`, `coords_only_fit_predict`), isolated `runner` subprocess + pinned PyPI `pip install`, `config/verifier.yaml`, `starter/`, local sealed base image (`localhost/agentrsi-sealed:epoch1` via `scripts/build_verifier_base.py`), 11 unit tests, Harbor Oracle/Nop integration trials (`spike/m2-verifier`), and 5 container adversarial/pip checks. *Exit:* all checks green in Progress.
- **M3 — Builder, runner, collector** · *closed 23 Nov 2026.* Prebuilt `localhost/agentrsi-agent-base:latest` (`scripts/build_agent_base.py`) with in-container `agentrsi` CLI (`selfcheck`, `cv`, `gcp-init`), `config/session.yaml`, `templates/`, `src/libbuilder/` (`build_task`, `run_session`, `collect`, `render_handoff`, `stage_secrets`, `lineage`, `events`, `env_utils`), `scripts/{build_task,run_session,collect,regrade}.py`, 5 builder/collector unit tests, and end-to-end `s000-dry` session (`claude-code` / `claude-haiku-5-5`). *Exit:* all checks green in Progress (`harbor check` waived on `-e podman`: empirically confirmed still broken in Harbor `0.27.0` because `_wait_until_compose_exec_ready()` runs before `_upload_environment_dir_after_start()` on `crun`).
- **M4 — Dashboard** · *closed 23 Nov 2026.* `src/libbuilder/dashboard.py` + thin CLI `scripts/dashboard.py` (`--verify`) + auto-refresh in `collect.py`, writing self-contained offline `dashboard/index.html` with separate V (search signal) and T (certified + CI + `+0.03` floor) SVG charts broken at epoch boundaries, sessions table, events ledger, and inline `HANDOFF.md`/`REPORT.md` context inspector. *Exit:* all checks green in Progress (20/20 unit tests, `0` mismatches on scripted diff against `s000-dry` and multi-epoch fixtures).
- **M5 — First real session (S1, 1 h pilot, spend-capped) · *closed 24 Nov 2026.*** S1 (`261124-s001-open`) completed, collected, and promoted; all four audit gaps fixed and verified (25/25 unit tests + live S2-from-S1 `oracle` confirming dynamic incumbent & Option B breadth gate).
- **M6 — Fast startup & S2 iteration (manual kickoff, no scheduler) · *closed 24 Nov 2026.*** Prebaked `node v22.23.5`, `npm 10.9.9`, `ripgrep 13.0.0`, `codex-cli 0.172.0`, and `claude 2.1.331` into `localhost/agentrsi-agent-base:latest` (`agent_setup` `203.4s → 0.8s`, 254× faster). Verified S2-from-S1 end-to-end (`261124-s002-open`: handoff injection, `/app/datasets/alos_winstats/` extraction & audit, dynamic S1 incumbent grading, Option B breadth-gate rejection, `0` false-positive smells, `--dry-session` collection preserving `1 / 20` T attempts and archiving `alos_winstats`, `dashboard.py --verify` `0 mismatches`, leak checks `PASS`). *(Cron / Windows Task Scheduler / `daily.sh` dropped by user decision — sessions are launched manually.)*
- **M7 — In-run verifier (v2) · *closed 24 Nov 2026.*** Multi-container `verifier-sidecar` (`verifier/inrun_server.py`, isolated `/opt/agentrsi/sidecar-venv` with `mcp==2.3.0` + `uvicorn==0.54.0` keeping `/usr/local/bin/python` clean for Epoch-1 v1 grading), Streamable HTTP MCP endpoint (`http://verifier-sidecar:8765/mcp`) guarded by ephemeral `AGENTRSI_VERIFIER_MCP_TOKEN` (`hmac.compare_digest`), V-only stripped bundle (zero T labels loaded), `threading.Lock` query budget (`15`), CSPRNG Gaussian noise (`σ = 0.005`) cached by `content_sha256`, jailed `/shared` staging, and `sidecar_queries.jsonl` collection. *Exit:* 10/10 M7 unit tests + live Podman sidecar test + live Harbor multi-container session (`261124-m78-live-open`).
- **M8 — Optional Colab GPU offload (MCP broker) · *closed 24 Nov 2026.*** Host-side WSL MCP broker (`src/colab_broker/`, `scripts/colab_broker.py`) wrapping `google-colab-cli` over per-session `/mnt/wsl/agentrsi/colab-<session_id>.sock` + jailed `/run/agentrsi/colab-io` + container `stdio` relay (`/usr/local/bin/agentrsi_mcp_relay.py`) wired via Harbor `[[environment.mcp_servers]]`, with GPU allowlist (`T4`/`L4`, `A100` opt-in), max-1-VM cap, wall-clock cap, idle/exit auto-teardown, `sanitized_colab_env()`, and script/path guardrails blocking `colab auth`, `drive.mount`, `gcloud`/`gsutil`, `sar-oilpalm`, and metadata IPs. *Exit:* 14/14 M8 unit tests + live Podman + real Colab **Tesla T4** extraction both standalone and inside Harbor (`261124-m78-live-open`), `sar-oilpalm` blocked (`RefreshError`) with zero leaked env/Drive/ADC tokens inside the VM, and `colab sessions` verified `0` active VMs after auto-teardown.
- **Later:** more focus lenses, split-refresh epochs, extra estates as folds, remote sandboxes.

---

## Progress

### M0 exit checks (11–19 Nov 2026)

Tooling lives in `AgentRSI/` (gitignored `.venv/`, `.tools/`).

| Exit check | Result | Evidence |
|---|---|---|
| Pinned Harbor installs | **Pass.** `harbor==0.27.0` (upgraded from `0.25.0`, Python 3.12) | `pyproject.toml`, `uv.lock` |
| Linux `podman` and `docker-compose` | **Pass.** podman remote v6.1.4, docker-compose v5.7.0, SHA-256 verified | `.tools/bin/` |
| Harbor Podman preflight | **Pass** with `.tools/bin` first on `PATH` | `PodmanEnvironment.preflight()` |
| Oracle hello-world on `-e podman` | **Pass.** reward `1.0` | `jobs/m0-oracle3/` |
| Artifacts and verifier after agent timeout | **Pass.** `AgentTimeoutError` recorded, artifact collected, reward `1.0` | `jobs/m0-timeout/` |
| No egress sidecar for all-`public` | **Pass.** source gates sidecar on non-public policies | `docker.py:307` |
| Secret via `--ae` env | **Pass** (marker). Reached container; 0 of 14 job files and run log contain it | `jobs/m0-secret2/` |
| Secret via read-only `--mounts` | **Pass** (marker). Readable, write fails, 0 job files/logs contain it | `jobs/m0-mount/` |
| `regrade` after editing `tests/` | **Pass.** reward `1.0` to `0.5`; source job unchanged | `jobs/m0-regrade/` |
| Resource limits bite (rootful) | **Pass.** 256 MB killed 800 MB alloc (exit 137); `--cpus=1` held 4 workers to 0.98 CPU/wall | direct `podman run` |
| Claude Code trial through gateway | **Pass.** reward `1.0`, `claude-haiku-5-5`, `anthropic_messages` (no `/v1`), 3 steps, ~\$0.0025 | `jobs/m0-claude-ok/` |
| Codex trial through gateway | **Pass.** reward `1.0`, `gpt-6.1-terra`, `openai_responses` (`…/v1`), 8 steps, ~\$0.015 | `jobs/m0-codex-ok/` |

### M1 exit checks (closed 20 Nov 2026, hardened 23 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Canary leak test green (view, jobs, prior docs, rasters) | **Pass.** `002-CheckLeaks.py` passes on `.work/agent_view/` (`L_labelled.csv` 2,747 rows, merged `unlabelled.csv` 4,473 rows), `jobs/` (143 files), `docs/POCReport` + `docs/AgentPlan` (12 files, 0 false positives), and `data/SAR-scenes` (19 GeoTIFFs). `--control` catches all 4 seeded leaks (view CSV, task CSV, lowercase-class JSON, split-assignment line). | `.work/agent_view/`, `src/DataPipeline/002-CheckLeaks.py` |
| Stable split hash & single source of truth | **Pass.** `common.py` loads `config/split.yaml`; `001-BuildData.py` reproduces `26f34d4d8e0159e53985fce735adf410e0cae392d7f1cb8aded8a7490cd0e2ed` byte-identically and verifies counts against `split.yaml`. | `config/split.yaml`, `config/split_assignment.csv` |
| Every part clears 50 L / 35 V / 35 T positives & ±10 pt share | **Pass.** Lowest Unhealthy: L 56 and V 38 (AirHitam), T 47 (Serting). Max share deviation: 3.7 pts (Palong L). | Split table in Design › Data and splits |
| Block size chosen | **Pass.** 80 m (70 / 47 / 29 blocks). | `config/split.yaml` |
| Sealed master and verifier staging | **Pass (host side).** `.sealed/` mode `700`, files `600`, git-ignored. `labels_master.csv` (`ea434957…`, 7,220 rows) staged in `.sealed/verifier_context/`. Root-owned image copy is M2 (`[?]`: likely prebuilt local image + `FROM` since Harbor `tests/` context cannot reach `.sealed/`). | `.sealed/`, `.sealed/verifier_context/manifest.json` |
| Multi-date features from raw sources | **Pass.** 7,220 trees × 18 features, **0 NaNs**: auto-discovered ALOS-2 `260610` + `261014` (`HH, HV, VH, VV, HAlpha_H, HAlpha_A, HAlpha_alpha` per date) + S2 `B4, B8, NDVI, clear_count`. AirHitam NDVI matches POC (`0.0` diff); local SAR re-extraction matches (`0.0` diff). | `.work/features.csv`, `src/DataPipeline/features.py` |
| Least-privilege GEE SA verified | **Pass.** `Keys_SAEarthEngine.json` (`gee-resources@…`) succeeds on live S2 Earth Engine queries (`60` scenes sampled) and returns HTTP `403` on all `sar-oilpalm` GCS probes (bucket metadata, list objects, list project buckets, object get) and IAM APIs. **[T]** | `AgentRSI/.secrets/Keys_SAEarthEngine.json` |
| Composite gate calibrated (resampled splits) | **Pass on synthetic arms.** `003-Calibration.py gate` over 120 **resampled** 80 m splits (never touching frozen T, 17 candidate features): 0.000 false adopts, 0.967 power at +0.04, 1.000 at +0.06; `comp_only` recorded. Non-degradation confirmed at `-0.03`. Real candidates stay open. | `.work/gate_results.json`, Observations › Gate calibration |

### M2 exit checks (closed 23 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Sealed verifier base image + `FROM` route | **Pass.** `scripts/build_verifier_base.py` builds `localhost/agentrsi-sealed:epoch1` (`python:3.12-slim-bookworm` + pinned `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`, `pyarrow==26.0.1`), root-owned `/sealed` (`0700`/`0600`), unprivileged `runner` (`uid 10001`). `runner` `ls /sealed` denied; Harbor `tests/Dockerfile` `FROM localhost/agentrsi-sealed:epoch1` works cleanly on `-e podman`. **[T]** | `scripts/build_verifier_base.py`, `spike/m2-verifier/tests/Dockerfile` |
| Host unit tests (metrics, gate, runner) | **Pass.** 11/11 tests pass (`exact_ap` vs sklearn, `nop` metrics, bootstrap identity/gain, gate boundaries, `requirements.txt` sanitizer, dataset audit, symlink rejection, host starter/`nop`/failure grading). | `tests/test_verifier.py`, `verifier/` |
| Starter oracle reproduces fixed V & T scores | **Pass.** Harbor `-a oracle` matches host run exactly: composite V score `0.054469` (`AirHitam +0.046745`, `Palong +0.042640`, `Serting +0.074023`), `v_delta_vs_incumbent = 0.0`, `v_delta_vs_coords = -0.011992`, T `comp_delta = 0.0`, `gate_passed = False`, `runtime_sec = 4.209`. **[T]** | `jobs/m2-oracle/`, `starter/pipeline.py` |
| `nop` neutral on all three estates | **Pass.** Harbor `-a nop` returns `score = 0.0`, per-estate `v_lift_* = 0.0`, `recall_at_25 = 0.25`, `enrichment_top10 = 1.0`, `submission_valid = 0`, `runtime_sec = 0.994`. **[T]** | `jobs/m2-nop/` |
| Pinned PyPI install & adversarial isolation | **Pass (5/5 container tests).** In `localhost/agentrsi-sealed:epoch1`: `read_sealed` (`PermissionError`), `write_reward_json` (`PermissionError`, `/logs/verifier` locked `0700`), `timeout_loop` (killed in `3.24s`), `pip_tabulate` (`tabulate==0.9.0` installed as `runner`, `score = 0.054469`, recorded in `pip_freeze`), `malicious_reqs` (blocked `-r /sealed/...` before pip). **[T]** | `verifier/runner.py` |
| `harbor lint` & leak check on M2 task/jobs | **Pass.** `harbor lint` exits 0; `002-CheckLeaks.py` passes on `spike/m2-verifier` (8 files) and `jobs/` (172 files). **[T]** | `spike/m2-verifier/`, `jobs/m2-*` |

### M3 exit checks (closed 23 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Agent base image & `agentrsi` CLI | **Pass.** `scripts/build_agent_base.py` builds `localhost/agentrsi-agent-base:latest` (`numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `rasterio`, `earthengine-api`, `xgboost`, `lightgbm`, `geopandas`, `/usr/local/bin/agentrsi`). `agentrsi --help` verified inside container. **[T]** | `scripts/build_agent_base.py`, `templates/agentrsi_cli.py` |
| Builder, runner & collector unit tests | **Pass.** 5/5 new tests pass (16/16 total suite): `.env` `#` stripping & gateway `/v1` rule, strict allowlist task build, handoff V-injection & disagreement warning, secret scan abort, smell flags, idempotent re-collect + promotion, spend cap guard. **[T]** | `tests/test_builder_collector.py`, `src/libbuilder/` |
| Live dry session (`s000-dry`) | **Pass.** `claude-code` / `claude-haiku-5-5` (`--max-turns 20 --max-budget-usd 0.50`, `/run/secrets/gcp-sa.json` mounted read-only) completed in `3m 59s` (`$0.012239`), produced valid `pipeline.py`, `REPORT.md`, `HANDOFF.md` (`0` missing sections), `0` smells: composite V lift `+0.057286` (`Δinc +0.002816`; AirHitam `+0.021679`, Palong `+0.068732`, Serting `+0.081446`, verifier `6.768s`). *(Retry after one transient `downloads.claude.ai` DNS error.)* **[T]** | `jobs/s000-dry/`, `sessions/s000-dry/` |
| Idempotent double collection | **Pass.** Re-collecting `s000-dry` twice leaves `state/events.jsonl` byte-identical, `daily_spend_usd` at `$0.012239`, and `t_attempts_used = 0` (dry flag). **[T]** | `state/events.jsonl`, `state/lineage.json`, `scripts/collect.py` |
| Handoff injection & leak checks | **Pass.** `build_task --parent s000-dry` injects verified V score (`+0.057286`) with zero T numbers (`ZERO_T_NUMBERS_IN_HANDOFF_TASK`). `harbor lint` exits 0 on `tasks/s000-dry`; `002-CheckLeaks.py` passes on `tasks/` (83 files), `jobs/` (197 files), and `sessions/` (8 files). (`harbor check` waived on `-e podman`: upstream `check-task-template` lacks `/app` in `python:3.13-slim`, failing `crun` before agent start.) **[T]** | `tasks/s000-dry/`, `src/libbuilder/build_task.py` |

### M4 exit checks (closed 23 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Exact-number scripted diff (`dashboard.py --verify`) | **Pass.** Generated `dashboard/index.html` from real `s000-dry` + `state/`; `verify_dashboard` reports `0 mismatches` against `reward.json`, `certification.json`, `lineage.json`, and `events.jsonl`. **[T]** | `dashboard/index.html`, `scripts/dashboard.py`, `src/libbuilder/dashboard.py` |
| Multi-epoch charts, CI bars & no-drift unit tests | **Pass.** 4/4 dashboard tests pass (20/20 total suite): real `s000-dry`, 2-epoch 5-session fixture (disjoint per-epoch `<polyline>` elements, `.epoch-band`/`.epoch-boundary`, `.ci-bar` + `+0.03` floor, zero `REPORT.md`/`HANDOFF.md` number bleed, HTML/XSS escaping), empty state, and `collect_session()` auto-regeneration. **[T]** | `tests/test_dashboard.py`, `src/libbuilder/collect.py` |
| Offline & T-isolation checks | **Pass.** Zero external HTTP(S) `<script>`/`<link>`/`url(...)` refs; `dashboard/index.html` confirmed gitignored; `AgentRSI/tasks/` holds zero `certification.json`, `dashboard/`, or T numbers; `002-CheckLeaks.py` passes across `tasks/`, `jobs/`, `sessions/`, and `docs/AgentPlan/`. **[T]** | `.gitignore`, `tasks/`, `002-CheckLeaks.py` |

### M5 observations — S1 (`261124-s001-open`, 24 Nov 2026)

| Check / Finding | Result | Evidence |
|---|---|---|
| Live S1 run (`codex` / `gpt-6.1-terra`, 1 h cap) | **Pass & Promoted.** `17m 34s` total (`5.03s` verifier), `$0.215`, `0` exceptions, `0` missing HANDOFF sections, `0` leak findings, `0` dashboard mismatches. Composite V `+0.118751` (`+0.064281` vs starter, `+0.052289` vs coords); Composite T Δ `+0.073438` (CI `[+0.036705, +0.109823]`, Δ vs coords `+0.084491`; AirHitam `-0.0174`, Palong `+0.2101`, Serting `+0.0276`). **[T]** | `sessions/261124-s001-open/`, `dashboard/index.html` |
| Signal forensic audit (on L only) | **Real synergy, not a fluke.** In Palong, `NDVI_lin` (`+0.030`), `NDVI_poly2` (`+0.030`), and `XY_poly2_no_NDVI` (`+0.097`) are all far below `NDVI_XY_poly2` (`+0.259` 80 m block CV, `+0.260` 4-quadrant far-field LORO): spatial terms detrend estate-wide background NDVI so within-locality NDVI becomes strongly predictive. **[T]** | S1 audit (`L_labelled.csv`) |
| Gap 1 — Static verifier incumbent (**CRITICAL**) | **Fixed & verified.** `build_task.py` stages the promoted incumbent (`submission/` + optional `datasets/`) into `tests/incumbent/`; `Dockerfile.verifier.tmpl` + `test.sh.tmpl` + `verifier/{grade,runner}.py` grade dynamically against `/tests/incumbent/submission`. Rebuilt `localhost/agentrsi-sealed:epoch1` and verified via live Harbor `oracle` on an S2-from-S1 task: `score = 0.118751`, `v_delta_vs_incumbent = 0.0`, `t_comp_delta = 0.0`, `gate_passed = False`. **[T]** | `verifier/runner.py`, `src/libbuilder/build_task.py`, `templates/{Dockerfile.verifier,test.sh}.tmpl` |
| Gap 2 — Early stopping & estate branching | **Fixed.** Added Research Depth & Persistence (explore >= 3–4 orthogonal directions before stopping) and Generalizable Recipe Design (prefer inner spatial-CV selection over hardcoding `ctx["estate"]`) rules to `templates/instruction_base.md`. **[T]** | `templates/instruction_base.md` |
| Gap 3 — Single-estate dominance on gate | **Fixed & verified (Option B).** Added `min_improved_estates: 2` to `config/verifier.yaml` and `verifier/metrics.py` (`evaluate_gate`), documented in `instruction_base.md`, and verified in unit tests + live `oracle` `certification.json` (`n_improved_estates = 0 < 2`). **[T]** | `config/verifier.yaml`, `verifier/metrics.py`, `tests/test_verifier.py` |
| Gap 4 — False-positive smell & slow CLI setup | **Smell fixed; CLI bake queued for M6.** Broadened `PROMPT_ECHO_RE` in `src/libbuilder/collect.py` (unit-tested: ignores prompt echo while still catching real `git clone` / access); baking Node 22 + `@openai/codex` + `@anthropic-ai/claude-code` into `agentrsi-agent-base:latest` is next in M6. **[T]** | `src/libbuilder/collect.py`, `tests/test_builder_collector.py` |

### M6 exit checks (closed 24 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Prebaked agent base image fast path | **Pass.** `scripts/build_agent_base.py` prebakes SHA-256-verified `node v22.23.5`, `npm 10.9.9`, `codex-cli 0.172.0`, `claude 2.1.331`, `ripgrep 13.0.0`. Harbor logged `Codex is already available at the requested version`; `agent_setup` dropped from `203.4s` (S1) to `0.8s` (S2, **254× faster**; total cold start `224.1s → 25.2s`). **[T]** | `scripts/build_agent_base.py`, `jobs/261124-s002-open/` |
| S2-from-S1 handoff, dataset pipeline & dynamic gate | **Pass.** `261124-s002-open` (`--parent 261124-s001-open`) extracted valid `/app/datasets/alos_winstats/features.parquet` (`7,220 × 403`, `14.1 MB`, native CRS + signed dB Yamaguchi + T3 `T11` masking, `datasets_audit.valid=True`). After mid-run agent self-`SIGTERM` (exit 143), verifier dynamically graded against S1 (`v_delta_vs_incumbent = -0.098964`, `t_comp_delta = -0.100262`, `n_improved_estates = 0 < 2`) and safely rejected promotion. **[T]** | `sessions/261124-s002-open/`, `state/lineage.json` |
| Clean dry-session collection, smells, dashboard & leaks | **Pass.** Collected with `--dry-session` (`t_attempts_used` kept at `1 / 20`, incumbent kept at `261124-s001-open`, spend `$0.348` recorded, `alos_winstats` archived), `smells = []` (prompt-echo fix verified), `dashboard.py --verify` `0 mismatches`, `002-CheckLeaks.py` `PASS`. **[T]** | `sessions/261124-s002-open/meta.json`, `dashboard/index.html` |

### Focus-lens & dataset expansion checks (`s003`–`s006`, 24 Nov 2026)

| Check / Session | Result | Evidence |
|---|---|---|
| Concurrent 3-lens Podman run (`s003`, `s004`, `s005`) | **Pass.** Added per-task CPU/RAM CLI overrides (`build_task.py`, `run_session.py`) and atomic `stage_secrets.py`. Ran `data-expansion`, `literature`, and `ablation` simultaneously inside the 12-CPU / 15.2 GiB Podman envelope with zero OOMs or exceptions (`$0.303`, `$0.236`, `$0.284`), collected sequentially (`--dry-session`), `smells = []`, `0` dashboard mismatches. **[T]** | `sessions/261124-s00{3,4,5}-*`, `src/libbuilder/{build_task,run_session,stage_secrets}.py` |
| Tightened `data-expansion` live GEE extraction (`261124-s006-data-expansion`) | **Pass.** Extracted two new 7,220-row, 0-NaN GEE datasets (`gee_s2_stress`: 54 cols, `970 KB`; `gee_topo`: 11 cols, `179 KB`) via read-only GEE SA alongside archived `alos_winstats` (`15.99 MB` total, `datasets_audit.valid=True`). Graded via sealed `localhost/agentrsi-sealed:epoch1` (`V +0.108501`, `T comp_delta -0.009851`, `gate_passed = False`) and collected (`t_attempts_used = 2 / 20`, incumbent remains `261124-s001-open`). **[T]** | `sessions/261124-s006-data-expansion/`, `templates/lenses/data-expansion.md` |

### M7 exit checks (closed 24 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Isolated `sidecar-venv` & V-only bundle | **Pass.** `scripts/build_verifier_base.py` installs `mcp==2.3.0` + `uvicorn==0.54.0` inside root-only `/opt/agentrsi/sidecar-venv` (`chmod -R go-rwx`), leaving `/usr/local/bin/python` untouched for Epoch-1 v1 grading; `_load_v_only_bundle` strips all `part != "V"` labels immediately. **[T]** | `verifier/inrun_server.py`, `scripts/build_verifier_base.py` |
| Auth, budget lock, cached CSPRNG noise & unit tests | **Pass (10/10 tests).** Verified unauthenticated `/health` (`200`), fail-closed Bearer `401` (`hmac.compare_digest`), `threading.Lock` budget cap (`15`), `content_sha256`-cached `secrets.SystemRandom().gauss` noise (`σ = 0.005`), jailed `/shared` traversal/symlink rejection, and `runner` `/sealed` denial. **[T]** | `tests/test_inrun_verifier.py` |
| Live Harbor multi-container sidecar trial (`261124-m78-live-open`) | **Pass.** `verifier-sidecar` passed `service_healthy` on Podman Compose; `claude-haiku-5-5` called `verifier_budget()` (`15/15`) and `score_v(shared_submission_dir="sub_smoke")` (`4.84s`, noisy V lift `+0.121518` vs exact `+0.118751`, budget `14/15`). Harbor collected `sidecar/queries.jsonl`, `collect.py` archived `sidecar_queries.jsonl` (`inrun_queries_used: 1`), secret scan & `002-CheckLeaks.py` passed (`0` token/T leaks). **[T]** | `templates/{docker-compose.inrun.yaml,Dockerfile.sidecar}.tmpl`, `src/libbuilder/{build_task,run_session,collect}.py` |

### M8 exit checks (closed 24 Nov 2026)

| Exit check | Result | Evidence |
|---|---|---|
| Broker guardrails, relay & unit tests | **Pass (14/14 M8 + 5/5 wiring tests; 54/54 full suite).** Narrow 4-tool MCP surface (`colab_run`, `colab_status`, `colab_fetch`, `colab_stop`), `agentrsi_mcp_relay.py` baked into `agentrsi-agent-base:latest`, blocks `drive.mount`, `google.colab.auth`, `authenticate_user`, `gcloud`/`gsutil`, `sar-oilpalm`, `Leejames2305`, metadata IPs, non-allowlisted GPUs (`A100` off by default), unsafe `pip` specs, traversal/symlinks; strips host secrets via `sanitized_colab_env()`. **[T]** | `src/colab_broker/`, `templates/agentrsi_mcp_relay.py`, `tests/{test_colab_broker,test_mcp_wiring}.py` |
| Live Colab Tesla T4 extraction via Podman & Harbor | **Pass.** Host `colab_broker.py --check-auth` confirmed host-only `google-colab-cli` auth. Both standalone Podman (`test_m8_real_colab.py`) and full live Harbor session (`261124-m78-live-open`, `$0.0327`) provisioned a real **Tesla T4** (`cuda_available: true`, `device_name: "Tesla T4"`), downloaded JSON output artifacts to `/run/agentrsi/colab-io/out/<job_id>/`, and logged `colab_vm_start` / `colab_job_run` / `colab_vm_stop` in `events.jsonl`. **[T]** | `scripts/colab_broker.py`, `src/colab_broker/{backend,server}.py` |
| Zero token / `sar-oilpalm` exposure & VM auto-teardown | **Pass.** Inside the live Colab T4 VM: `leaked_env_keys: []`, `/content/drive/MyDrive` absent, and `storage.Client().list_blobs("sar-oilpalm")` failed (`RefreshError`); inside container: `/root/.config/colab-cli` absent and `drive.mount`/`A100`/`sar-oilpalm` rejected by broker guardrails. Post-shutdown `colab sessions` confirmed `[colab] No active sessions found on server.` and socket/`io_dir` removed. **[T]** | `src/libbuilder/run_session.py` (`finally:`), `src/colab_broker/guardrails.py` |

### Next actions

**M0–M8 are all closed (55/55 unit tests, live Harbor M7+M8+T4 verified, `002-CheckLeaks.py` `PASS`). Next: run 4 h default modeling sessions (M7 in-run verifier + M8 Colab `T4`/`L4` enabled by default, `$10` spend cap) over all 3 archived covariate stacks (`alos_winstats`, `gee_s2_stress`, `gee_topo`).**

**Parked items for upcoming milestones**
- **Deferred:** Yamaguchi SAR in the base table (dB-like values; deferred until a conversion rule is chosen, raw scenes already exposed to agents in `rasters/`).

---

## Observations

### Data

| Estate | Healthy | Unhealthy | Middle | Prevalence (H+U) | 80 m blocks | NDVI Cohen's d (H vs U) |
|---|---|---|---|---|---|---|
| AirHitam | 2351 | 160 | 0 | 6.4% | 70 | `-0.511` |
| Palong | 1971 | 200 | 493 | 9.2% | 47 | `-0.301` |
| Serting | 1513 | 154 | 378 | 9.2% | 29 | `-0.428` |

- **Split feasibility (`003-Calibration.py feasibility`, Middle travels with its block).** Expected redraws to satisfy both Unhealthy minimums (`50 L / 35 V / 35 T`) and ±10 pt tree-share tolerance (50,000 draws/size):

  | Block size | AirHitam (blocks · draws) | Palong (blocks · draws) | Serting (blocks · draws) |
  |---|---|---|---|
  | 220 m | 15 · 9.1 | 6 · 85.2 | 4 · **infeasible** |
  | 165 m | 19 · 6.1 | 12 · 7.1 | 9 · 12.9 |
  | 110 m | 42 · 3.1 | 24 · 3.5 | 15 · 9.9 |
  | **80 m (chosen)** | **70 · 2.0** | **47 · 2.0** | **29 · 6.2** |
  | 55 m | 129 · 1.4 | 86 · 1.3 | 54 · 2.6 |

  Serting's 154 Unhealthy trees sit in fewer clusters, so it is the binding constraint at every size.
- **Leakage check (`003-Calibration.py leakage`, NDVI-only balanced LogReg, 5-fold × 5 seeds, pooled OOF PR-AUC minus random 5-fold):**
  - AirHitam: `220m +0.0003`, `165m -0.0010`, `110m +0.0006`, **`80m +0.0000`** (baseline OOF PR `0.1289`, lift `+0.0652`).
  - Palong: `220m -0.0219`, `165m -0.0047`, `110m -0.0011`, **`80m -0.0021`** (baseline OOF PR `0.1268`, lift `+0.0347`).
  - Serting: `220m n/a` (<5 blocks), `165m -0.0047`, `110m -0.0015`, **`80m -0.0017`** (baseline OOF PR `0.1473`, lift `+0.0549`).
  Univariate NDVI is a global level shift, so its spatial-vs-random gap is near zero (`<= 0.005`) at 80–165 m across all estates. Palong and Serting average ~10 clear S2 observations per tree vs ~26 on AirHitam, so their composites are noisier.
- **Spatial clustering.** Chance that an Unhealthy tree's 10 nearest neighbours are Unhealthy: `0.119 / 0.336 / 0.255` (AirHitam / Palong / Serting) vs `0.060 / 0.060 / 0.070` for Healthy (`PathForwardTrials.md`). Coordinates-only is a strong baseline and is reported alongside every submission.
- **POC reference lifts** (broad holdout, best SAR + spatial fusion): `+0.062 / +0.150 / +0.071` over prevalence (PR-AUC `0.125 / 0.242 / 0.164`). Adding SAR to coordinates gave `+0.016` to `+0.053`, with CIs crossing zero in several cases.
- **Middle** (`MeanSampling.md` §7b): sits on the disease front (median 14.6 m from nearest Unhealthy vs 26.5 m / 20.7 m for Healthy on Palong / Serting). Folding it into positives dilutes signal and lowers ROC, so it stays visible in L (for optional agent use) and unmarked/unscored in `unlabelled.csv`.

### Gate calibration **[S]**

**Why a T-only floor gate exists (AirHitam winner's-curse study, `misc/temp/sim_winners_curse.py`):** selecting the best of `k` random feature-subset LogReg candidates on V inflates apparent V gain while true T gain stays near 0 (`k=1: -0.008 / -0.006`; `k=10: +0.023 / +0.002`; `k=30: +0.033 / +0.005`; `k=60: +0.038 / +0.007`). The naive V-CI + T-non-degrading gate falsely adopted **21–25%** of null ideas at `k=10..30`, whereas adding a T composite floor `d >= 0.03` dropped single-estate AirHitam false adopts to 4–6%.

**Three-estate composite gate calibration (`src/DataPipeline/003-Calibration.py gate`, 23 Nov):**
6,349 scored trees, **120 resampled 80 m splits** (never touching frozen T), 200 paired block-bootstrap draws per test, 17 candidate features (multi-date SAR `260610` + `261014` + `B4, B8, clear_count`, NDVI forced in 50% of draws). Gain arms add `gamma * y` to standardized incumbent T scores, with `gamma` calibrated over 80 resampled splits (`+0.02 → 0.128`, `+0.04 → 0.248`, `+0.06 → 0.359`).

| Arm | `comp_only` | Adopt (`-0.03 / -0.02 / -0.04`) | Mean comp ΔPR-AUC (SD) | Mean worst-estate Δ | P(degrade `< -0.03 / -0.02 / -0.04`) |
|---|---:|---:|---:|---:|---:|
| Null, V-winner `k=1` | 0.000 | 0.000 / 0.000 / 0.000 | `-0.0316` (`0.0260`) | `-0.0628` | `0.775 / 0.817 / 0.658` |
| Null, V-winner `k=10` | 0.000 | 0.000 / 0.000 / 0.000 | `-0.0165` (`0.0199`) | `-0.0381` | `0.492 / 0.633 / 0.392` |
| Null, V-winner `k=30` | 0.000 | 0.000 / 0.000 / 0.000 | `-0.0166` (`0.0218`) | `-0.0372` | `0.458 / 0.592 / 0.350` |
| Gain `+0.02` | 0.000 | 0.000 / 0.000 / 0.000 | `+0.0200` (`0.0032`) | `+0.0144` | `0.000 / 0.000 / 0.000` |
| Gain `+0.04` | 0.967 | 0.967 / 0.967 / 0.967 | `+0.0402` (`0.0059`) | `+0.0304` | `0.000 / 0.000 / 0.000` |
| Gain `+0.06` | 1.000 | 1.000 / 1.000 / 1.000 | `+0.0605` (`0.0081`) | `+0.0467` | `0.000 / 0.000 / 0.000` |

- **Power & false adopts:** The `0.03` composite floor reliably certifies `+0.04` gains (`96.7%`) and rejects `+0.02` (`0%`). Zero of 120 null draws passed even the composite-only gate (`comp_only = 0.000`, 95% upper bound ~`2.5%` by rule of three).
- **Non-degradation (`-0.03`):** Adoption is identical across `-0.02`, `-0.03`, and `-0.04` on all synthetic arms; `-0.03` stays pre-registered and matches the floor scale.
- **Caveats:** Synthetic gain arms use a uniform oracle shift without per-estate candidate noise, so power (`96.7%` at `+0.04`) is a ceiling and non-degradation cost on real noisy candidates remains open until M5.

### Harbor design **[V]**

- A task is a directory: `instruction.md`, `task.toml`, `environment/`, `tests/`, optional `solution/`.
- **Separate verifier:** grading runs in a fresh sandbox built from `tests/Dockerfile`. Data baked there never reaches the agent. Only declared `artifacts` cross over (sanitized: no setuid, no outward symlinks, no special files). Separate verifiers do not inherit runtime mounts, so the service-account key cannot reach grading.
- **Reward** goes to `/logs/verifier/reward.txt` or `reward.json` (richer, wins if both exist). **`reward.json` must be a flat dict of finite `int | float` only** (`_parse_reward_json` rejects strings, nested dicts, and lists) **[V]/[T]**; sibling files (`/logs/verifier/v_report.json`, `certification.json`) are downloaded automatically alongside it **[T]**.
- **Claude Code:** gateway via `ANTHROPIC_BASE_URL` plus bearer `ANTHROPIC_AUTH_TOKEN`; options `max_turns`, `max_budget_usd`, `reasoning_effort`, `append_system_prompt`, `disable_web_search`; defaults to `--permission-mode bypassPermissions`; Anthropic Messages only.
- **Codex:** Responses API only. `codex.py` injects `OPENAI_API_KEY` and `OPENAI_BASE_URL`. **[T]** The gateway returns 404 for the Responses WebSocket, so Codex retries 5× (~10 s) then falls back to HTTPS, which works.
- **`--api-key-name`** names the **host** env var holding the key (`agent_auth_resolution.py` maps it to the agent's own variable). Both trials passed with their matching names (`OPENAI_API_KEY` for Codex, `ANTHROPIC_AUTH_TOKEN` for Claude Code).
- **Network modes** per phase: `public`, `no-network`, `allowlist` (restricted modes need an nftables sidecar; we use `public` everywhere).
- **Flags:** `--base-url`, `--api-key-name`, `--api-format`, `--ae`/`--ak`, `--mounts`, `-e`, `--override-cpus`/`--override-memory-mb`, `--dry-run`, `--print-config`. `harbor lint` is static; `harbor check` is an LLM rubric. `regrade` re-runs a newer verifier on recorded outputs.
- **`-e podman` (harbor#398):** using `docker-compose` as the compose provider avoids backend mis-detection.

### Harbor and Podman on this machine **[T]** (11 Nov 2026)

- **Engine.** Podman 6.1.4, API 1.44, crun, netavark, overlay, cgroups v2. Rootful socket `/mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock`. 15.2 GiB / 12 CPUs shared with WSL and Windows (~11.5 GiB free at test time).
- **Rootless vs rootful.** Rootless lacks delegated cgroup controllers (cannot enforce CPU/memory limits). Rootful has `cpuset, cpu, io, memory, pids`. Mitigations: no `privileged`, no extra capabilities, no host mounts except one read-only secret, `harbor lint`.
- **WSL distro boundary.** Engine runs in a separate WSL distro: bind-mounting `/home/lee/...` fails, while `/mnt/wsl` tmpfs is shared. Data/code reach the sandbox via build context; secrets use `/mnt/wsl/...` or env injection.
- **Kernel.** `CONFIG_NF_TABLES=y`, so restricted verifier networking stays possible later.

**Operational rules from M0:**
1. **Base image needs bash:** use `python:3.12-slim-bookworm` (not `alpine`).
2. **Separate verifiers need tests baked in:** `tests/Dockerfile` must `COPY test.sh /tests/`.
3. **Artifacts must be declared** in `task.toml`, or the verifier cannot see agent outputs.
4. **Regrade needs `-e podman`** and task names matching the source job's trial names.
5. **`cpus` is a CFS quota, not a cpuset:** `os.cpu_count()` still reports 12 inside `--cpus=1`, so pass worker counts via `ctx` and env.
6. **Put `.tools/bin` first on `PATH`** so `podman compose` invokes Linux `docker-compose`.
7. **Runner env:** set `PATH`, `XDG_RUNTIME_DIR`, `CONTAINER_HOST`, `DOCKER_HOST`, and `PODMAN_COMPOSE_WARNING_LOGS=false` (without the last, `podman compose` prints a banner to stdout that breaks Harbor's package-manager probe during Codex install).
8. **Keep install errors visible** in job records (transient `downloads.claude.ai` TLS/DNS errors occurred once in M0 and once in M3 and succeeded on rerun; also guard memory on large downloads after an M0 WSL memory-pressure reboot).
9. **`harbor check` fails under `-e podman` (tested on `0.25.0` and `0.27.0`)** **[T]**: its internal `check-task-template/task.toml` hardcodes `docker_image = "python:3.13-slim"` with `workdir = "/app"`. In `0.27.0`, `docker.py` moved `_upload_environment_dir_after_start()` ahead of `ensure_dirs()` (fixing Docker), but for Podman (`supports_compose_wait=False`) `_wait_until_compose_exec_ready()` still runs one line earlier and executes `exec -w /app main bash -c true` before `/app` is created, so `crun` rejects it (`chdir to /app: No such file or directory`). Always create `/app` in our own task Dockerfiles and rely on `harbor lint` + `002-CheckLeaks.py`.
10. **Harbor agent `--ak` flags differ by agent** **[V]/[T]**: `claude-code` supports `max_turns` and `max_budget_usd`, whereas `codex` rejects both (`CodexOptions` supports `reasoning_effort`, `reasoning_summary`, `web_search`). `run_session.py` gates them accordingly.
11. **Harbor MCP env prompt & Compose lint on Podman-only hosts** **[V]/[T]**: Harbor `0.27.0` prompts interactively (`Proceed? (Y/n):`) whenever `[[environment.mcp_servers.headers]]` reads a host env var (`AGENTRSI_VERIFIER_MCP_TOKEN`); `run_session.py` passes `--yes` so runs never block. Also, `harbor lint` invokes the `docker compose` CLI plugin (absent on Podman-only hosts) when `environment/docker-compose.yaml` is present; validate M7 Compose tasks using `AgentRSI/.tools/bin/docker-compose config` + `002-CheckLeaks.py`.

---

## Risks

| Risk | Mitigation |
|---|---|
| Label or V/T-split leakage into agent image | Allowlist copy, merged `unlabelled.csv` (no V/T split, no IDs), and `002-CheckLeaks.py` canary on every build (verified 0 false positives on `prior/` docs and GeoTIFF rasters). |
| Agent code reads sealed labels at grading | Separate sandbox, unprivileged subprocess, root-only label files, trusted scoring process. Public verifier network is accepted for a throwaway container. |
| Pipeline depends on the internet at grading | Instruction rule, URL/network-call grep in `collect.py`, optional offline `regrade` on nominees. |
| **Holdout overuse and false adoption** | T-only certification of one nominee, `0.03` floor, 20-attempt budget, epoch refresh, paired deltas only. |
| Gate cannot see small gains | Stated openly; accumulate small gains in `REPORT.md` and certify in batches. |
| **Block bootstrap thin on Serting** | 80 m gives 70 / 47 / 29 blocks (10 T blocks on Serting, 9 with positives). |
| **Spatial location mistaken for model skill** | Coordinates-only baseline reported per estate. |
| **Dashboard shows optimistic V as certified** | V labelled "search signal"; T shown separately with CI and `+0.03` line; never merged. |
| Dashboard numbers drift from verifier | Reads verifier/collector JSONs only; M4 diffs page against them. |
| **Credential theft** (agent on open web holds SA key) | Least-privilege **GEE-only** SA (`Keys_SAEarthEngine.json`, Earth Engine viewer only — verified HTTP 403 on `sar-oilpalm` GCS and IAM), read-only mount outside build context, instruction forbids printing, `collect.py` secret scan fails collection, `.secrets/` and `.env` gitignored. |
| **Colab OAuth/ADC exposure or credit burn** (M8) | Never pass `google-colab-cli` tokens into containers (`cloud-platform` scope exposes personal GCP/GCS). Host-side WSL MCP broker only; block `colab auth` and Drive mounts; enforce GPU allowlist, max 1 VM, per-job/session caps, and auto-teardown. |
| GEE quota or cost from unattended agent | Instruction asks for point sampling at tree coordinates; monitor quota/billing. |
| Package install runs untrusted code / PyPI drift | Unprivileged user with unreadable labels; exact pins required and `pip freeze` recorded. |
| Rootful engine containment | No privileged mode, no extra caps, no host mounts except read-only secret, `harbor lint`. |
| Cost runaway on third-party gateway | `max_budget_usd`, `max_turns`, spend-capped key, daily cap ledger + kill switch (enforced in M3). |
| Podman/Harbor instability (harbor#398) | Pinned `harbor==0.27.0`, `docker-compose` provider, Docker Desktop fallback (only `-e` changes). |
| Resource contention on 12-core / 15 GiB WSL box | One session at a time, enforced rootful limits, manual session launches (no cron/scheduler). |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 3.1 | 25 Nov 2026 | Updated default session profile (`config/session.yaml`, `.env.example`, `run_session.py`, `AgentRSI/README.md`, `tests/test_mcp_wiring.py`): `4 h` agent timeout (`14400 s`, down from `8 h`), M7 in-run verifier enabled by default (`inrun_verifier: true`), M8 Colab GPU broker enabled by default (`colab_broker.enabled: true`, `T4`/`L4` allowed, `A100` restricted), and `$10` cost/daily spend cap (`max_budget_usd: 10.0`, `DAILY_SPEND_CAP_USD=10`). Verified 55/55 host unit tests and `002-CheckLeaks.py` `PASS`. |
| 3.0 | 24 Nov 2026 | Closed M7 (In-run verifier v2 MCP sidecar) & M8 (Host Colab GPU offload MCP broker), and recorded 4-lens runs (`s003`–`s006`, archiving `alos_winstats` + `gee_s2_stress` + `gee_topo`). Built `verifier/inrun_server.py` (`/opt/agentrsi/sidecar-venv` isolated), `src/colab_broker/`, `templates/agentrsi_mcp_relay.py`, and full Harbor/Compose/runner/collector wiring (`--yes`, `colab exec --timeout`, `~/.local/bin` fallback); verified 54/54 unit tests, live Podman M7+M8 smoke tests, real Colab Tesla T4 execution + `sar-oilpalm` isolation + VM auto-teardown, `002-CheckLeaks.py` `PASS`, and updated `AgentRSI/README.md`. |
| 2.9 | 24 Nov 2026 | Closed M6: verified exact prebaked startup speedup (`agent_setup` `203.4s → 0.8s`, 254×; total cold start `224.1s → 25.2s`), collected `261124-s002-open` with `--dry-session` (archived validated `14.1 MB` `alos_winstats` `7,220 × 403` dataset, kept `t_attempts_used = 1 / 20` and incumbent `261124-s001-open`, confirmed `smells = []`, `dashboard.py --verify` `0 mismatches`, `002-CheckLeaks.py` `PASS`). |
| 2.8 | 24 Nov 2026 | Closed M5 (all 4 audit fixes live-verified via S2-from-S1 `oracle` + 25/25 unit tests). Started M6: prebaked SHA-256-verified Node `v22.23.5`, `npm 10.9.9`, `ripgrep 13.0.0`, `codex-cli 0.172.0`, and `claude 2.1.331` into `localhost/agentrsi-agent-base:latest` (Harbor `is_requested_version_installed()` fast-path skipped reinstall completely; cold start to `codex exec` dropped from `224s` in S1 to `~31s` in S2) and launched live S2 (`261124-s002-open`, `--parent 261124-s001-open`, 1 h). |
| 2.7 | 24 Nov 2026 | Recorded M5 S1 (`261124-s001-open`, 1 h pilot, promoted at T `+0.0734`, V `+0.1188`, `$0.215`) and forensic audit findings (Palong NDVI×spatial synergy verified on 4-quadrant LORO; caught critical static-incumbent verifier bug, early stopping, single-estate dominance, and prompt-echo smell). Updated M6 to bake `nodejs`/`npm`/Node 22/`ripgrep`/`@openai/codex`/`@anthropic-ai/claude-code` into `agentrsi-agent-base:latest` and dropped cron / Task Scheduler / `daily.sh` in favour of manual session launches. |
| 2.6 | 24 Nov 2026 | Overhauled `templates/instruction_base.md`, `templates/lenses/*.md`, and `src/libbuilder/build_task.py` against `docs/AgentPlan/AGENTS.md` and `harbor-framework/terminal-bench-science` standards (`rubrics/task-implementation.toml`, `CONTRIBUTING.md`, merged Earth/Ecology/ML tasks): added operational triage context & outdated-POC parity warning, authoritative GeoTIFF band-order/CRS/dB-Yamaguchi/T3 specs, explicit agent-vs-verifier package/runtime divergence, block-honest spatial-CV & non-degradation discipline, dynamic timeout/cap placeholders with unreplaced-placeholder guard, and canonical timeout closing line; 22/22 unit tests, 4-lens `harbor lint`, and leak checks pass. |
| 2.5 | 23 Nov 2026 | M4 closed (Dashboard) & added `AgentRSI/README.md`: built `src/libbuilder/dashboard.py`, `scripts/dashboard.py --verify`, and `collect.py` auto-refresh generating self-contained offline `dashboard/index.html` (separate V/T SVG charts, epoch boundary breaks, bootstrap CI bars, `+0.03` floor, events + `HANDOFF.md`/`REPORT.md` inspector); 20/20 unit tests and leak checks pass. |
| 2.4 | 23 Nov 2026 | Upgraded pinned Harbor `0.25.0 -> 0.27.0` and refreshed `uv.lock` (147 packages). Re-verified 16/16 unit tests, `harbor lint`, `run_session.py --dry-run`, M2 `oracle` separate-verifier trial (score `0.054469`), and leak checks. Empirically confirmed upstream `harbor check -e podman` `crun: chdir to /app` failure persists in `0.27.0` (`_wait_until_compose_exec_ready()` still precedes `_upload_environment_dir_after_start()`). |
| 2.3 | 23 Nov 2026 | M3 closed (Builder, runner, collector): `localhost/agentrsi-agent-base:latest`, `agentrsi` CLI, `config/session.yaml`, `templates/`, `src/libbuilder/`, `scripts/`, 16/16 unit tests, live `s000-dry` session (`claude-haiku-5-5`, V `+0.057286`, `$0.0122`), idempotent double collect, handoff & leak checks. Documented Harbor `0.25.0` `harbor check` Podman/`crun` bug and per-agent `--ak` support. |
| 2.2 | 23 Nov 2026 | M2 closed (Verifier v1): `config/verifier.yaml`, `starter/`, `verifier/` library, `scripts/build_verifier_base.py` (`localhost/agentrsi-sealed:epoch1`), 11 unit tests, `spike/m2-verifier` Harbor Oracle/Nop trials, and 5 container adversarial/pip checks. Recorded Harbor `_parse_reward_json` flat-dict rule (`reward.json` + sibling `v_report.json`/`certification.json`). |
| 2.1 | 23 Nov 2026 | Verified new GEE-only SA (`Keys_SAEarthEngine.json`: GEE OK, GCS `sar-oilpalm` & IAM HTTP 403) and updated `common.py`. Designed M8 optional Colab GPU offload via host-side WSL MCP broker + Harbor `[[environment.mcp_servers]]` (keeping `cloud-platform`-scoped user tokens strictly off containers). |
| 2.0 | 23 Nov 2026 | M1 hardened after audit: `config/split.yaml` wired as single source of truth in `common.py`; multi-date ALOS-2 SAR (`260610` + `261014`, auto-discovered) and renamed `HAlpha_H/A/alpha` in base table; V + T merged into single `unlabelled.csv` (no V/T exposure); `002-CheckLeaks.py` fixed for `prior/` docs and binary rasters + 4 seeded controls; `003-Calibration.py` fixed to resample 80 m splits (never touching frozen T) and pooled OOF leakage; stale `tree_id` and single-estate starter references cleaned up. |
| 1.9 | 20 Nov 2026 | M1 closed. SAR band order confirmed from POC/SNAP pipeline. M2 not started. |
| 1.8 | 20 Nov 2026 | Non-degradation `-0.03` confirmed; split CSV host exposure accepted; research repo blocked for agent only; re-extraction matched frozen features. |
| 1.7 | 20 Nov 2026 | M1 pipeline consolidated into `src/DataPipeline/`. Pseudonymous row keys removed in favour of `(estate, Long, Lat)`. |
| 1.6 | 20 Nov 2026 | Initial M1 exit checks met; sealed master staged in `.sealed/verifier_context/`. |
| 1.5 | 20 Nov 2026 | M1 features from raw sources only; split frozen at 80 m. |
| 1.4 | 19 Nov 2026 | M0 closed: Codex and Claude Code gateway trials pass; `PODMAN_COMPOSE_WARNING_LOGS=false` workaround. |
| 1.3 | 17 Nov 2026 | Middle visible in L but unscored; dashboard paths pinned inside `AgentRSI/`. |
| 1.2 | 17 Nov 2026 | Equal estate trust & reward weights, composite T gate, proportional splits, `/v1` rule per API format. |
| 1.0 | 11 Nov 2026 | Verifier network public; gate floor `0.03` and 20-attempt budget locked; M0 results added. |
