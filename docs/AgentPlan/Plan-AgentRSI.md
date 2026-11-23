# AgentRSI Plan

**Version:** 2.0 (23 Nov 2026) | **Phase:** M0 and M1 closed (M1 hardened 23 Nov) | **Paper:** `docs/Reference/...RSI4EO_2610.18845v2.pdf`

**Goal.** Turn this project into an RSI4EO-style loop. A coding agent builds the oil-palm triage model in a sandbox, a sealed verifier scores it, and each day's session starts from what the last one found. Humans design the harness and verifier. Agents build the models.

**Where we are.** M0 and M1 are closed. All M0 gateway and Podman checks pass. M1's data pipeline (`src/DataPipeline/`) reads the frozen 80 m split from `config/split.yaml` (single source of truth), extracts multi-date ALOS-2 SAR (`260610` and `261014`, auto-discovered) plus Sentinel-2 NDVI from raw sources, emits a combined `L_labelled.csv` and merged `unlabelled.csv` (V + T, no IDs, no V/T exposure), passes hardened leak checks (including `prior/` docs and GeoTIFF rasters), and validates the composite gate over 120 resampled 80 m splits. **M2 is next and waits for go-ahead.**

Evidence tags: **[V]** read in Harbor docs or source | **[T]** tested on this machine | **[S]** simulation on the real estate tables | **[?]** unverified.

---

## Overview

| RSI4EO concept | Here |
|---|---|
| Harness H (data, libs, budget) | Combined multi-date SAR + S2 NDVI tables and all ROI-masked ALOS-2 rasters for 3 estates, prior POC reports, the incumbent pipeline, GEE/GCS access, a Python geo/ML stack. **CPU only.** |
| Verifier V (sealed, scores only) | Sealed held-out blocks for all three estates, each with a validation part (V) and a certification part (T). Returns scores, never labels. |
| Pipeline pi | `submission/`: `fit_predict(labeled, unlabeled, ctx) -> risk scores`, a **training recipe** (the product trains on about 1000 labels in a new field and predicts the rest). |
| Few-label task | Same shape: a labelled slice (L) per estate, the rest (V + T) merged into one unlabeled pool. |
| Recursion inside a task | Agent-driven within the 8 h session. Pseudo-labelling on the unlabeled pool is allowed; nothing is prescribed. |
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
| **Promotion** | One **composite T test** on the same equal weights, plus a per-estate non-degradation check. One nominee per session. Effect floor `d >= 0.03`, T-attempt budget 20. | Confirmed 20 Nov; re-verified over 120 resampled 80 m splits (23 Nov). |
| **Non-degradation** | An estate fails if its T ΔPR-AUC against the incumbent is below `-0.03` (pre-registered 19 Nov, confirmed 20 Nov). Blocks adoption only. Tested at `-0.02`, `-0.03`, `-0.04` on resampled splits: adoption is identical on synthetic arms. | Confirmed at `-0.03`. Open: real-candidate cost. |
| **Dashboard** | Static self-contained HTML inside `AgentRSI/`: `AgentRSI/scripts/dashboard.py` writes `AgentRSI/dashboard/index.html`. Numbers come from verifier and collector files only; agent reports supply event context only. | Your call (17 Nov). Design in Design › Dashboard. |
| **Verifier phases** | v1 grades once at session end. v2 adds an in-run sealed query tool with a query budget. | |
| **Agents** | Claude Code and Codex through a third-party gateway. Agent, model, base URL and key name are run-time parameters. Current default: Codex, `gpt-6.1-terra`, OpenAI Responses. | |
| **Budget and network** | 8 h agent timeout. Agent has public internet, with one blocked site: this research repo, `https://github.com/Leejames2305/SAR-OilPalm-Detection`. **Verifier network is fully public** (throwaway container). Label protection is the unprivileged subprocess plus root-only label files, not the network. | Your calls (11 Nov, 20 Nov). Block covers the agent only; verifier stays unblocked. |
| **Packages** | The agent ships a pinned `requirements.txt` and the verifier installs it from PyPI. No allowlist, no approval flow. | Your call (11 Nov). |
| **Runtime** | Rootful Podman from WSL, Docker Desktop as fallback. Harbor pinned at `0.25.0`. | Rootless cannot enforce CPU or memory limits. |
| **Focus** | Open-ended research, explicitly including data expansion. Optional lenses (`literature`, `data-expansion`, `ablation`) bias the instruction. | |
| **Data egress** | ROI-masked ALOS-2, tree coordinates and labels go to the inference provider. The GCS/GEE service-account key is passed to the agent. | Your call (11 Nov). |
| **Layout** | Everything AgentRSI builds, runs or generates lives under `AgentRSI/`, the dashboard included. Generated tasks live in `AgentRSI/tasks/<id>/`. Harbor's file is `instruction.md` (singular) **[V]**. | |
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
state/lineage.json + sessions/s(N-1)/HANDOFF.md
        |  build_task.py   (templates + agent_view + rasters + prior knowledge)
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
  pyproject.toml             uv project, pinned deps (harbor==0.25.0, pyyaml==6.0.3, ...)
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
  scripts/                   thin CLIs: build_task  run_session  collect  regrade  dashboard  daily
  state/lineage.json         incumbent, scores, epochs, spend ledger, T-attempt counter (committed)
  state/events.jsonl         append-only event log (committed)
  sessions/sNNN/             COLLECTED: REPORT.md HANDOFF.md submission/ datasets/ reward.json
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

[task]            name = "agentrsi/s003-open"
[agent]           timeout_sec = 28800.0            network_mode = "public"
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
- **`instruction.md` sections:** Mission | Harness map | Pipeline contract | Deliverables with absolute paths | Freedom (internet, papers, GEE/GCS, pseudo-labelling, data expansion) | Constraints (CPU only, 28800 s, never touch `/tests`, never try to recover sealed labels, never print credentials) | Checkpoint rule | Nominee (name the one pipeline to certify) | Handoff from the previous session (auto-injected). The paper's "do not use online solutions" line is dropped on purpose.
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

**Outputs.**
- `reward.json` (V, visible to the agent at session end): composite `score`; per estate `pr_auc_lift`, `pr_auc`, `prevalence`, `recall_at_25`, `enrichment_top10`, `roc_auc`, V CI bounds; paired delta vs the incumbent; delta vs coordinates-only; runtime; `pip freeze`.
- `certification.json` (T, host-only, never injected into any instruction or `tasks/` directory): composite T result with its CI, per-estate T deltas, the non-degradation values, the outcome and the T-attempt count.

**Checks before trusting it (M2 exit).** The starter incumbent reproduces a fixed composite and per-estate score; `nop` scores neutral on all three estates; adversarial pipelines (read `/tests`, write the reward, import labels, run forever, malicious package build code) fail safely; a pipeline that fetches data at grading time is flagged.

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

### Runtime and cost

- **Sizing.** The engine has 15.2 GiB and 12 CPUs, shared with WSL and Windows. One session at a time (`6 + 4 GiB` agent + verifier). Limits are enforced only on the **rootful** engine.
- **Runner environment.** `run_session.py` sets `PATH` (Linux binaries first), `XDG_RUNTIME_DIR`, `CONTAINER_HOST`, `DOCKER_HOST`, `PODMAN_COMPOSE_WARNING_LOGS=false`, picks `BASE_URL` by agent, strips inline `# ...` comments from `.env`, and exports `API_KEY` under `API_KEY_NAME`.
- **Spend guards.** `max_budget_usd`, `max_turns`, a dedicated spend-capped key, and a ledger with a hard daily cap and kill switch (`DAILY_SPEND_CAP_USD=1` enforced once the ledger lands in M3).

---

## Milestones

Each has an exit test.

- **M0 — Harbor + Podman spike** · *closed 19 Nov 2026.* *Exit:* full list in Progress.
- **M1 — Data and sealed split** · *closed 20 Nov 2026, hardened 23 Nov 2026.* `src/DataPipeline/` reading `config/split.yaml` (single source of truth), frozen 80 m split with proportional L/V/T (Middle visible but unscored), multi-date SAR (`260610`, `261014`, auto-discovered, `HAlpha_H/A/alpha`) + S2 NDVI, combined `L_labelled.csv` + merged `unlabelled.csv`, hardened leak checker (verified clean on `prior/` docs and GeoTIFF rasters), and resampled-split composite gate calibration. *Exit:* all checks green in Progress.
- **M2 — Verifier v1.** `fit_predict` contract (`(estate, Long, Lat)` aligned); `config/verifier.yaml`; root-owned sealed master baked into verifier image (via prebuilt local image + `FROM`, verified); per-estate and composite V scoring; T scoring + gate library with unit tests; pip-install at grading; starter pipeline (NDVI-only balanced LogReg trained on L from the M1 three-estate view, matching the POC baseline); coordinates-only baseline per estate. *Exit:* starter incumbent reproduces a fixed composite and per-estate score; `nop` neutral on all three; adversarial pipelines fail safely; runtime inside caps.
- **M3 — Builder, runner, collector.** `build_task` with the allowlist copier into `environment/` (data, all rasters, prior docs), `run_session`, `collect`, `lineage.json`, T-attempt counter per nominee, spend ledger, `events.jsonl` and `certification.json`. *Exit:* a 20-minute dry session on a cheap model flows end to end; `harbor lint` and `harbor check` pass; `002-CheckLeaks.py` passes on the generated `tasks/` and `jobs/`; re-collecting the same job does not duplicate events.
- **M4 — Dashboard.** `AgentRSI/scripts/dashboard.py` writes `AgentRSI/dashboard/index.html`, built on fixtures and the M3 dry session. *Exit:* page numbers match `reward.json`, `certification.json` and `lineage.json` exactly (scripted diff); events show context on hover/click; epoch boundaries not joined; CI bands and `+0.03` line shown; works offline; no T number in any `tasks/` dir.
- **M5 — First real session (S1, 8 h, spend-capped).** *Exit:* you review the report and dashboard; we tune the templates.
- **M6 — Iteration and schedule.** S2 generated from S1, gate works, resume is frictionless. Then Windows Task Scheduler or cron runs `daily.sh` (collect, dashboard, ledger, kill switch).
- **M7 — In-run verifier (v2).** Multi-container sidecar, MCP score tool, query budget covering all three estates, noise added to returned V scores.
- **Later:** more focus lenses, split-refresh epochs, extra estates as folds, remote sandboxes.

---

## Progress

### M0 exit checks (11–19 Nov 2026)

Tooling lives in `AgentRSI/` (gitignored `.venv/`, `.tools/`).

| Exit check | Result | Evidence |
|---|---|---|
| Pinned Harbor installs | **Pass.** `harbor==0.25.0` (Python 3.12) | `pyproject.toml`, `uv.lock` |
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
| Composite gate calibrated (resampled splits) | **Pass on synthetic arms.** `003-Calibration.py gate` over 120 **resampled** 80 m splits (never touching frozen T, 17 candidate features): 0.000 false adopts, 0.967 power at +0.04, 1.000 at +0.06; `comp_only` recorded. Non-degradation confirmed at `-0.03`. Real candidates stay open. | `.work/gate_results.json`, Observations › Gate calibration |

### Next actions

**M2 is next and has not started. It waits for your go-ahead.**

**Parked items for upcoming milestones**
- **M2:** Root-owned sealed master copy inside the verifier image (test prebuilt local image + `FROM` **[?]**); `config/verifier.yaml`; starter pipeline (NDVI-only balanced LogReg on M1 three-estate view) and coordinates-only baseline.
- **M3:** `run_session.py` picks `BASE_URL` per agent, strips inline `# ...` comments from `.env`, sets `PODMAN_COMPOSE_WARNING_LOGS=false`; allowlist copier bundles `L_labelled.csv`, `unlabelled.csv`, all `data/SAR-scenes/*.tif`, and `prior/` docs; `002-CheckLeaks.py` runs on every built task.
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
- **Reward** goes to `/logs/verifier/reward.txt` or `reward.json` (richer, wins if both exist). We use graded metrics on purpose.
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
8. **Keep install errors visible** in job records (one transient TLS error occurred during an M0 Claude Code install and succeeded on rerun; also guard memory on large downloads after an M0 WSL memory-pressure reboot).

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
| **Credential theft** (agent on open web holds SA key) | Least-privilege SA (Object Viewer + EE read), read-only mount outside build context, instruction forbids printing, `collect.py` secret scan fails collection, `.secrets/` and `.env` gitignored. |
| GEE quota or cost from unattended agent | Instruction asks for point sampling at tree coordinates; monitor quota/billing. |
| Package install runs untrusted code / PyPI drift | Unprivileged user with unreadable labels; exact pins required and `pip freeze` recorded. |
| Rootful engine containment | No privileged mode, no extra caps, no host mounts except read-only secret, `harbor lint`. |
| Cost runaway on third-party gateway | `max_budget_usd`, `max_turns`, spend-capped key, daily cap ledger + kill switch (enforced in M3). |
| Podman/Harbor instability (harbor#398) | Pinned `harbor==0.25.0`, `docker-compose` provider, Docker Desktop fallback (only `-e` changes). |
| Resource contention on 12-core / 15 GiB WSL box | One session at a time, enforced rootful limits, schedule only after M6. |

---

## Changelog

| Version | Date | Change |
|---|---|---|
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
