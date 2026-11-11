# AgentRSI - Plan (v1.0, final for M0)

**Date:** 11 November 2026 | **Status:** plan finalized, nothing built yet | **Paper:** `docs/Reference/...RSI4EO_2610.18845v2.pdf`
**v1.0 changes:** verifier network is **fully public** (your call: throwaway container), so no egress allowlist and no nftables sidecar are needed anywhere; gate parameters locked (effect floor 0.03, T-attempt budget 20). Earlier in v0.3: AirHitam-dominant reward weights, the T-only certification gate (section 7), rootful engine. Only M0's installs and the `.env` placeholders remain open (section 11).

Goal: turn this project into an RSI4EO-style loop. A coding agent builds the oil-palm triage model inside a sandbox, a sealed verifier scores it, and each day's session starts from what the last one found. Humans design the harness and verifier. Agents build the models.

Legend: **[V]** verified from Harbor docs or source | **[T]** tested on this machine | **[S]** from a simulation on your real AirHitam tables | **[?]** unverified, check in M0 | **[D]** decided with you.

---

## 1. RSI4EO mapped onto this project

| RSI4EO concept | Here |
|---|---|
| Harness H (data, checkpoints, libs, budget) | Per-tree SAR tables + ROI-masked ALOS-2 rasters for 3 estates, prior POC reports, incumbent pipeline, GEE/GCS access, a generous Python geo/ML stack. **CPU only.** |
| Verifier V (sealed, scores only) | Sealed held-out blocks for AirHitam (ground truth) and Palong/Serting (proxy labels, discounted). Returns scores, never labels. |
| Pipeline pi | `submission/`: `fit_predict(labeled, unlabeled, ctx) -> risk scores`, a **training recipe** (the product trains on ~1000 labels in a new field and predicts the rest). Self-contained apart from pip-installable packages (section 7). |
| Few-label task (0.2-4% labels) | Same shape: ~1000 labelled trees per estate, the rest is the "unlabeled pool". |
| Recursion inside a task | Agent-driven within the 8 h session. Pseudo-labelling on the unlabeled pool is permitted, nothing is prescribed. |
| **Not in the paper:** recursion across tasks | Our addition. Session N+1 is generated from session N's handoff and the incumbent. |
| 12 h, 4 x GB200, offline | 8 h, CPU-only, **public internet** for the agent, with GEE/GCS access. |

CPU-only is fine: earlier POCs show nonlinear and deep models lose to linear ones at n=160 positives (MainChecklist B). Ideas and verification are the bottleneck, not compute.

## 2. Decisions locked

- **[D] Reward (what the agent optimises)** = `(1.0 * lift_AirHitam + 0.75 * mean(lift_Palong, lift_Serting)) / 1.75`, where `lift = pr_auc - prevalence` on sealed V blocks. AirHitam is ~57% of the total, the two proxy estates ~43% together. Weights live in `verifier.yaml`. The agent sees the per-estate breakdown.
- **[D] Promotion (what moves the incumbent) is AirHitam-only**, through a **T-only certification gate** (section 7). Proxy estates never vote. They are an advisory sanity check only.
- **[D] Phased verifier:** v1 grades once at end of session; v2 adds an in-run sealed query tool (MCP sidecar) with a query budget.
- **[D] Agents:** Claude Code and Codex, both via third-party API (provider supports Anthropic Messages and OpenAI Responses). Agent, model, base URL and key name are run-time parameters with placeholders in `.env`.
- **[D] Budget / network:** 8 h agent timeout (terminal-bench-science's `timeout_sec = 28800` **[V]**). Agent phase `network_mode = "public"`.
- **[D] Verifier network: fully public** (Harbor's default). The verifier container is throwaway, so we accept open egress there. The real label protection is the unprivileged subprocess plus root-only label files (section 7), not the network. Consequences are listed in section 7 ("Network at grading").
- **[D] Gate parameters locked:** effect floor `d >= 0.03`, T-attempt budget 20 (section 7). Lowering the floor to 0.02 would roughly double false adoptions, so revisit only after real sessions show the floor is too strict.
- **[D] No package list to maintain, no approval flow.** The agent ships a pinned `requirements.txt`; the verifier installs it from PyPI.
- **[D] Runtime:** Podman from WSL, **rootful** (confirmed), Docker Desktop as fallback.
- **[D] Focus:** open-ended research is the default and **explicitly includes data expansion**. Optional lenses (`literature`, `data-expansion`, `ablation`) just bias the instruction.
- **[D] Data egress accepted:** ALOS-2 scenes are ROI-masked and the full images live elsewhere. What leaves the machine to your inference provider: ROI-masked ALOS-2, tree coordinates and labels.
- **[D] GCS/GEE service-account key is passed to the agent** (delivery and hygiene in sections 6 and 9).
- **[D]** Generated tasks live in `AgentRSI/tasks/<id>/`. Harbor's file is `instruction.md` (singular) **[V]**.

## 3. Harbor + Podman: what is verified

### Harbor design facts **[V]** (docs.harborframework.com and Harbor source)

- A task is a directory: `instruction.md`, `task.toml`, `environment/`, `tests/` (`test.sh`), optional `solution/`.
- **Separate verifier** (default for new tasks): grading runs in a fresh sandbox built from `tests/Dockerfile`. Data baked there **never reaches the agent container**. Only declared `artifacts` cross over (at original paths, sanitized: no setuid, no outward symlinks, no special files). Separate verifiers do **not** inherit runtime mounts or `extra_docker_compose` overlays, so the service-account key cannot leak into grading.
- Reward goes to `/logs/verifier/reward.txt` or `reward.json` (richer; wins if both). terminal-bench-science mandates 0/1; **we deviate on purpose** (graded metrics).
- Agent timeout **still triggers grading**, and the reward counts. Whether artifacts are collected after a timeout is undocumented **[?]** (M0 tests it).
- **Claude Code:** custom gateway via `ANTHROPIC_BASE_URL` + bearer `ANTHROPIC_AUTH_TOKEN`; options `max_turns`, `max_budget_usd`, `reasoning_effort`, `append_system_prompt`, `disable_web_search`; defaults to `--permission-mode bypassPermissions`; Anthropic Messages only.
- **Codex:** `OPENAI_BASE_URL` + `OPENAI_API_KEY`; `reasoning_effort`; **Responses API only**.
- Network modes per phase: `public` | `no-network` | `allowlist` (hostnames, wildcards, IPs, CIDRs). Docker and Podman support all three and per-phase overrides, but restricted modes need **nftables support for Harbor's egress-control sidecar**. We use `public` for every phase, so we should not need the sidecar at all (M0 confirms Harbor does not build it anyway **[?]**).
- Flags: `--base-url`, `--api-key-name`, `--api-format`, `--sm`, `--ae`/`--ak`, `-e`, `--override-cpus`/`--override-memory-mb`, `--dry-run`, `--print-config`. `harbor lint` / `harbor check` review a task. `regrade` re-runs a newer verifier on recorded outputs (separate verifier only).
- `-e podman` exists upstream (merged Sep 2026) but **harbor#398 is open**: with `podman-compose` only, Harbor mis-detects the backend and fails on `compose ls`. A working setup in that thread had `docker-compose` as the compose provider.

### Tested on your machine **[T]** (11 Nov 2026)

| Check | Result |
|---|---|
| Engine reachable from WSL | **Yes.** Podman **6.1.4**, API 1.44, crun, netavark, overlay, cgroups v2. Rootless socket `podman-user.sock` and rootful `podman-root.sock` (you can use it via group `nogroup`). |
| Container run + internet | **Yes.** An alpine container started on the rootful engine, saw 12 CPUs, fetched `pypi.org`. |
| Resources | Engine sees **15.2 GiB RAM, 12 CPUs**, shared with WSL and Windows (~11.5 GiB free at test time). The "2 GiB / 6 CPU" in `podman machine list` is nominal. |
| Kernel firewall | `CONFIG_NF_TABLES=y` is built into the WSL2 kernel. Not needed now that every phase is `public`, but it keeps a restricted verifier possible later. |
| Linux `podman` / `podman-compose` / `harbor` / `claude` / `codex` in WSL | **None installed.** Only `podman.exe` (Windows interop) and Docker Desktop's Windows `docker` / `docker-compose` shims. |

Findings that shaped the design:

1. **Rootless has no cgroup controllers delegated**, so it cannot enforce CPU or memory limits. **Rootful** has `cpuset, cpu, io, memory, pids`. Hence rootful (decided). Mitigations for weaker containment: no `privileged`, no extra capabilities, no host mounts except one read-only secret, `harbor lint`.
2. **The engine runs in a different WSL distro.** Bind-mounting `/home/lee/...` fails, but the shared `/mnt/wsl` tmpfs **is** visible. Data and code reach the sandbox through the build context; secrets go through `/mnt/wsl/...` or env injection.

Note: the Bash sandbox inside Claude sessions blocks Unix sockets and Windows interop, so engine tests had to run outside it. That is a property of this session, not of your setup. A leftover `sharp_poincare` hello container (23 h old) is not mine and was left alone.

## 4. Proposed layout

```
AgentRSI/
  README.md                  how to run a session / how to customise
  pyproject.toml             uv project; pins harbor==<version chosen in M0>
  .env.example               AGENT, MODEL, BASE_URL, API_KEY_NAME, API_FORMAT, ENV=podman,
                             GCP_PROJECT, GCP_SA_KEY_PATH, DAILY_SPEND_CAP_USD ...  (placeholders)
  .secrets/                  GITIGNORED. gcp-sa.json lives here on the WSL disk.
  config/
    session.yaml             focus lens, agent, model, budget, cpus/mem, effort, max_budget_usd
    split.yaml               FROZEN: seeds, block size, per-estate L/V/T sizes (change = new epoch)
    verifier.yaml            reward weights, gate thresholds, T-attempt budget, bootstrap B,
                             runtime caps, verifier network_mode (public)
  templates/                 instruction (base + focus lenses), task.toml, Dockerfile.agent/.verifier
  src/agentrsi/              builder library (build_task, build_data, stage_secrets, render_handoff, collect, lineage)
  verifier/                  scoring + gate library baked into every tests/ image (unit-tested)
  starter/                   session-0 pipeline (NDVI-LogReg incumbent from the POCs)
  scripts/                   thin CLIs: build_task  run_session  collect  regrade  daily
  state/lineage.json         incumbent, scores, epochs, spend ledger, T-attempt counter (committed, small)
  sessions/sNNN/             COLLECTED: REPORT.md HANDOFF.md submission/ datasets/ reward.json meta.json
  tasks/  jobs/              GENERATED (gitignored)
```

The builder uses an **allowlist copy**. Nothing reaches `environment/` unless a rule names it, because `data/Labels/*.csv`, the SHPs and the KMLs contain *all* labels.

## 5. Session lifecycle

```
state/lineage.json + sessions/s(N-1)/HANDOFF.md
        |  build_task.py   (templates + split + prior knowledge)
        v
tasks/sNNN/  --run_session.py-->  stage secrets -> harbor run -p ... -a <agent> -m <model> --base-url ... -e podman
        |                          agent (8 h, public internet, GEE) writes /app/{submission,report,datasets}
        |                          agent sandbox ends -> artifacts copied out -> separate (fresh, public-network) verifier grades
        v
jobs/sNNN/  --collect.py-->  secret scan -> sessions/sNNN/  +  lineage.json (certify T -> promote or not)
        v   next day: build_task.py reads it all again
```

Rules:
- `collect.py` runs on the host and **only copies, scans and parses**. It never imports or executes agent-written code on the host.
- **Handoff numbers come from the verifier, not the agent.** `render_handoff` injects `reward.json` values into the next instruction and flags any figure in the agent's HANDOFF that disagrees.

## 6. Generated task anatomy (sketch; validate with `harbor lint` / `--dry-run` [?])

```toml
schema_version = "1.4"
artifacts = ["/app/submission", "/app/report", "/app/datasets"]

[task]            name = "agentrsi/s003-open"
[agent]           timeout_sec = 28800.0            network_mode = "public"
[environment]     network_mode = "public"          cpus = 6   memory_mb = 6144   storage_mb = 20480
[verifier]        environment_mode = "separate"    timeout_sec = 3600
[verifier.environment]
  cpus = 4   memory_mb = 4096
  network_mode = "public"
```

Sizing: engine total 15.2 GiB, ~11.5 free; 6 + 4 GiB. Limits are enforced only on the **rootful** engine. One session at a time on this box.

**Agent-visible `environment/`:**
- `data/`: per estate, the **L** labelled trees; **V and T trees without labels** (coordinates and features, since in deployment you know where every tree is). Palong/Serting L labels are marked `proxy`.
- `rasters/`: the ROI-masked ALOS-2 scenes.
- `prior/`: `docs/POCReport/*`, MainChecklist (settled facts and dead ends), `GEECatalogSurvey.md`, the last `HANDOFF.md`, incumbent `submission/`, earlier derived datasets (size-capped).
- Base image: Python geo/ML stack (numpy/scipy/pandas/sklearn/xgboost/lightgbm, torch CPU, rasterio/geopandas, `earthengine-api`, `google-cloud-storage`), plus the helper CLI `agentrsi`: `cv` (reusable spatial-block CV on L, like the paper's local validation), `selfcheck` (runs the verifier contract offline on L only), `gcp-init`.

**Google credentials.** The agent may use GEE and GCS. Delivery avoids baking the key into any image layer:
- **Preferred:** `stage_secrets.py` copies `.secrets/gcp-sa.json` to a mode-0600 file under `/mnt/wsl/agentrsi/` (tmpfs the engine can see; wiped on WSL restart, restaged each run). The task mounts it **read-only** via `environment.mounts` **[V]** at `/run/secrets/gcp-sa.json`, with `GOOGLE_APPLICATION_CREDENTIALS` pointing there.
- **Fallback:** inject as an agent-phase env var (`--ae`), scoped to the agent phase only **[V]**. M0 must check that neither route writes the key into `jobs/*/config.json`, logs or trajectories **[?]**.
- Only the placeholders `GCP_PROJECT` and `GCP_SA_KEY_PATH` live in `.env`. The instruction gives the key's location, says to initialise `ee` with it, and **never to print or copy it**.

**`instruction.md` sections:** Mission | Harness map | Pipeline contract | Deliverables with absolute paths | Freedom (internet, papers, GEE/GCS, pseudo-labelling, data expansion all welcome) | Constraints (CPU only, 28800 s, never touch `/tests`, never try to recover sealed labels, never print the credentials) | **Checkpoint rule** | **Nominee** (name the one pipeline to certify) | Handoff from the previous session (auto-injected). The paper's "do not use online solutions" line is dropped on purpose; recovering the *sealed labels* stays forbidden.

**Checkpoint rule:** write or refresh `HANDOFF.md` at least hourly. A hard timeout can kill the agent mid-write.

`HANDOFF.md` schema (the builder validates it): State of play | What I tried (idea / result / verdict / evidence path) | Dead ends | Open leads, ranked | How to resume (commands) | Artifact index. `REPORT.md` is the long report with literature notes.

## 7. Verifier v1, certification gate and submission contract

### Splits

Frozen in `split.yaml`, stratified so each part keeps enough positives.
- **AirHitam:** L ~1000 (~64 pos, matches "saturates by ~1000 labels"), V ~750 (~48 pos), **T ~760 (~48 pos, never reported to the agent)**.
- **Palong, Serting:** L ~1000 each, V = the remainder (~108 / ~62 proxy positives). No T.
- **Block size is open [S].** At 220 m AirHitam has only **15 blocks**, so T would hold ~4-5 blocks and a block bootstrap over 4-5 clusters is unreliable (see calibration: nominal 5% false-adopt came out at 7-16%). M1 re-runs the gate calibration on finer blocks (110 m gives 42) and picks the coarsest block size whose null false-adopt rate is close to nominal. The 110 m run so far covers only the V-selection simulation, not the gate.

### Contract

Per estate the verifier calls `fit_predict(labeled=L, unlabeled=V(+T), ctx)`. `ctx` holds the estate name, the agent's `datasets/` path, the other estates' L data, a scratch dir and the CPU count. It runs in an **unprivileged subprocess that cannot read the labels** (label files root-owned, mode 0700); a separate trusted process scores afterwards. The verifier also runs the **incumbent** and a **coordinates-only** baseline in the same run, so every result is a **paired delta**.

### Packages (replaces the wheelhouse and approval flow)

- The agent ships a pinned `submission/requirements.txt`. Its own sandbox is fully open, so it can install anything while experimenting.
- The verifier runs `pip install` into an isolated venv **as the unprivileged user, before any label is readable by that user**. Same base image and Python in both sandboxes.
- Model weights and datasets should still be shipped **inside `submission/` and `datasets/`**. This is now an instruction rule, not something the network enforces (see below).
- Residual risks: pip can execute sdist build code (mitigated by the unprivileged user and unreadable labels); PyPI can drift or yank versions (mitigated by exact pins, and `pip freeze` is recorded in `reward.json` so a rerun is comparable). Nothing here needs per-run maintenance.

### Network at grading (fully public)

What the open network means, so we are choosing it with eyes open:
- **Label protection no longer has a network layer.** The labels sit in the same container as the agent's pipeline, protected by the unprivileged user and root-only (0700) label files. A pipeline could only send labels out after breaking that isolation (privilege escalation or a container escape, which matters more on a rootful engine). You accepted this because the container is throwaway. Nothing sensitive besides V/T labels is in the verifier: the service-account key is not mounted there.
- **Self-containment is no longer enforced.** A pipeline could download weights, data or an API result at grading time. It would then score well but be unreproducible, and the product (a new field, possibly with no internet) could silently depend on it. Mitigation: the instruction requires shipping everything, the agent's `selfcheck` runs the contract in the agent sandbox, and `collect.py` greps the pipeline for URLs and network calls and lists them in the session report as a flag for you to read. This is a smell test, not a guarantee.
- **If it ever matters:** flip `verifier.environment.network_mode` to `allowlist` or `no-network` (one line, needs the nftables sidecar on this host). A nominee about to be promoted could be re-graded that way via `regrade` as an extra check.

### Data expansion

New covariates ship as `datasets/<name>/features.parquet` keyed by `(estate, tree_id)` for **all** trees (L, V and T), plus `MANIFEST.md` (source, date, resolution, licence) and the extraction script. The agent has no V/T labels, so features at V/T coordinates cannot leak labels. The script exists for the product: a new field needs the same features pulled for its own coordinates. The verifier checks id coverage, schema and a size cap.

### Roles of V and T

- **V:** its score is returned to the agent (session-end `reward.json` in v1, in-run queries in v2). It is the agent's search signal and **is expected to become optimistic**.
- **T:** never reported to the agent, **spent only on certification**: one evaluation per session, for the single declared nominee. Every certification is one **T attempt** in the ledger.

### Why a T-only gate **[S]**

Simulation on your real AirHitam tables (`misc/temp/sim_winners_curse.py`, `misc/temp/sim_certification_gate.py`; scratch and gitignored). "Agent ideas" were LogReg on random subsets of the 26 NDVI/spectral/SAR features, split at 220 m blocks into ~40/30/30 L/V/T (~47 positives in V and in T), compared with the NDVI-LogReg incumbent. This is a rough stand-in for agent behaviour, not a model of it.

- A typical idea is slightly **worse** than the incumbent (mean dPR-AUC -0.005, matching the POCs). The spread between candidates (SD ~0.026) is as large as any real gain.
- **Selecting on V inflates V.** Best-of-k apparent gain on V versus the same candidate on fresh T: k=1: -0.008 / -0.006; k=3: +0.009 / 0.000; k=10: +0.023 / +0.002; k=30: +0.033 / +0.005; k=60: +0.038 / +0.007. 110 m blocks gave the same pattern.
- **In v1 the agent touches V once per session**, so selection pressure on V is roughly the number of sessions, adaptively through handoffs. A few days is mild (best-of-3 optimism ~+0.01, best-of-10 ~+0.02). **In v2 (in-run queries) k reaches 30-60 within one session (+0.03 to +0.04).** The gate below is cheap, so adopt it from day one.
- The v0.2 gate (V CI-separated from zero and T non-degrading) passed a null idea **21-25%** of the time once k was 10-30.

Gate calibration (T-only, one nominee, paired block bootstrap, 220 m blocks, 120 repetitions):

| Gate | Null idea, V-winner of k=10 | Null, k=30 | Real +0.02 (ceiling) | Real +0.04 (ceiling) | Real +0.06 (ceiling) |
|---|---|---|---|---|---|
| CI > 0 | 15.8% | 14.2% | 100% | 100% | 99% |
| CI > 0 and d >= 0.02 | 7.5% | 7.5% | 51% | 100% | 99% |
| **CI > 0 and d >= 0.03 (chosen)** | **5.8%** | **4.2%** | **12%** | **77%** | **99%** |
| CI > 0 and d >= 0.04 | 1.7% | 2.5% | 0% | 50% | 93% |

The "real gain" columns are a **ceiling**: the synthetic candidates carry no candidate-specific noise, while real candidates differ from the incumbent with SD ~0.026. Expect real power to be noticeably lower than shown.

### Promotion gate (host-side, in `collect.py`)

Promote the nominee only if on **AirHitam T**: paired block-bootstrap 5th percentile of dPR-AUC vs the incumbent is **> 0 and dPR-AUC >= 0.03**. Thresholds live in `verifier.yaml`. Advisory flag, not a vote: if the nominee is significantly worse than the incumbent on Palong and Serting (untested **[?]**), `collect.py` marks the session `REVIEW` and you decide.

### What this gate cannot do

- It reliably certifies only gains of roughly **+0.04 PR-AUC or more**. A real +0.012 improvement like NDVI + red-edge (0.133 to 0.145) would almost never pass on its own. The honest framing: the verifier detects big wins, not increments. Small plausible gains accumulate in `REPORT.md` and are **certified in batches** (stack them, nominate the combination).
- A false adopt costs little when it happens: an adopted null idea has a true effect around -0.005 to +0.005. The real cost of reuse is a T score that is optimistic for the incumbent and a spent T.
- At ~5% per attempt, 20 attempts give about a 64% chance of at least one false adopt. Hence a **T-attempt budget (default 20)**. When exhausted, or when new labels arrive, the split is refreshed as a new epoch. Incumbent T scores are never headline numbers; only paired deltas are.

### `reward.json`

`score` (weighted composite), per-estate `pr_auc_lift`, `pr_auc`, `prevalence`, `recall_at_25`, `enrichment_top10`, `roc_auc`, V CI bounds, paired delta vs incumbent, delta vs coordinates-only per estate, runtime, `pip freeze`. AirHitam **T** metrics go to a separate file that is never injected into any instruction.

### Proxy-label caveat

Palong/Serting labels are model-derived and spatially smooth (POC lifts 0.15-0.24 vs 0.06 for AirHitam). A coordinates-only model already scores well there, so an agent can earn reward from spatial smoothing. The 0.75 weight discounts it, promotion ignores it, and the coordinates-only baseline is reported per estate so inflated proxy scores are visible.

### Checks before trusting it

Oracle (starter) reproduces a known score exactly; `nop` scores neutral; adversarial pipelines (read `/tests`, write the reward, import the labels, run forever, malicious package build code) all fail safely, and a pipeline that fetches data from the internet at grading time is flagged; gate calibration re-run on the real verifier and chosen block size.

## 8. Iteration and resume

- State is **files, not conversation**. Each session starts fresh with curated context: ledger, HANDOFF, incumbent code, dead-end list. No trajectory replay, because it restores no sandbox files and bloats context **[V]**.
- Dead ends from `MainChecklist.md` section B go into every `prior/`.
- Several agents or models can attempt from the same parent. They are branches in `lineage.json`, and the gate picks. **Each branch's certification spends a T attempt**, so branching speeds up T consumption.
- A verifier or split change creates a new **epoch**. The incumbent is re-scored with `regrade` or a rerun; scores are never compared across epochs.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Label leakage into the agent image | Allowlist copy. Automated canary test: no sealed id has a label anywhere in the image or build context. Runs in M1 and on every build. |
| Agent code reads sealed labels at grading time | Separate sandbox, unprivileged subprocess, root-only label files, scoring in a trusted process. **The network is open (decided), so a pipeline that escaped this isolation could send labels out.** Accepted for a throwaway container. Alternative: restrict the verifier network (section 7). |
| Pipeline depends on the internet at grading time (unreproducible, not self-contained) | Instruction rule to ship all weights and data; URL/network-call grep in `collect.py` flags it in the report; optional offline re-grade of a nominee before promotion. |
| **Holdout overuse / false adoption over weeks** | T-only certification of one nominee, effect-size floor, T-attempt budget, epoch refresh, paired deltas only (section 7). |
| Gate cannot see small gains | Stated openly; batch certification; the report ledger preserves them. |
| Block bootstrap unreliable with few blocks | Block size chosen by calibration in M1 (section 7). |
| Proxy labels steering the search | 0.75 weight on their mean, no vote in promotion, coordinates-only baseline reported per estate. |
| Spatial prior mistaken for model skill | Same baseline (MainChecklist section E gotcha). |
| **Credential theft:** the agent browses the open web while holding a service-account key | Least-privilege SA (Object Viewer plus Earth Engine read only), key in a read-only mount not env or image, instruction forbids printing it, rotate on a schedule and after any anomaly. |
| Key leaks into artifacts, logs or git | Read-only mount outside the build context, `collect.py` secret scan (`private_key`, `BEGIN PRIVATE KEY`, key id) that **fails collection**, `jobs/`, `tasks/`, `.secrets/`, `.env` gitignored. M0 confirms Harbor never records the value. |
| GEE quota or cost from an unattended agent | Instruction asks for point sampling at tree coordinates rather than full-scene exports; check quota and billing; the ledger logs usage. |
| Package install runs untrusted code in the verifier | Unprivileged user, labels unreadable by it. |
| PyPI drift makes a rerun differ | Exact pins required; `pip freeze` recorded. |
| Rootful engine weakens containment | No privileged mode or extra capabilities, no host mounts but the secret, `harbor lint`. |
| Cost runaway on a third-party provider | `max_budget_usd`, `max_turns`, spend ledger with a hard daily cap and kill switch, a dedicated spend-capped API key. |
| Provider key exposure | Gitignored `.env`, passed with `--api-key-name`/`--ae`, never in `task.toml` or the task dir. |
| Podman/Harbor instability (open issue harbor#398) | M0 spike, pinned Harbor version, Docker Desktop fallback (only the `-e` flag changes). |
| 8 h/day on a 12-core, 15 GiB WSL box shared with Windows | One session at a time, enforced limits, schedule only after M5; remote sandboxes remain a Harbor option. |

## 10. Milestones (each has an exit test)

- **M0 Harbor + Podman spike** (blocked on your go-ahead for installs). Install pinned `harbor` (via `uv tool`), a Linux `podman` remote client and a Linux `docker-compose` (PATH must put them **before** Docker Desktop's Windows shims). Point `DOCKER_HOST` and the default connection at the rootful socket. *Exit:* oracle hello-world passes on `-e podman`; a Claude Code trial runs through your gateway; a Codex trial runs; a 2-minute-timeout trial shows **artifacts are collected and the verifier still runs**; a trial with `network_mode = "public"` in all phases runs **without** Harbor building the egress sidecar; the secret route works and **no key value appears** in `jobs/`; resource limits actually bite on rootful; a `regrade` after editing a `tests/` file works.
- **M1 Data + sealed split.** `build_data.py`, frozen `split.yaml`, allowlist copier, **gate calibration at candidate block sizes**. *Exit:* canary leak test green; stable split hash; block size chosen.
- **M2 Verifier v1.** `fit_predict` contract, scoring and gate library with unit tests, pip-install-at-grading, starter pipeline (NDVI-LogReg from `misc/POC/RVI_NDVI/Results/airhitam_ndvi.csv` plus SAR). *Exit:* oracle reproduces a fixed score; `nop` neutral; adversarial pipelines fail safely; runtime inside caps.
- **M3 Builder + runner + collector.** `build_task` / `run_session` / `collect` plus `lineage.json` and the T-attempt counter. *Exit:* a 20-minute dry session on a cheap model flows end to end; `harbor lint` and `harbor check` pass.
- **M4 First real session (S1, 8 h, spend-capped).** *Exit:* you review the report; we tune templates.
- **M5 Iteration + schedule.** S2 generated from S1; the gate works; resume is frictionless. Then Windows Task Scheduler or cron triggers `daily.sh`, with the spend ledger and kill switch.
- **M6 (v2) In-run verifier.** Multi-container sidecar, MCP score tool, query budget, noise added to returned V scores. The gate is already in place.
- **Later:** more focus lenses, split refresh epochs, extra estates as folds, remote sandboxes.

## 11. Open items (the design questions are all closed)

Closed on 11 Nov 2026: reward weights, rootful engine, package policy, verifier network (fully public), gate floor 0.03 and T budget 20.

Still open, neither blocks planning:
1. **M0 installs** (`harbor` via `uv tool`, a Linux `podman` client and a Linux `docker-compose` into `~/.local/bin`): waiting on your go-ahead. They download third-party binaries, so I will not run them before you say so.
2. **Placeholders** to fill later in `AgentRSI/.env`: provider base URL(s) and model names, `GCP_PROJECT`, and the service-account JSON path. Real values go only in that gitignored file, never in chat.

Things the plan expects to revisit with real data, not now: the gate floor (if it proves too strict after a few sessions), the block size (M1 calibration), and whether to restrict the verifier network again (section 7).
