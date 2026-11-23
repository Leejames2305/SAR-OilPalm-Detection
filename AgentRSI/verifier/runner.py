"""Isolated grading runner: dependency install, unprivileged per-estate execution, V/T scoring."""
from __future__ import annotations

import importlib.metadata
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from verifier.baselines import coords_only_fit_predict, incumbent_fit_predict
from verifier.metrics import (
    estate_metrics,
    evaluate_gate,
    paired_block_bootstrap,
    weighted_mean,
)

FORBIDDEN_DATASET_COLS = {
    "label",
    "class",
    "id",
    "fid",
    "objectid",
    "tree_id",
    "treeid",
    "row_id",
    "block",
    "part",
}
PINNED_REQ_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9.,_-]+\])?==[^\s;#]+(\s*;.*)?$"
)
UNSAFE_REQ_FRAGMENTS = ("file:", "://", "/sealed", "..", "\\")


def _resolve_uid_gid(run_user: str) -> tuple[int | None, int | None]:
    if os.geteuid() != 0:
        return None, None
    pw = pwd.getpwnam(run_user)
    return pw.pw_uid, pw.pw_gid


def _chown_tree(path: Path, uid: int | None, gid: int | None) -> None:
    if uid is None or gid is None:
        return
    os.chown(path, uid, gid)
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            for name in dirs:
                os.chown(os.path.join(root, name), uid, gid)
            for name in files:
                os.chown(os.path.join(root, name), uid, gid)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: float,
    uid: int | None,
    gid: int | None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        user=uid,
        group=gid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise TimeoutError(f"{label} exceeded timeout of {timeout_sec:.1f}s") from exc

    res = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if res.returncode != 0:
        tail = (stderr or stdout or "").strip().splitlines()[-8:]
        detail = " | ".join(tail) if tail else f"exit code {res.returncode}"
        raise RuntimeError(f"{label} failed: {detail}")
    return res


def _current_freeze(python_bin: Path, env: dict[str, str], uid: int | None, gid: int | None) -> list[str]:
    try:
        res = _run_subprocess(
            [str(python_bin), "-m", "pip", "freeze"],
            cwd=python_bin.parent,
            env=env,
            timeout_sec=30,
            uid=uid,
            gid=gid,
            label="pip freeze",
        )
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pkgs = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        )
        return pkgs


def validate_requirements(requirements_path: Path) -> list[str]:
    """Parse and strictly validate pinned PyPI requirement lines."""
    if not requirements_path.exists():
        return []
    lines: list[str] = []
    for raw in requirements_path.read_text().splitlines():
        cleaned = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if cleaned.startswith("-") or cleaned.startswith("/") or cleaned.startswith("."):
            raise ValueError(f"Forbidden requirements.txt option or path: {cleaned!r}")
        if any(frag in lower for frag in UNSAFE_REQ_FRAGMENTS):
            raise ValueError(f"Forbidden local/URL reference in requirements.txt: {cleaned!r}")
        if not PINNED_REQ_RE.match(cleaned):
            raise ValueError(
                f"Requirements must use exact 'package==version' pins (got {cleaned!r})"
            )
        lines.append(cleaned)
    return lines


def install_submission_deps(
    requirements_path: Path,
    work_root: Path,
    run_user: str,
    timeout_sec: float,
) -> tuple[Path, list[str]]:
    """Install validated requirements into an isolated unprivileged venv and return (python_bin, freeze)."""
    uid, gid = _resolve_uid_gid(run_user)
    home_dir = work_root / "pip_home"
    cache_dir = work_root / "pip_cache"
    home_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(home_dir, 0o700)
    os.chmod(cache_dir, 0o700)
    _chown_tree(home_dir, uid, gid)
    _chown_tree(cache_dir, uid, gid)

    scrub_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home_dir),
        "TMPDIR": str(home_dir),
        "PIP_CACHE_DIR": str(cache_dir),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
    }

    reqs = validate_requirements(requirements_path)
    base_py = Path(sys.executable)
    if not reqs:
        return base_py, _current_freeze(base_py, scrub_env, uid, gid)

    req_file = work_root / "requirements.sanitized.txt"
    req_file.write_text("\n".join(reqs) + "\n")
    os.chmod(req_file, 0o644)

    venv_dir = work_root / "venv"
    venv_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(venv_dir, 0o755)
    _chown_tree(venv_dir, uid, gid)

    _run_subprocess(
        [str(base_py), "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=work_root,
        env=scrub_env,
        timeout_sec=120,
        uid=uid,
        gid=gid,
        label="venv create",
    )
    venv_py = venv_dir / "bin" / "python"
    _run_subprocess(
        [str(venv_py), "-m", "pip", "install", "--no-cache-dir", "-r", str(req_file)],
        cwd=work_root,
        env=scrub_env,
        timeout_sec=timeout_sec,
        uid=uid,
        gid=gid,
        label="pip install",
    )
    freeze = _current_freeze(venv_py, scrub_env, uid, gid)
    return venv_py, freeze


def run_estate_isolated(
    *,
    python_bin: Path,
    submission_dir: Path,
    work_root: Path,
    estate: str,
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    other_labeled_dfs: dict[str, pd.DataFrame],
    datasets_dir: Path,
    cpus: int,
    timeout_sec: float,
    run_user: str,
) -> np.ndarray:
    """Run submission fit_predict for one estate in an unprivileged isolated subprocess."""
    uid, gid = _resolve_uid_gid(run_user)
    stage_dir = work_root / f"stage_{estate}"
    inputs_dir = stage_dir / "inputs"
    other_dir = inputs_dir / "other_labeled"
    sub_copy = stage_dir / "submission"
    scratch_dir = stage_dir / "scratch"
    out_dir = stage_dir / "out"
    for d in (stage_dir, inputs_dir, other_dir, scratch_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    for p in submission_dir.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Symlink not permitted in submission/: {p.name}")
    shutil.copytree(submission_dir, sub_copy, dirs_exist_ok=True, symlinks=True)
    worker_src = Path(__file__).resolve().parent / "worker.py"
    worker_dst = inputs_dir / "worker.py"
    shutil.copyfile(worker_src, worker_dst)

    labeled_csv = inputs_dir / "labeled.csv"
    unlabeled_csv = inputs_dir / "unlabeled.csv"
    labeled_df.to_csv(labeled_csv, index=False)
    unlabeled_df.to_csv(unlabeled_csv, index=False)
    for other_name, df in other_labeled_dfs.items():
        df.to_csv(other_dir / f"{other_name}.csv", index=False)

    out_npy = out_dir / "preds.npy"

    # Inputs & copied submission are read-only for runner; scratch & out are runner-owned
    for d in (stage_dir, inputs_dir, other_dir, sub_copy):
        os.chmod(d, 0o755)
    for root, dirs, files in os.walk(sub_copy):
        os.chmod(root, 0o755)
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)
    for p in [worker_dst, labeled_csv, unlabeled_csv, *other_dir.glob("*.csv")]:
        os.chmod(p, 0o644)
    for d in (scratch_dir, out_dir):
        os.chmod(d, 0o700)
        _chown_tree(d, uid, gid)

    scrub_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(scratch_dir),
        "TMPDIR": str(scratch_dir),
        "OMP_NUM_THREADS": str(cpus),
        "OPENBLAS_NUM_THREADS": str(cpus),
        "MKL_NUM_THREADS": str(cpus),
        "NUMEXPR_NUM_THREADS": str(cpus),
        "PYTHONNOUSERSITE": "1",
    }
    cmd = [
        str(python_bin),
        str(worker_dst),
        "--submission",
        str(sub_copy),
        "--labeled",
        str(labeled_csv),
        "--unlabeled",
        str(unlabeled_csv),
        "--other-labeled-dir",
        str(other_dir),
        "--datasets-dir",
        str(datasets_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--estate",
        estate,
        "--cpus",
        str(cpus),
        "--out",
        str(out_npy),
    ]
    _run_subprocess(
        cmd,
        cwd=scratch_dir,
        env=scrub_env,
        timeout_sec=timeout_sec,
        uid=uid,
        gid=gid,
        label=f"worker[{estate}]",
    )
    arr = np.load(out_npy, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != len(unlabeled_df) or not np.isfinite(arr).all():
        raise ValueError(f"{estate}: invalid prediction output")
    return arr


def validate_datasets(
    datasets_dir: Path,
    expected_keys: set[tuple[str, float, float]],
    max_mb: float,
) -> dict:
    """Audit /app/datasets size, coordinate-key coverage, and absence of forbidden label/split columns."""
    if not datasets_dir.exists():
        return {"exists": False, "valid": True, "total_mb": 0.0, "datasets": {}, "errors": []}

    errors: list[str] = []
    total_bytes = 0
    for p in datasets_dir.rglob("*"):
        if p.is_symlink():
            errors.append(f"Symlink not permitted in datasets/: {p.name}")
            continue
        if p.is_file():
            total_bytes += p.stat().st_size

    total_mb = round(total_bytes / (1024 * 1024), 3)
    if total_mb > max_mb:
        errors.append(f"datasets/ size {total_mb:.2f} MB exceeds cap {max_mb} MB")

    ds_summary: dict[str, dict] = {}
    for sub in sorted(datasets_dir.iterdir()):
        if not sub.is_dir():
            continue
        pq = sub / "features.parquet"
        csv_p = sub / "features.csv"
        fpath = pq if pq.exists() else (csv_p if csv_p.exists() else None)
        if fpath is None:
            continue
        try:
            df = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)
        except Exception as exc:
            errors.append(f"{sub.name}/{fpath.name}: unreadable ({exc})")
            continue

        cols_lc = {c.lower() for c in df.columns}
        bad_cols = sorted(cols_lc & FORBIDDEN_DATASET_COLS)
        if bad_cols:
            errors.append(f"{sub.name}/{fpath.name}: forbidden column(s) {bad_cols}")
        missing_keys = [k for k in ("estate", "Long", "Lat") if k not in df.columns]
        if missing_keys:
            errors.append(f"{sub.name}/{fpath.name}: missing key column(s) {missing_keys}")
            continue

        keys = list(zip(df["estate"].astype(str), df["Long"].astype(float), df["Lat"].astype(float)))
        if len(set(keys)) != len(keys):
            errors.append(f"{sub.name}/{fpath.name}: duplicate (estate, Long, Lat) rows")
        if set(keys) != expected_keys:
            errors.append(
                f"{sub.name}/{fpath.name}: key coverage mismatch ({len(set(keys))} vs {len(expected_keys)})"
            )
        ds_summary[sub.name] = {
            "file": fpath.name,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "nans": int(df.isna().sum().sum()),
        }

    return {
        "exists": True,
        "valid": len(errors) == 0,
        "total_mb": total_mb,
        "datasets": ds_summary,
        "errors": errors,
    }


def _load_sealed_bundle(sealed_dir: Path, verifier_cfg: dict) -> dict:
    estates = list(verifier_cfg["estates"])
    split_cfg = yaml.safe_load((sealed_dir / "split.yaml").read_text())
    master = pd.read_csv(sealed_dir / "labels_master.csv", dtype={"Long": str, "Lat": str})
    split_df = pd.read_csv(sealed_dir / "split_assignment.csv")
    sealed = master.merge(split_df[["estate", "id", "block", "part"]], on=["estate", "id"], validate="one_to_one")
    sealed_map = {
        (est, lo, la): (part, int(blk), cls)
        for est, lo, la, part, blk, cls in zip(
            sealed["estate"],
            sealed["Long"],
            sealed["Lat"],
            sealed["part"],
            sealed["block"],
            sealed["class"],
        )
    }

    l_str = pd.read_csv(sealed_dir / "L_labelled.csv", dtype={"Long": str, "Lat": str})
    u_str = pd.read_csv(sealed_dir / "unlabelled.csv", dtype={"Long": str, "Lat": str})
    l_flt = pd.read_csv(sealed_dir / "L_labelled.csv")
    u_flt = pd.read_csv(sealed_dir / "unlabelled.csv")

    expected_keys = set(
        zip(
            sealed["estate"].astype(str),
            sealed["Long"].astype(float),
            sealed["Lat"].astype(float),
        )
    )

    by_estate: dict[str, dict] = {}
    for e in estates:
        l_mask = l_str["estate"] == e
        u_mask = u_str["estate"] == e
        l_df = l_flt.loc[l_mask].reset_index(drop=True)
        u_df = u_flt.loc[u_mask].reset_index(drop=True)
        u_keys = list(zip(u_str.loc[u_mask, "estate"], u_str.loc[u_mask, "Long"], u_str.loc[u_mask, "Lat"]))
        meta = [sealed_map[k] for k in u_keys]
        parts = np.array([m[0] for m in meta])
        blocks = np.array([m[1] for m in meta], dtype=np.int64)
        classes = np.array([m[2] for m in meta])
        by_estate[e] = {
            "labeled": l_df,
            "unlabeled": u_df,
            "parts": parts,
            "blocks": blocks,
            "classes": classes,
        }

    return {
        "estates": estates,
        "split_cfg": split_cfg,
        "by_estate": by_estate,
        "expected_keys": expected_keys,
    }


def _score_partition(
    bundle: dict,
    scores_by_estate: dict[str, np.ndarray],
    inc_by_estate: dict[str, np.ndarray],
    coords_by_estate: dict[str, np.ndarray],
    part: str,
    weights: dict[str, float],
    bootstrap_B: int,
    bootstrap_seed: int,
    ci_pct: float,
    min_pos: int,
) -> dict:
    estates = bundle["estates"]
    y_e: dict[str, np.ndarray] = {}
    blk_e: dict[str, np.ndarray] = {}
    cand_e: dict[str, np.ndarray] = {}
    inc_e: dict[str, np.ndarray] = {}
    coords_e: dict[str, np.ndarray] = {}

    cand_metrics: dict[str, dict] = {}
    inc_metrics: dict[str, dict] = {}
    coords_metrics: dict[str, dict] = {}

    for e in estates:
        data = bundle["by_estate"][e]
        mask = (data["parts"] == part) & (data["classes"] != "Middle")
        y = (data["classes"][mask] == "Unhealthy").astype(int)
        y_e[e] = y
        blk_e[e] = data["blocks"][mask]
        cand_e[e] = np.asarray(scores_by_estate[e][mask], dtype=float)
        inc_e[e] = np.asarray(inc_by_estate[e][mask], dtype=float)
        coords_e[e] = np.asarray(coords_by_estate[e][mask], dtype=float)

        cand_metrics[e] = estate_metrics(y, cand_e[e])
        inc_metrics[e] = estate_metrics(y, inc_e[e])
        coords_metrics[e] = estate_metrics(y, coords_e[e])

    comp_lift = weighted_mean({e: float(cand_metrics[e]["pr_auc_lift"]) for e in estates}, weights)
    boot_inc = paired_block_bootstrap(
        y_e, cand_e, inc_e, blk_e, weights, B=bootstrap_B, seed=bootstrap_seed, ci_pct=ci_pct, min_pos=min_pos
    )
    boot_coords = paired_block_bootstrap(
        y_e, cand_e, coords_e, blk_e, weights, B=bootstrap_B, seed=bootstrap_seed, ci_pct=ci_pct, min_pos=min_pos
    )
    return {
        "comp_lift": comp_lift,
        "cand_metrics": cand_metrics,
        "inc_metrics": inc_metrics,
        "coords_metrics": coords_metrics,
        "boot_inc": boot_inc,
        "boot_coords": boot_coords,
    }


def grade_submission(
    *,
    sealed_dir: Path,
    submission_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    verifier_cfg: dict | None = None,
    run_user: str = "runner",
    timeout_override: float | None = None,
) -> tuple[dict, dict, dict]:
    """Grade a submission on V and T, lock output_dir before running untrusted code, and write outputs."""
    t0 = time.monotonic()
    cfg = verifier_cfg or yaml.safe_load((sealed_dir / "verifier.yaml").read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(output_dir, 0, 0)
        os.chmod(output_dir, 0o700)
        if datasets_dir.exists():
            for root, dirs, files in os.walk(datasets_dir, followlinks=False):
                rp = Path(root)
                if not rp.is_symlink():
                    os.chmod(rp, 0o755)
                for name in dirs:
                    dp = rp / name
                    if not dp.is_symlink():
                        os.chmod(dp, 0o755)
                for name in files:
                    fp = rp / name
                    if not fp.is_symlink():
                        os.chmod(fp, 0o644)

    bundle = _load_sealed_bundle(sealed_dir, cfg)
    estates: list[str] = bundle["estates"]
    weights = {str(k): float(v) for k, v in cfg["reward_weights"].items()}
    gate_cfg = cfg["gate"]
    rt_cfg = cfg["runtime"]

    env_timeout = os.environ.get("AGENTRSI_ESTATE_TIMEOUT_SEC")
    estate_timeout = float(
        timeout_override
        if timeout_override is not None
        else (env_timeout if env_timeout else rt_cfg["per_estate_timeout_sec"])
    )
    pip_timeout = float(rt_cfg["pip_install_timeout_sec"])
    cpus = int(rt_cfg["cpus"])

    # Trusted reference baselines
    inc_scores: dict[str, np.ndarray] = {}
    coords_scores: dict[str, np.ndarray] = {}
    for e in estates:
        data = bundle["by_estate"][e]
        other = {o: bundle["by_estate"][o]["labeled"] for o in estates if o != e}
        ctx = {
            "estate": e,
            "datasets_dir": datasets_dir,
            "other_labeled": other,
            "scratch_dir": output_dir,
            "cpus": cpus,
        }
        inc_scores[e] = incumbent_fit_predict(data["labeled"], data["unlabeled"], ctx)
        coords_scores[e] = coords_only_fit_predict(data["labeled"], data["unlabeled"], ctx)

    datasets_audit = validate_datasets(datasets_dir, bundle["expected_keys"], float(rt_cfg["max_datasets_mb"]))

    cand_scores: dict[str, np.ndarray] = {}
    status = "ok"
    valid = True
    error_msg: str | None = None
    pip_freeze: list[str] = []

    pipeline_file = submission_dir / "pipeline.py"
    if not submission_dir.exists() or not pipeline_file.exists():
        status = "nop"
        valid = False
        error_msg = "No submission/pipeline.py found"
    elif not datasets_audit["valid"]:
        status = "failed"
        valid = False
        error_msg = "; ".join(datasets_audit["errors"])
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="agentrsi_grade_"))
        try:
            os.chmod(tmp_root, 0o755)
            py_bin, pip_freeze = install_submission_deps(
                submission_dir / "requirements.txt",
                tmp_root,
                run_user=run_user,
                timeout_sec=pip_timeout,
            )
            for e in estates:
                data = bundle["by_estate"][e]
                other = {o: bundle["by_estate"][o]["labeled"] for o in estates if o != e}
                cand_scores[e] = run_estate_isolated(
                    python_bin=py_bin,
                    submission_dir=submission_dir,
                    work_root=tmp_root,
                    estate=e,
                    labeled_df=data["labeled"],
                    unlabeled_df=data["unlabeled"],
                    other_labeled_dfs=other,
                    datasets_dir=datasets_dir,
                    cpus=cpus,
                    timeout_sec=estate_timeout,
                    run_user=run_user,
                )
        except Exception as exc:
            status = "failed"
            valid = False
            error_msg = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if not valid:
        cand_scores = {
            e: np.zeros(len(bundle["by_estate"][e]["unlabeled"]), dtype=np.float64)
            for e in estates
        }

    B = int(gate_cfg["bootstrap_B"])
    seed = int(gate_cfg["bootstrap_seed"])
    ci_pct = float(gate_cfg["bootstrap_ci_percentile"])
    min_pos = int(gate_cfg["min_bootstrap_positives"])

    v_res = _score_partition(
        bundle, cand_scores, inc_scores, coords_scores, "V", weights, B, seed, ci_pct, min_pos
    )
    t_res = _score_partition(
        bundle, cand_scores, inc_scores, coords_scores, "T", weights, B, seed + 1, ci_pct, min_pos
    )

    gate = evaluate_gate(
        comp_delta=t_res["boot_inc"]["comp_delta"],
        comp_ci_low=t_res["boot_inc"]["comp_ci_low"],
        per_estate_deltas={e: float(d["delta"]) for e, d in t_res["boot_inc"]["per_estate"].items()},
        effect_floor=float(gate_cfg["effect_floor"]),
        non_degradation_floor=float(gate_cfg["non_degradation_floor"]),
    )
    if not valid:
        gate["passed"] = False
        if error_msg and error_msg not in gate["failed_reasons"]:
            gate["failed_reasons"].insert(0, f"submission invalid ({status}): {error_msg}")

    runtime_sec = round(time.monotonic() - t0, 3)
    epoch = int(cfg["epoch"])
    split_sha = str(bundle["split_cfg"]["sha256_of_assignment"])

    # 1. Flat finite numeric dict for Harbor's _parse_reward_json()
    reward_json: dict[str, float | int] = {
        "score": float(v_res["comp_lift"]),
        "v_lift_AirHitam": float(v_res["cand_metrics"]["AirHitam"]["pr_auc_lift"]),
        "v_lift_Palong": float(v_res["cand_metrics"]["Palong"]["pr_auc_lift"]),
        "v_lift_Serting": float(v_res["cand_metrics"]["Serting"]["pr_auc_lift"]),
        "v_delta_vs_incumbent": float(v_res["boot_inc"]["comp_delta"]),
        "v_delta_vs_coords": float(v_res["boot_coords"]["comp_delta"]),
        "submission_valid": int(1 if valid else 0),
        "runtime_sec": float(runtime_sec),
    }

    # 2. V-only detailed report (strictly zero T numbers)
    v_report = {
        "epoch": epoch,
        "split_sha256": split_sha,
        "status": status,
        "submission_valid": bool(valid),
        "error": error_msg,
        "runtime_sec": float(runtime_sec),
        "composite": {
            "score": float(v_res["comp_lift"]),
            "delta_vs_incumbent": {
                "comp_delta": float(v_res["boot_inc"]["comp_delta"]),
                "ci_low": float(v_res["boot_inc"]["comp_ci_low"]),
                "ci_high": float(v_res["boot_inc"]["comp_ci_high"]),
                "n_valid_draws": int(v_res["boot_inc"]["n_valid_draws"]),
                "n_requested_draws": int(v_res["boot_inc"]["n_requested_draws"]),
            },
            "delta_vs_coords": {
                "comp_delta": float(v_res["boot_coords"]["comp_delta"]),
                "ci_low": float(v_res["boot_coords"]["comp_ci_low"]),
                "ci_high": float(v_res["boot_coords"]["comp_ci_high"]),
                "n_valid_draws": int(v_res["boot_coords"]["n_valid_draws"]),
                "n_requested_draws": int(v_res["boot_coords"]["n_requested_draws"]),
            },
        },
        "estates": {
            e: {
                **v_res["cand_metrics"][e],
                "delta_vs_incumbent": v_res["boot_inc"]["per_estate"][e],
                "delta_vs_coords": v_res["boot_coords"]["per_estate"][e],
                "incumbent_v_pr_auc": float(v_res["inc_metrics"][e]["pr_auc"]),
                "coords_v_pr_auc": float(v_res["coords_metrics"][e]["pr_auc"]),
            }
            for e in estates
        },
        "datasets_audit": datasets_audit,
        "pip_freeze": pip_freeze,
    }

    # 3. Host-only T certification (no tree IDs or coordinates)
    certification = {
        "epoch": epoch,
        "split_sha256": split_sha,
        "status": status,
        "submission_valid": bool(valid),
        "error": error_msg,
        "gate_passed": bool(gate["passed"]),
        "gate": gate,
        "composite": {
            "comp_delta": float(t_res["boot_inc"]["comp_delta"]),
            "ci_low": float(t_res["boot_inc"]["comp_ci_low"]),
            "ci_high": float(t_res["boot_inc"]["comp_ci_high"]),
            "delta_vs_coords": float(t_res["boot_coords"]["comp_delta"]),
            "n_valid_draws": int(t_res["boot_inc"]["n_valid_draws"]),
            "n_requested_draws": int(t_res["boot_inc"]["n_requested_draws"]),
        },
        "estates": {
            e: {
                "cand_metrics": t_res["cand_metrics"][e],
                "incumbent_metrics": t_res["inc_metrics"][e],
                "coords_metrics": t_res["coords_metrics"][e],
                "delta_vs_incumbent": t_res["boot_inc"]["per_estate"][e],
                "delta_vs_coords": t_res["boot_coords"]["per_estate"][e],
            }
            for e in estates
        },
    }

    (output_dir / "reward.json").write_text(json.dumps(reward_json, indent=2) + "\n")
    (output_dir / "v_report.json").write_text(json.dumps(v_report, indent=2) + "\n")
    (output_dir / "certification.json").write_text(json.dumps(certification, indent=2) + "\n")
    return reward_json, v_report, certification
