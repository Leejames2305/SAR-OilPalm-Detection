"""In-Run Verifier v2 sidecar MCP server (V-only, noisy feedback, budgeted, isolated)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd
import yaml

from verifier.baselines import coords_only_fit_predict, incumbent_fit_predict
from verifier.metrics import estate_metrics, weighted_mean
from verifier.runner import (
    install_submission_deps,
    run_estate_isolated,
    validate_datasets,
)

MAX_INLINE_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
DEFAULT_MAX_QUERIES = 15
DEFAULT_NOISE_SIGMA = 0.005
DEFAULT_PORT = 8765
DEFAULT_PER_ESTATE_TIMEOUT_SEC = 180.0
DEFAULT_RUN_USER = "runner"
DEFAULT_TOKEN_ENV = "AGENTRSI_VERIFIER_MCP_TOKEN"


def _resolve_candidate_base_python() -> Path:
    """Resolve the base Python binary used to run candidate submissions.

    Prefers the untouched container Python (`/usr/local/bin/python`) so candidate
    code and `pip_freeze` never inherit `sidecar-venv` packages (`mcp`, `uvicorn`).
    Falls back to `sys.base_prefix` when running inside a `sidecar-venv` without
    `/usr/local/bin/python`, and to `sys.executable` for host unit tests.
    """
    container_py = Path("/usr/local/bin/python")
    if container_py.exists():
        return container_py
    if Path(sys.prefix).name == "sidecar-venv":
        for cand in (
            Path(sys.base_prefix) / "bin" / "python3",
            Path(sys.base_prefix) / "bin" / "python",
        ):
            if cand.exists():
                return cand
    return Path(sys.executable)


def _install_submission_deps_isolated(
    requirements_path: Path,
    work_root: Path,
    *,
    run_user: str,
    timeout_sec: float,
) -> tuple[Path, list[str]]:
    """Invoke `runner.install_submission_deps` using the candidate base Python."""
    base_py = _resolve_candidate_base_python()
    if base_py == Path(sys.executable):
        return install_submission_deps(requirements_path, work_root, run_user, timeout_sec)

    agent_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{agent_root}:{existing_pp}" if existing_pp else agent_root

    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from verifier.runner import install_submission_deps\n"
        "try:\n"
        "    py_bin, freeze = install_submission_deps(\n"
        "        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])\n"
        "    )\n"
        "    print(json.dumps({'ok': True, 'python_bin': str(py_bin), 'freeze': freeze}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok': False, 'exc_type': type(exc).__name__, 'msg': str(exc)}))\n"
        "    sys.exit(2)\n"
    )
    proc = subprocess.run(
        [
            str(base_py),
            "-c",
            script,
            str(requirements_path),
            str(work_root),
            run_user,
            str(timeout_sec),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec + 150.0,
        check=False,
    )
    last_line = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else ""
    if last_line:
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if payload.get("ok"):
                return Path(payload["python_bin"]), list(payload["freeze"])
            exc_type = payload.get("exc_type", "RuntimeError")
            msg = payload.get("msg", "dependency install failed")
            if exc_type == "ValueError":
                raise ValueError(msg)
            if exc_type == "TimeoutError":
                raise TimeoutError(msg)
            raise RuntimeError(msg)
    detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
    raise RuntimeError(f"isolated dependency install failed: {detail}")


def _load_v_only_bundle(sealed_dir: Path, verifier_cfg: dict[str, Any]) -> dict[str, Any]:
    """Load sealed tables and immediately strip all non-V (T) labels from memory."""
    estates = [str(e) for e in verifier_cfg["estates"]]
    master = pd.read_csv(sealed_dir / "labels_master.csv", dtype={"Long": str, "Lat": str})
    split_df = pd.read_csv(sealed_dir / "split_assignment.csv")
    sealed = master.merge(
        split_df[["estate", "id", "block", "part"]],
        on=["estate", "id"],
        validate="one_to_one",
    )

    expected_keys = set(
        zip(
            sealed["estate"].astype(str),
            sealed["Long"].astype(float),
            sealed["Lat"].astype(float),
        )
    )

    # Strip all T/L labels immediately; never retain T classes or block IDs in memory.
    v_only_map: dict[tuple[str, str, str], tuple[str, str]] = {
        (est, lo, la): ("V", cls) if part == "V" else ("NON_V", "UNSCORED")
        for est, lo, la, part, cls in zip(
            sealed["estate"],
            sealed["Long"],
            sealed["Lat"],
            sealed["part"],
            sealed["class"],
        )
    }
    del master, split_df, sealed

    l_str = pd.read_csv(sealed_dir / "L_labelled.csv", dtype={"Long": str, "Lat": str})
    u_str = pd.read_csv(sealed_dir / "unlabelled.csv", dtype={"Long": str, "Lat": str})
    l_flt = pd.read_csv(sealed_dir / "L_labelled.csv")
    u_flt = pd.read_csv(sealed_dir / "unlabelled.csv")

    by_estate: dict[str, dict[str, Any]] = {}
    for e in estates:
        l_mask = l_str["estate"] == e
        u_mask = u_str["estate"] == e
        l_df = l_flt.loc[l_mask].reset_index(drop=True)
        u_df = u_flt.loc[u_mask].reset_index(drop=True)
        u_keys = list(zip(u_str.loc[u_mask, "estate"], u_str.loc[u_mask, "Long"], u_str.loc[u_mask, "Lat"]))
        v_mask = np.array(
            [v_only_map[k][0] == "V" and v_only_map[k][1] != "Middle" for k in u_keys],
            dtype=bool,
        )
        v_y = np.array(
            [1 if v_only_map[k][1] == "Unhealthy" else 0 for k, keep in zip(u_keys, v_mask) if keep],
            dtype=int,
        )
        by_estate[e] = {
            "labeled": l_df,
            "unlabeled": u_df,
            "v_mask": v_mask,
            "v_y": v_y,
        }

    return {
        "estates": estates,
        "by_estate": by_estate,
        "expected_keys": expected_keys,
    }


def _resolve_jailed_dir(shared_dir: Path, user_path: str, label: str) -> Path:
    """Resolve a path strictly inside `shared_dir`, rejecting traversal and symlinks."""
    if not isinstance(user_path, str) or not user_path.strip() or "\x00" in user_path:
        raise ValueError(f"{label}: empty or invalid path")
    if not shared_dir.exists() or not shared_dir.is_dir() or shared_dir.is_symlink():
        raise ValueError(f"{label}: shared_dir is unavailable")

    raw = Path(user_path)
    if ".." in raw.parts:
        raise ValueError(f"{label}: path traversal '..' is not permitted")

    base_resolved = shared_dir.resolve()
    if raw.is_absolute():
        rel: Path | None = None
        for anchor in (shared_dir, base_resolved):
            try:
                rel = raw.relative_to(anchor)
                break
            except ValueError:
                continue
        if rel is None:
            raise ValueError(f"{label}: path escapes shared_dir")
    else:
        rel = raw

    cur = shared_dir
    for part in rel.parts:
        if part in ("", "."):
            continue
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"{label}: symlink not permitted in path: {part}")

    candidate = cur
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"{label}: directory does not exist: {user_path}")
    if candidate.is_symlink():
        raise ValueError(f"{label}: symlink not permitted: {user_path}")

    resolved = candidate.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"{label}: resolved path escapes shared_dir")

    for root, dirs, files in os.walk(candidate, followlinks=False):
        rp = Path(root)
        if rp.is_symlink():
            raise ValueError(f"{label}: symlink not permitted inside directory: {rp.name}")
        for dname in dirs:
            dp = rp / dname
            if dp.is_symlink():
                raise ValueError(f"{label}: symlink not permitted inside directory: {dname}")
        for fname in files:
            fp = rp / fname
            if fp.is_symlink():
                raise ValueError(f"{label}: symlink not permitted inside directory: {fname}")

    return candidate


def _hash_file_streaming(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_content_sha256(submission_dir: Path, datasets_dir: Path | None) -> str:
    """Compute a deterministic SHA-256 over submission files and optional dataset files."""
    h = hashlib.sha256()

    sub_files: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(submission_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        rp = Path(root)
        for fname in sorted(files):
            if fname.endswith((".pyc", ".pyo")):
                continue
            fp = rp / fname
            if fp.is_file() and not fp.is_symlink():
                sub_files.append((fp.relative_to(submission_dir).as_posix(), fp))
    sub_files.sort(key=lambda item: item[0])

    h.update(f"sub_count={len(sub_files)}\n".encode("utf-8"))
    for rel_posix, fp in sub_files:
        digest = _hash_file_streaming(fp)
        h.update(f"sub|{rel_posix}|{fp.stat().st_size}|{digest}\n".encode("utf-8"))

    if datasets_dir is not None and datasets_dir.exists():
        ds_files: list[tuple[str, Path]] = []
        for root, dirs, files in os.walk(datasets_dir, followlinks=False):
            dirs[:] = sorted(dirs)
            rp = Path(root)
            for fname in sorted(files):
                fp = rp / fname
                if fp.is_file() and not fp.is_symlink():
                    ds_files.append((fp.relative_to(datasets_dir).as_posix(), fp))
        ds_files.sort(key=lambda item: item[0])
        h.update(f"ds_count={len(ds_files)}\n".encode("utf-8"))
        for rel_posix, fp in ds_files:
            digest = _hash_file_streaming(fp)
            h.update(f"ds|{rel_posix}|{fp.stat().st_size}|{digest}\n".encode("utf-8"))
    else:
        h.update(b"ds_count=0\n")

    return h.hexdigest()


def _chmod_datasets_for_runner(datasets_dir: Path) -> None:
    if os.geteuid() != 0 or not datasets_dir.exists():
        return
    for root, dirs, files in os.walk(datasets_dir, followlinks=False):
        rp = Path(root)
        if not rp.is_symlink():
            os.chmod(rp, 0o755)
        for dname in dirs:
            dp = rp / dname
            if not dp.is_symlink():
                os.chmod(dp, 0o755)
        for fname in files:
            fp = rp / fname
            if not fp.is_symlink():
                os.chmod(fp, 0o644)


class InRunVerifierState:
    """Thread-safe V-only budgeted verifier state with cached per-content Gaussian noise."""

    def __init__(
        self,
        *,
        sealed_dir: Path,
        shared_dir: Path,
        log_path: Path,
        incumbent_dir: Path | None = None,
        incumbent_datasets_dir: Path | None = None,
        max_queries: int | None = None,
        noise_sigma: float | None = None,
        per_estate_timeout_sec: float | None = None,
        run_user: str | None = None,
        verifier_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.sealed_dir = Path(sealed_dir)
        self.shared_dir = Path(shared_dir)
        self.log_path = Path(log_path)
        self.incumbent_dir = Path(incumbent_dir) if incumbent_dir is not None else None
        self.incumbent_datasets_dir = Path(incumbent_datasets_dir) if incumbent_datasets_dir is not None else None

        cfg = verifier_cfg or yaml.safe_load((self.sealed_dir / "verifier.yaml").read_text())
        inrun_cfg = cfg.get("inrun") or {}
        rt_cfg = cfg.get("runtime") or {}

        self.epoch = int(cfg["epoch"])
        self.weights = {str(k): float(v) for k, v in cfg["reward_weights"].items()}
        self.max_queries = int(max_queries if max_queries is not None else inrun_cfg.get("max_queries", DEFAULT_MAX_QUERIES))
        self.noise_sigma = float(noise_sigma if noise_sigma is not None else inrun_cfg.get("noise_sigma", DEFAULT_NOISE_SIGMA))
        self.per_estate_timeout_sec = float(
            per_estate_timeout_sec
            if per_estate_timeout_sec is not None
            else inrun_cfg.get("per_estate_timeout_sec", DEFAULT_PER_ESTATE_TIMEOUT_SEC)
        )
        self.run_user = str(run_user if run_user is not None else inrun_cfg.get("run_user", DEFAULT_RUN_USER))
        self.pip_install_timeout_sec = float(rt_cfg.get("pip_install_timeout_sec", 600.0))
        self.max_datasets_mb = float(rt_cfg.get("max_datasets_mb", 512.0))
        self.cpus = int(rt_cfg.get("cpus", 4))

        if self.max_queries < 0:
            raise ValueError("max_queries must be >= 0")
        if self.noise_sigma < 0.0:
            raise ValueError("noise_sigma must be >= 0.0")
        if self.per_estate_timeout_sec <= 0.0:
            raise ValueError("per_estate_timeout_sec must be > 0")

        self.bundle = _load_v_only_bundle(self.sealed_dir, cfg)
        self.estates: list[str] = list(self.bundle["estates"])

        self._lock = threading.Lock()
        self._queries_used = 0
        self._eps_cache: dict[str, dict[str, float]] = {}
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []

        self._inc_v_lift: dict[str, float] = {}
        self._coords_v_lift: dict[str, float] = {}
        self._precompute_reference_v_lifts()

    def _precompute_reference_v_lifts(self) -> None:
        tmp_ref = Path(tempfile.mkdtemp(prefix="agentrsi_inrun_ref_"))
        try:
            os.chmod(tmp_ref, 0o755)
            empty_ds = tmp_ref / "_empty_datasets"
            empty_ds.mkdir(parents=True, exist_ok=True)
            os.chmod(empty_ds, 0o755)
            scratch_dir = tmp_ref / "_scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)

            for e in self.estates:
                data = self.bundle["by_estate"][e]
                other = {o: self.bundle["by_estate"][o]["labeled"] for o in self.estates if o != e}
                ctx = {
                    "estate": e,
                    "datasets_dir": empty_ds,
                    "other_labeled": other,
                    "scratch_dir": scratch_dir,
                    "cpus": self.cpus,
                }
                coords_scores = coords_only_fit_predict(data["labeled"], data["unlabeled"], ctx)
                coords_m = estate_metrics(data["v_y"], np.asarray(coords_scores[data["v_mask"]], dtype=float))
                self._coords_v_lift[e] = float(coords_m["pr_auc_lift"])

            if self.incumbent_dir is not None and (self.incumbent_dir / "pipeline.py").exists():
                inc_ds = (
                    self.incumbent_datasets_dir
                    if (self.incumbent_datasets_dir is not None and self.incumbent_datasets_dir.exists())
                    else empty_ds
                )
                _chmod_datasets_for_runner(inc_ds)
                inc_py, _ = _install_submission_deps_isolated(
                    self.incumbent_dir / "requirements.txt",
                    tmp_ref,
                    run_user=self.run_user,
                    timeout_sec=self.pip_install_timeout_sec,
                )
                for e in self.estates:
                    data = self.bundle["by_estate"][e]
                    other = {o: self.bundle["by_estate"][o]["labeled"] for o in self.estates if o != e}
                    inc_scores = run_estate_isolated(
                        python_bin=inc_py,
                        submission_dir=self.incumbent_dir,
                        work_root=tmp_ref,
                        estate=e,
                        labeled_df=data["labeled"],
                        unlabeled_df=data["unlabeled"],
                        other_labeled_dfs=other,
                        datasets_dir=inc_ds,
                        cpus=self.cpus,
                        timeout_sec=self.per_estate_timeout_sec,
                        run_user=self.run_user,
                    )
                    inc_m = estate_metrics(data["v_y"], np.asarray(inc_scores[data["v_mask"]], dtype=float))
                    self._inc_v_lift[e] = float(inc_m["pr_auc_lift"])
            else:
                for e in self.estates:
                    data = self.bundle["by_estate"][e]
                    other = {o: self.bundle["by_estate"][o]["labeled"] for o in self.estates if o != e}
                    ctx = {
                        "estate": e,
                        "datasets_dir": empty_ds,
                        "other_labeled": other,
                        "scratch_dir": scratch_dir,
                        "cpus": self.cpus,
                    }
                    inc_scores = incumbent_fit_predict(data["labeled"], data["unlabeled"], ctx)
                    inc_m = estate_metrics(data["v_y"], np.asarray(inc_scores[data["v_mask"]], dtype=float))
                    self._inc_v_lift[e] = float(inc_m["pr_auc_lift"])
        finally:
            shutil.rmtree(tmp_ref, ignore_errors=True)

    def health(self) -> dict[str, Any]:
        with self._lock:
            used = self._queries_used
        return {
            "ok": True,
            "epoch": self.epoch,
            "estates": list(self.estates),
            "queries_used": used,
            "max_queries": self.max_queries,
        }

    def budget(self) -> dict[str, Any]:
        with self._lock:
            used = self._queries_used
            history = copy.deepcopy(self._history)
        return {
            "ok": True,
            "epoch": self.epoch,
            "estates": list(self.estates),
            "queries_used": used,
            "queries_remaining": max(0, self.max_queries - used),
            "max_queries": self.max_queries,
            "noise_sigma": self.noise_sigma,
            "history": history,
        }

    def _current_budget_snapshot(self) -> tuple[int, int]:
        with self._lock:
            used = self._queries_used
        return used, max(0, self.max_queries - used)

    def _append_audit_record(self, result: dict[str, Any]) -> None:
        summary = {
            "query_index": result.get("queries_used"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_sha256": result.get("content_sha256"),
            "note": result.get("note", ""),
            "status": result.get("status"),
            "submission_valid": result.get("submission_valid"),
            "error": result.get("error"),
            "cached": result.get("cached", False),
            "composite": copy.deepcopy(result.get("composite")),
            "estates": copy.deepcopy(result.get("estates")),
            "datasets_audit": copy.deepcopy(result.get("datasets_audit")),
            "queries_used": result.get("queries_used"),
            "queries_remaining": result.get("queries_remaining"),
            "noise_sigma": result.get("noise_sigma"),
            "runtime_sec": result.get("runtime_sec"),
        }
        with self._lock:
            self._history.append(
                {
                    "query_index": summary["query_index"],
                    "timestamp": summary["timestamp"],
                    "content_sha256": summary["content_sha256"],
                    "note": summary["note"],
                    "status": summary["status"],
                    "submission_valid": summary["submission_valid"],
                    "error": summary["error"],
                    "cached": summary["cached"],
                    "composite": summary["composite"],
                    "runtime_sec": summary["runtime_sec"],
                }
            )
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")

    def score_v(
        self,
        shared_submission_dir: str | None = None,
        shared_datasets_dir: str | None = None,
        pipeline_code: str | None = None,
        requirements_txt: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Validate, execute, and return noisy V-only metrics for a candidate submission."""
        t0 = time.monotonic()

        if (shared_submission_dir is None) == (pipeline_code is None):
            used, rem = self._current_budget_snapshot()
            return {
                "ok": False,
                "error": "INVALID_REQUEST",
                "detail": "Provide exactly one of shared_submission_dir or pipeline_code",
                "queries_used": used,
                "queries_remaining": rem,
                "max_queries": self.max_queries,
            }

        inline_stage_dir: Path | None = None
        try:
            if pipeline_code is not None:
                if not isinstance(pipeline_code, str) or not pipeline_code.strip():
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "INVALID_SUBMISSION",
                        "detail": "pipeline_code must be a non-empty string",
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }
                req_text = requirements_txt or ""
                total_bytes = len(pipeline_code.encode("utf-8")) + len(req_text.encode("utf-8"))
                if total_bytes > MAX_INLINE_PAYLOAD_BYTES:
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "PAYLOAD_TOO_LARGE",
                        "detail": f"Inline submission {total_bytes} bytes exceeds {MAX_INLINE_PAYLOAD_BYTES} bytes",
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }
                inline_stage_dir = Path(tempfile.mkdtemp(prefix="agentrsi_inline_sub_"))
                sub_dir = inline_stage_dir / "submission"
                sub_dir.mkdir(parents=True, exist_ok=True)
                (sub_dir / "pipeline.py").write_text(pipeline_code, encoding="utf-8")
                if requirements_txt is not None:
                    (sub_dir / "requirements.txt").write_text(requirements_txt, encoding="utf-8")
            else:
                try:
                    sub_dir = _resolve_jailed_dir(self.shared_dir, str(shared_submission_dir), "shared_submission_dir")
                except ValueError as exc:
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "INVALID_PATH",
                        "detail": str(exc),
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }
                if not (sub_dir / "pipeline.py").is_file():
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "INVALID_SUBMISSION",
                        "detail": "No pipeline.py found in shared_submission_dir",
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }

            ds_dir: Path | None = None
            if shared_datasets_dir is not None and str(shared_datasets_dir).strip() != "":
                try:
                    ds_dir = _resolve_jailed_dir(self.shared_dir, str(shared_datasets_dir), "shared_datasets_dir")
                except ValueError as exc:
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "INVALID_PATH",
                        "detail": str(exc),
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }
                datasets_audit = validate_datasets(ds_dir, self.bundle["expected_keys"], self.max_datasets_mb)
                if not datasets_audit["valid"]:
                    used, rem = self._current_budget_snapshot()
                    return {
                        "ok": False,
                        "error": "INVALID_DATASETS",
                        "detail": "; ".join(datasets_audit["errors"]),
                        "datasets_audit": datasets_audit,
                        "queries_used": used,
                        "queries_remaining": rem,
                        "max_queries": self.max_queries,
                    }
            else:
                datasets_audit = {"exists": False, "valid": True, "total_mb": 0.0, "datasets": {}, "errors": []}

            content_sha256 = _compute_content_sha256(sub_dir, ds_dir)

            with self._lock:
                if self._queries_used >= self.max_queries:
                    return {
                        "ok": False,
                        "error": "BUDGET_EXHAUSTED",
                        "queries_used": self._queries_used,
                        "queries_remaining": 0,
                        "max_queries": self.max_queries,
                    }
                self._queries_used += 1
                query_idx = self._queries_used
                queries_remaining = max(0, self.max_queries - query_idx)

                cached_template = self._result_cache.get(content_sha256)
                if content_sha256 not in self._eps_cache:
                    rng = secrets.SystemRandom()
                    self._eps_cache[content_sha256] = {
                        e: rng.gauss(0.0, self.noise_sigma) for e in self.estates
                    }
                eps_by_estate = dict(self._eps_cache[content_sha256])

            if cached_template is not None:
                res = copy.deepcopy(cached_template)
                res["cached"] = True
                res["note"] = str(note or "")
                res["queries_used"] = query_idx
                res["queries_remaining"] = queries_remaining
                res["runtime_sec"] = round(time.monotonic() - t0, 3)
                self._append_audit_record(res)
                return res

            tmp_root = Path(tempfile.mkdtemp(prefix="agentrsi_inrun_"))
            try:
                os.chmod(tmp_root, 0o755)
                if ds_dir is not None:
                    _chmod_datasets_for_runner(ds_dir)
                    run_ds_dir = ds_dir
                else:
                    run_ds_dir = tmp_root / "_empty_datasets"
                    run_ds_dir.mkdir(parents=True, exist_ok=True)
                    os.chmod(run_ds_dir, 0o755)

                py_bin, _ = _install_submission_deps_isolated(
                    sub_dir / "requirements.txt",
                    tmp_root,
                    run_user=self.run_user,
                    timeout_sec=self.pip_install_timeout_sec,
                )

                estates_noisy: dict[str, dict[str, float]] = {}
                for e in self.estates:
                    data = self.bundle["by_estate"][e]
                    other = {o: self.bundle["by_estate"][o]["labeled"] for o in self.estates if o != e}
                    cand_scores = run_estate_isolated(
                        python_bin=py_bin,
                        submission_dir=sub_dir,
                        work_root=tmp_root,
                        estate=e,
                        labeled_df=data["labeled"],
                        unlabeled_df=data["unlabeled"],
                        other_labeled_dfs=other,
                        datasets_dir=run_ds_dir,
                        cpus=self.cpus,
                        timeout_sec=self.per_estate_timeout_sec,
                        run_user=self.run_user,
                    )
                    raw_m = estate_metrics(data["v_y"], np.asarray(cand_scores[data["v_mask"]], dtype=float))
                    raw_lift_e = float(raw_m["pr_auc_lift"])
                    eps_e = float(eps_by_estate[e])
                    estates_noisy[e] = {
                        "noisy_pr_auc_lift": float(raw_lift_e + eps_e),
                        "noisy_delta_vs_incumbent": float((raw_lift_e - self._inc_v_lift[e]) + eps_e),
                        "noisy_delta_vs_coords": float((raw_lift_e - self._coords_v_lift[e]) + eps_e),
                    }

                composite_noisy = {
                    "noisy_pr_auc_lift": weighted_mean(
                        {e: estates_noisy[e]["noisy_pr_auc_lift"] for e in self.estates},
                        self.weights,
                    ),
                    "noisy_delta_vs_incumbent": weighted_mean(
                        {e: estates_noisy[e]["noisy_delta_vs_incumbent"] for e in self.estates},
                        self.weights,
                    ),
                    "noisy_delta_vs_coords": weighted_mean(
                        {e: estates_noisy[e]["noisy_delta_vs_coords"] for e in self.estates},
                        self.weights,
                    ),
                }
                res = {
                    "ok": True,
                    "epoch": self.epoch,
                    "status": "ok",
                    "submission_valid": True,
                    "error": None,
                    "content_sha256": content_sha256,
                    "note": str(note or ""),
                    "cached": False,
                    "composite": composite_noisy,
                    "estates": estates_noisy,
                    "datasets_audit": datasets_audit,
                    "queries_used": query_idx,
                    "queries_remaining": queries_remaining,
                    "max_queries": self.max_queries,
                    "noise_sigma": self.noise_sigma,
                    "runtime_sec": round(time.monotonic() - t0, 3),
                }
                with self._lock:
                    self._result_cache[content_sha256] = copy.deepcopy(res)
            except Exception as exc:
                res = {
                    "ok": False,
                    "epoch": self.epoch,
                    "status": "failed",
                    "submission_valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "content_sha256": content_sha256,
                    "note": str(note or ""),
                    "cached": False,
                    "composite": None,
                    "estates": None,
                    "datasets_audit": datasets_audit,
                    "queries_used": query_idx,
                    "queries_remaining": queries_remaining,
                    "max_queries": self.max_queries,
                    "noise_sigma": self.noise_sigma,
                    "runtime_sec": round(time.monotonic() - t0, 3),
                }
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

            self._append_audit_record(res)
            return res
        finally:
            if inline_stage_dir is not None:
                shutil.rmtree(inline_stage_dir, ignore_errors=True)


async def _send_json(send: Any, status: int, payload: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _extract_bearer_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() == b"authorization":
            decoded = value.decode("latin-1").strip()
            parts = decoded.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
            return ""
    return None


def build_asgi_app(
    state: InRunVerifierState,
    *,
    token_env: str = DEFAULT_TOKEN_ENV,
    allow_no_auth: bool = False,
):
    """Build the pure-ASGI wrapper around MCPServer Streamable HTTP + /health + Bearer auth."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    expected_token = os.environ.get(token_env, "")
    if not allow_no_auth and not expected_token:
        raise RuntimeError(f"Required bearer token env var {token_env!r} is unset or empty")

    mcp_server = MCPServer("agentrsi-verifier")

    @mcp_server.tool()
    def verifier_budget() -> dict[str, Any]:
        """Return remaining V-only query budget, noise sigma, and query history (no budget cost)."""
        return state.budget()

    @mcp_server.tool()
    def verifier_health() -> dict[str, Any]:
        """Return verifier health, epoch, estates, and current query usage."""
        return state.health()

    @mcp_server.tool()
    async def score_v(
        shared_submission_dir: str | None = None,
        shared_datasets_dir: str | None = None,
        pipeline_code: str | None = None,
        requirements_txt: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Run candidate submission on V partition only and return noisy triage lift metrics."""
        return await anyio.to_thread.run_sync(
            lambda: state.score_v(
                shared_submission_dir=shared_submission_dir,
                shared_datasets_dir=shared_datasets_dir,
                pipeline_code=pipeline_code,
                requirements_txt=requirements_txt,
                note=note,
            )
        )

    inner_app = mcp_server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host="0.0.0.0",
    )

    async def asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await inner_app(scope, receive, send)
            return

        if scope_type == "http":
            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()

            if path == "/health":
                if method != "GET":
                    await _send_json(send, 405, {"error": "method_not_allowed"})
                    return
                await _send_json(send, 200, state.health())
                return

            if not allow_no_auth:
                presented = _extract_bearer_token(scope.get("headers", []))
                if presented is None or not hmac.compare_digest(presented, expected_token):
                    await _send_json(
                        send,
                        401,
                        {"error": "unauthorized"},
                        extra_headers=[(b"www-authenticate", b"Bearer")],
                    )
                    return

        await inner_app(scope, receive, send)

    return asgi_app


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AgentRSI In-Run Verifier v2 MCP sidecar server.")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--sealed-dir", type=Path, default=Path("/sealed"))
    ap.add_argument("--incumbent-dir", type=Path, default=None)
    ap.add_argument("--incumbent-datasets-dir", type=Path, default=None)
    ap.add_argument("--shared-dir", type=Path, default=Path("/shared"))
    ap.add_argument("--log-path", type=Path, default=Path("/var/log/agentrsi-verifier/queries.jsonl"))
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--noise-sigma", type=float, default=None)
    ap.add_argument("--per-estate-timeout-sec", type=float, default=None)
    ap.add_argument("--run-user", type=str, default=None)
    ap.add_argument("--token-env", type=str, default=DEFAULT_TOKEN_ENV)
    ap.add_argument("--allow-no-auth", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _build_arg_parser().parse_args(argv)
    verifier_cfg = yaml.safe_load((args.sealed_dir / "verifier.yaml").read_text())
    inrun_cfg = verifier_cfg.get("inrun") or {}
    port = int(args.port if args.port is not None else inrun_cfg.get("port", DEFAULT_PORT))

    state = InRunVerifierState(
        sealed_dir=args.sealed_dir,
        shared_dir=args.shared_dir,
        log_path=args.log_path,
        incumbent_dir=args.incumbent_dir,
        incumbent_datasets_dir=args.incumbent_datasets_dir,
        max_queries=args.max_queries,
        noise_sigma=args.noise_sigma,
        per_estate_timeout_sec=args.per_estate_timeout_sec,
        run_user=args.run_user,
        verifier_cfg=verifier_cfg,
    )
    app = build_asgi_app(
        state,
        token_env=args.token_env,
        allow_no_auth=args.allow_no_auth,
    )
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
