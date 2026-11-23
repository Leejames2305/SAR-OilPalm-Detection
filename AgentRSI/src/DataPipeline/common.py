"""Shared paths, constants and helpers for the data pipeline.

Every step imports this, so the working directory does not matter. Split
parameters are loaded from config/split.yaml (single source of truth).
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import yaml

PIPE_DIR = Path(__file__).resolve().parent
AGENT_ROOT = PIPE_DIR.parents[1]            # AgentRSI/
REPO_ROOT = AGENT_ROOT.parent               # holds data/Labels and data/SAR-scenes

# Raw inputs
LABEL_DIR = REPO_ROOT / "data" / "Labels"
LABEL_FILES = {
    "AirHitam": "AirHitam-Classification_2026.csv",
    "Palong": "Palong-Classification_2022.csv",
    "Serting": "Serting-Classification_2022.csv",
}
ESTATES = list(LABEL_FILES)
CLASSES = ("Healthy", "Unhealthy", "Middle")
KEY_FILE = AGENT_ROOT / ".secrets" / "Keys_SABucketRead.json"   # Earth Engine service account, never print it

# SAR and optical feature definitions (SAR dates auto-discovered from data/SAR-scenes/)
SAR_DIR = REPO_ROOT / "data" / "SAR-scenes"
SAR_PRODUCTS = {
    "Spk": ("Cal_Spk_TC.tif", ["HH", "HV", "VH", "VV"]),
    "HAlpha": ("Cal_mat_HAlpha_Spk_TC.tif", ["HAlpha_H", "HAlpha_A", "HAlpha_alpha"]),
}
NDVI_COLS = ["B4", "B8", "NDVI", "clear_count"]


def discover_sar_dates() -> list[str]:
    """Return sorted SAR scene dates present for every estate and configured product."""
    if not SAR_DIR.exists():
        return []
    first_estate = ESTATES[0]
    first_suffix = next(iter(SAR_PRODUCTS.values()))[0]
    pat = re.compile(rf"^ALOS2-Subset_{re.escape(first_estate)}_(\d+)_{re.escape(first_suffix)}$")
    candidates = sorted(
        m.group(1) for p in SAR_DIR.iterdir() if (m := pat.match(p.name))
    )
    complete: list[str] = []
    for dt in candidates:
        if all(
            (SAR_DIR / f"ALOS2-Subset_{e}_{dt}_{suffix}").exists()
            for e in ESTATES
            for suffix, _ in SAR_PRODUCTS.values()
        ):
            complete.append(dt)
    return complete


def sar_feature_cols(dates: list[str] | None = None) -> list[str]:
    """Date-suffixed SAR column names for all configured products."""
    use_dates = discover_sar_dates() if dates is None else dates
    return [
        f"{col}_{dt}"
        for dt in use_dates
        for _, cols in SAR_PRODUCTS.values()
        for col in cols
    ]


def feature_cols(dates: list[str] | None = None) -> list[str]:
    """Full base feature column list (multi-date SAR + Sentinel-2 NDVI)."""
    return sar_feature_cols(dates) + list(NDVI_COLS)


SAR_DATES = discover_sar_dates()
FEATURE_COLS = feature_cols(SAR_DATES)

# Config: split.yaml is the single source of truth for the frozen split
CONFIG = AGENT_ROOT / "config"
SPLIT_CONFIG = CONFIG / "split.yaml"
_SPLIT_CFG = yaml.safe_load(SPLIT_CONFIG.read_text())

EPOCH = int(_SPLIT_CFG["epoch"])
BLOCK_M = int(_SPLIT_CFG["block_m"])
SPLIT_SEED = int(_SPLIT_CFG["seed"])
SPLIT_FRAC = {k: float(v) for k, v in _SPLIT_CFG["fractions"].items()}
SPLIT_MIN = {k: int(v) for k, v in _SPLIT_CFG["minimum_unhealthy"].items()}
SPLIT_TOL = float(_SPLIT_CFG["tree_share_tolerance"])
SPLIT_SHA256 = str(_SPLIT_CFG["sha256_of_assignment"])
SPLIT = AGENT_ROOT / _SPLIT_CFG["assignment_file"]
SPLIT_COUNTS = _SPLIT_CFG.get("counts", {})

# Verifier side: git-ignored, directory 700, files 600. Nothing here goes into tasks/ or jobs/.
SEALED = AGENT_ROOT / ".sealed"
MASTER = SEALED / "labels_master.csv"       # estate,id,Long,Lat,class
SEALED_REPORT = SEALED / "split_report.json"
VERIFIER_CONTEXT = SEALED / "verifier_context"

# Derived: git-ignored
WORK = AGENT_ROOT / ".work"
FEATURES = WORK / "features.csv"            # frozen feature table, no labels
AGENT_VIEW = WORK / "agent_view"
GATE_RESULTS = WORK / "gate_results.json"


def block_keys(lon, lat, bs):
    """Spatial block id per tree: square cells of bs metres in a local metric frame."""
    mx = (lon - lon.min()) * 111320 * np.cos(np.radians(lat.mean()))
    my = (lat - lat.min()) * 110540
    return np.floor(mx / bs).astype(np.int64) * 100000 + np.floor(my / bs).astype(np.int64)
