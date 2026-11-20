"""Shared paths, constants and helpers for the data pipeline. Every step imports this, so the working directory does not matter."""
from pathlib import Path

import numpy as np

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
FEATURE_COLS = ["HH", "HV", "VH", "VV", "HAlpha_1", "HAlpha_2", "HAlpha_3", "B4", "B8", "NDVI", "clear_count"]

# Frozen split rule (plan Decisions). 001-BuildData refuses to write a split that does not reproduce SPLIT_SHA256.
BLOCK_M = 80
SPLIT_SEED = 20261119
SPLIT_FRAC = {"L": 0.40, "V": 0.30, "T": 0.30}
SPLIT_MIN = {"L": 50, "V": 35, "T": 35}     # Unhealthy trees per part
SPLIT_TOL = 0.10                             # tree-share tolerance per part
SPLIT_SHA256 = "26f34d4d8e0159e53985fce735adf410e0cae392d7f1cb8aded8a7490cd0e2ed"

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

# Config: split with raw ids and part, no labels
CONFIG = AGENT_ROOT / "config"
SPLIT = CONFIG / "split_assignment.csv"


def block_keys(lon, lat, bs):
    """Spatial block id per tree: square cells of bs metres in a local metric frame."""
    mx = (lon - lon.min()) * 111320 * np.cos(np.radians(lat.mean()))
    my = (lat - lat.min()) * 110540
    return np.floor(mx / bs).astype(np.int64) * 100000 + np.floor(my / bs).astype(np.int64)
