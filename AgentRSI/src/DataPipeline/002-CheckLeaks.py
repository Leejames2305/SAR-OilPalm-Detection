"""Step 2: sealed-label and split-leak check on the agent view, tasks/ and jobs/. Run after every build.

View checks (--view, default .work/agent_view):
  - exact file set: L_labelled.csv and unlabelled.csv only (fails if V/T are split or extra files exist)
  - allowlisted columns only; no id, block or part column; unlabelled.csv has no label/class column
  - each row is a master row (estate, Long, Lat), sits in the right file (L vs V+T),
    and the row sets match the frozen split with zero overlap
  - no class word (case-insensitive) anywhere in unlabelled.csv
Scan checks (--scan, default tasks/ and jobs/ under AGENT_ROOT; missing dirs are reported and skipped):
  - filename/path check (all files, including binaries): no sealed artefacts, raw label files,
    split_assignment.csv, or separate V_unlabelled.csv / T_unlabelled.csv
  - binary files (rasters, images, archives, NUL-byte files) skip text regexes
  - CSV files: no id/block/part header; no sealed id in a non-allowlisted column; no sealed
    coordinate pair paired with a label/class value
  - all text files: no exact sealed master line, no exact sealed split_assignment line,
    and no sealed coordinate pair on a line that also carries a class word (case-insensitive)
Mode and ignore checks: .sealed/ is 700 with files and dirs no looser than 600/700; .sealed/ and .work/ are git-ignored.

--control writes deliberately leaky fixtures (view CSV, task CSV, lowercase-class JSON, and
split-assignment leak) under .sealed/control_leaky and passes only if every check catches them.

Exit 0 = PASS (or CONTROL CAUGHT), 1 = FAIL (or CONTROL MISSED).

Usage: python 002-CheckLeaks.py [--view DIR | --no-view] [--scan DIR ...] [--control]
"""
import argparse
import csv
import io
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import AGENT_ROOT, AGENT_VIEW, FEATURE_COLS, LABEL_FILES, MASTER, SEALED, SPLIT

CLASS_WORDS = ("healthy", "unhealthy", "middle")
CLASS_WORD_RE = re.compile(r"\b(?:healthy|unhealthy|middle)\b", re.IGNORECASE)
FORBIDDEN_HEADERS = {"id", "fid", "objectid", "tree_id", "treeid", "row_id", "block", "part"}
ALLOWED_COLS = {"estate", "Long", "Lat", "label", *FEATURE_COLS}
FORBIDDEN_NAMES = {
    "labels_master.csv",
    "split_report.json",
    "split_assignment.csv",
    "verifier_context",
    "V_unlabelled.csv",
    "T_unlabelled.csv",
    *LABEL_FILES.values(),
}
RAW_LABEL_STEMS = {Path(fn).stem for fn in LABEL_FILES.values()}
BINARY_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z",
    ".parquet", ".feather", ".npy", ".npz", ".pkl", ".joblib",
    ".pyc", ".pyo", ".whl", ".so", ".dll", ".dylib", ".bin",
}
VIEW_FILES = {"L": "L_labelled.csv", "U": "unlabelled.csv"}
COORD = re.compile(r"-?\d+\.\d+")

_MASTER_DF = pd.read_csv(MASTER, dtype=str)
_SPLIT_DF = pd.read_csv(SPLIT, dtype=str)
REF = _MASTER_DF.merge(_SPLIT_DF[["estate", "id", "block", "part"]], on=["estate", "id"], validate="one_to_one")
SEALED_ROWS = REF[REF.part != "L"]
SEALED_IDS = {e: set(g["id"]) for e, g in SEALED_ROWS.groupby("estate")}
ALL_SEALED_IDS = set().union(*SEALED_IDS.values())
SEALED_MASTER_LINES = set(
    ",".join(map(str, r))
    for r in SEALED_ROWS[["estate", "id", "Long", "Lat", "class"]].itertuples(index=False)
)
SEALED_SPLIT_LINES = set(
    ",".join(map(str, r))
    for r in SEALED_ROWS[["estate", "id", "block", "part"]].itertuples(index=False)
)
LATS_OF: dict[str, set[str]] = {}
for lo, la in zip(SEALED_ROWS["Long"], SEALED_ROWS["Lat"]):
    LATS_OF.setdefault(lo, set()).add(la)


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(AGENT_ROOT))
    except ValueError:
        return str(p)


def is_binary(f: Path) -> bool:
    if f.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        with open(f, "rb") as fh:
            chunk = fh.read(8192)
        return b"\x00" in chunk
    except OSError:
        return False


def view_checks(view_dir: Path, fails: list[str]):
    expected = {(r.estate, r.Long, r.Lat): r.part for r in REF.itertuples(index=False)}
    actual_files = {p.name for p in view_dir.iterdir() if p.is_file()}
    unexpected = sorted(actual_files - set(VIEW_FILES.values()))
    if unexpected:
        fails.append(f"view: unexpected files in {rel(view_dir)}: {unexpected}")

    seen = set()
    for role, name in VIEW_FILES.items():
        f = view_dir / name
        if not f.exists():
            fails.append(f"view: missing {name}")
            continue
        text = f.read_text()
        header = next(csv.reader(io.StringIO(text)))
        if set(header) - ALLOWED_COLS:
            fails.append(f"{name}: columns outside the allowlist {sorted(set(header) - ALLOWED_COLS)}")
        if {h.lower() for h in header} & FORBIDDEN_HEADERS:
            fails.append(f"{name}: forbidden id, block or part column")
        if role == "L" and "label" not in header:
            fails.append(f"{name}: no label column on L file")
        if role == "U" and {"label", "class"} & {h.lower() for h in header}:
            fails.append(f"{name}: label or class column on unlabelled file")
        if role == "U" and any(w in text.lower() for w in CLASS_WORDS):
            fails.append(f"{name}: class word in unlabelled file")

        df = pd.read_csv(f, dtype=str)
        if not {"estate", "Long", "Lat"} <= set(df.columns):
            fails.append(f"{name}: missing estate, Long or Lat column")
            continue
        keys = set(zip(df["estate"], df["Long"], df["Lat"]))
        if len(keys) != len(df):
            fails.append(f"{name}: duplicate tree")

        want_parts = {"L"} if role == "L" else {"V", "T"}
        bad = sum(expected.get(k) not in want_parts for k in keys)
        if bad:
            fails.append(f"{name}: {bad} rows do not belong to split part(s) {sorted(want_parts)}")
        want = {k for k, v in expected.items() if v in want_parts}
        if keys != want:
            fails.append(f"{name}: row set differs from split part(s) {sorted(want_parts)} ({len(keys)} vs {len(want)})")
        if keys & seen:
            fails.append("same tree appears in both L_labelled.csv and unlabelled.csv")
        seen |= keys


def csv_checks(f: Path, text: str, fails: list[str]):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return
    header = rows[0]
    header_lc = {h.lower() for h in header}
    if header_lc & FORBIDDEN_HEADERS:
        fails.append(f"{rel(f)}: forbidden id/block/part column header")
    est_i = header.index("estate") if "estate" in header else None
    extra = [i for i, h in enumerate(header) if h not in ALLOWED_COLS]
    for row in rows[1:]:
        ids = SEALED_IDS.get(row[est_i], set()) if est_i is not None and est_i < len(row) else ALL_SEALED_IDS
        for i in extra:
            if i < len(row) and row[i] in ids:
                fails.append(f"{rel(f)}: sealed id in column '{header[i]}'")
                return


def line_checks(f: Path, text: str, fails: list[str]):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in SEALED_MASTER_LINES:
            fails.append(f"{rel(f)}: exact sealed master line")
            return
        if stripped in SEALED_SPLIT_LINES:
            fails.append(f"{rel(f)}: exact sealed split_assignment line")
            return
        if CLASS_WORD_RE.search(line):
            toks = COORD.findall(line)
            if toks:
                tok_set = set(toks)
                for a in toks:
                    if a in LATS_OF and (LATS_OF[a] & tok_set):
                        fails.append(f"{rel(f)}: sealed coordinate pair with a class word")
                        return


def scan(roots: list[Path], fails: list[str], notes: list[str]):
    for root in roots:
        if not root.exists():
            notes.append(f"scan: {rel(root)} absent, skipped")
            continue
        files = [p for p in sorted(root.rglob("*")) if p.is_file()]
        skipped_bin = 0
        for f in files:
            if (
                f.name in FORBIDDEN_NAMES
                or f.stem in RAW_LABEL_STEMS
                or "verifier_context" in f.parts
            ):
                fails.append(f"{rel(f)}: forbidden sealed/raw artefact name")
            if is_binary(f):
                skipped_bin += 1
                continue
            text = f.read_text(errors="replace")
            if f.suffix.lower() == ".csv":
                csv_checks(f, text, fails)
            line_checks(f, text, fails)
        notes.append(f"scan: {len(files)} files under {rel(root)} ({skipped_bin} binary skipped for text scan)")


def mode_checks(fails: list[str]):
    if not SEALED.exists():
        fails.append(".sealed/ missing")
        return
    if stat.S_IMODE(SEALED.stat().st_mode) != 0o700:
        fails.append(".sealed/ is not mode 700")
    for p in SEALED.rglob("*"):
        if stat.S_IMODE(p.stat().st_mode) & 0o077:
            fails.append(f"{rel(p)}: group or other bits set")
    for path in (".sealed/", ".work/"):
        r = subprocess.run(["git", "-C", str(AGENT_ROOT), "check-ignore", "-q", path], capture_output=True)
        if r.returncode != 0:
            fails.append(f"{path} is not git-ignored")


def run_control() -> int:
    """Seed multiple realistic leaks and require every check to catch them."""
    control = SEALED / "control_leaky"
    view_dir = control / "agent_view"
    tasks_csv = control / "tasks_csv"
    tasks_json = control / "tasks_json"
    tasks_split = control / "tasks_split"
    for d in (control, view_dir, tasks_csv, tasks_json, tasks_split):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)

    leak = REF[REF.part == "V"][["estate", "id", "Long", "Lat", "class"]]
    for path in (view_dir / "unlabelled.csv", tasks_csv / "V_leaky.csv"):
        leak.to_csv(path, index=False)
        os.chmod(path, 0o600)
    shutil.copy(AGENT_VIEW / VIEW_FILES["L"], view_dir / VIEW_FILES["L"])
    os.chmod(view_dir / VIEW_FILES["L"], 0o600)

    # Non-CSV lowercase class leak with sealed coordinates
    row0 = SEALED_ROWS.iloc[0]
    json_leak = tasks_json / "dump.json"
    json_leak.write_text(f'{{"lon": {row0["Long"]}, "lat": {row0["Lat"]}, "cls": "{row0["class"].lower()}"}}\n')
    os.chmod(json_leak, 0o600)

    # Split assignment leak in a plain text file
    split_leak = tasks_split / "notes.txt"
    split_leak.write_text(f'{row0["estate"]},{row0["id"]},{row0["block"]},{row0["part"]}\n')
    os.chmod(split_leak, 0o600)

    me = str(Path(__file__).resolve())
    runs = {
        "view + csv scan": [sys.executable, me, "--view", str(view_dir), "--scan", str(tasks_csv)],
        "csv scan only": [sys.executable, me, "--no-view", "--scan", str(tasks_csv)],
        "lowercase coord json": [sys.executable, me, "--no-view", "--scan", str(tasks_json)],
        "split line leak": [sys.executable, me, "--no-view", "--scan", str(tasks_split)],
    }
    caught = True
    try:
        for name, cmd in runs.items():
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(f"[{name}] exit {r.returncode}")
            for line in (r.stdout + r.stderr).strip().splitlines()[-8:]:
                print("   ", line)
            caught &= r.returncode == 1 and "FAIL:" in r.stdout and "Traceback" not in r.stderr
    finally:
        shutil.rmtree(control)  # the seeded leaks must not outlive the test

    print("CONTROL CAUGHT" if caught else "CONTROL MISSED: the check passed a seeded leak")
    return 0 if caught else 1


def main():
    ap = argparse.ArgumentParser(description="Sealed-label and split-leak check for agent views and task/job trees.")
    ap.add_argument("--view", type=Path, default=AGENT_VIEW)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--scan", type=Path, nargs="*", default=[AGENT_ROOT / "tasks", AGENT_ROOT / "jobs"])
    ap.add_argument("--control", action="store_true", help="seeded-leak control: every leak must be caught")
    args = ap.parse_args()
    if args.control:
        sys.exit(run_control())

    fails: list[str] = []
    notes: list[str] = []
    if not args.no_view:
        if args.view.exists():
            view_checks(args.view, fails)
            notes.append(f"view: checked {rel(args.view)}")
        else:
            fails.append(f"view directory {args.view} missing")
    scan(args.scan, fails, notes)
    mode_checks(fails)

    for n in notes:
        print(" ", n)
    if fails:
        print("FAIL:", len(fails), "finding(s)")
        for f in fails[:50]:
            print("  -", f)
        sys.exit(1)
    print("PASS: no sealed id, label or artefact found in the checked trees")


if __name__ == "__main__":
    main()
