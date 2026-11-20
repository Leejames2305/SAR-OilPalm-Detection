"""Step 2: sealed-ID check on the agent view, tasks/ and jobs/. Run after every build.

View checks (--view, default .work/agent_view):
  - allowlisted columns only; no id or block column; V and T have no label/class column; L has label
  - each row is a master row (estate and coordinates), sits in the file its split part says,
    and the row set equals the split part
  - no class word in V or T files
Scan checks (--scan, default tasks/ and jobs/ under AGENT_ROOT; missing dirs are reported and skipped):
  - no file named like a sealed artefact (labels_master.csv, split_report.json, verifier_context)
  - no id-like CSV header; no sealed id in a non-allowlisted CSV column of that estate
  - no exact sealed master line (estate,id,Long,Lat,class)
  - no sealed coordinate pair on a line that also carries a class word
  - non-CSV files: no estate name and sealed id as tokens on one line
Mode and ignore checks: .sealed/ is 700 with files and dirs no looser than 600/700; .sealed/ and .work/ are git-ignored.

--control writes a deliberately leaky V file under .sealed/control_leaky, runs these checks on it, and
passes only if both runs are caught. The leaky folder is deleted afterwards.

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

from common import AGENT_ROOT, AGENT_VIEW, FEATURE_COLS, MASTER, SEALED, SPLIT

CLASS_WORDS = ("Healthy", "Unhealthy", "Middle")
ID_HEADERS = {"id", "fid", "objectid", "tree_id", "treeid", "row_id"}
ALLOWED_COLS = {"estate", "Long", "Lat", "label", *FEATURE_COLS}
ARTEFACTS = {"labels_master.csv", "split_report.json", "verifier_context"}
VIEW_FILES = {"L": "L_labelled.csv", "V": "V_unlabelled.csv", "T": "T_unlabelled.csv"}
COORD = re.compile(r"-?\d+\.\d+")
INT = re.compile(r"(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])")

REF = pd.read_csv(MASTER, dtype=str).merge(
    pd.read_csv(SPLIT, dtype=str)[["estate", "id", "part"]], on=["estate", "id"], validate="one_to_one"
)
SEALED_ROWS = REF[REF.part != "L"]
SEALED_IDS = {e: set(g["id"]) for e, g in SEALED_ROWS.groupby("estate")}
ALL_SEALED_IDS = set().union(*SEALED_IDS.values())
SEALED_LINES = set(",".join(map(str, r))
                   for r in SEALED_ROWS[["estate", "id", "Long", "Lat", "class"]].itertuples(index=False))
LATS_OF = {}
for lo, la in zip(SEALED_ROWS["Long"], SEALED_ROWS["Lat"]):
    LATS_OF.setdefault(lo, set()).add(la)


def rel(p):
    try:
        return str(p.relative_to(AGENT_ROOT))
    except ValueError:
        return str(p)


def view_checks(view_dir, fails):
    expected = {(r.estate, r.Long, r.Lat): r.part for r in REF.itertuples(index=False)}
    seen = set()
    for part, name in VIEW_FILES.items():
        f = view_dir / name
        if not f.exists():
            fails.append(f"view: missing {name}")
            continue
        text = f.read_text()
        header = next(csv.reader(io.StringIO(text)))
        if set(header) - ALLOWED_COLS:
            fails.append(f"{name}: columns outside the allowlist {sorted(set(header) - ALLOWED_COLS)}")
        if {h.lower() for h in header} & ID_HEADERS or "block" in header:
            fails.append(f"{name}: id or block column")
        if part == "L" and "label" not in header:
            fails.append(f"{name}: no label column on an L file")
        if part != "L" and {"label", "class"} & set(header):
            fails.append(f"{name}: label or class column on a sealed file")
        if part != "L" and any(w in text for w in CLASS_WORDS):
            fails.append(f"{name}: class word in a sealed file")

        df = pd.read_csv(f, dtype=str)
        if not {"estate", "Long", "Lat"} <= set(df.columns):
            fails.append(f"{name}: no estate, Long or Lat column")
            continue
        keys = set(zip(df["estate"], df["Long"], df["Lat"]))
        if len(keys) != len(df):
            fails.append(f"{name}: duplicate tree")
        bad = sum(expected.get(k) != part for k in keys)
        if bad:
            fails.append(f"{name}: {bad} rows are not master rows of split part {part}")
        want = {k for k, v in expected.items() if v == part}
        if keys != want:
            fails.append(f"{name}: row set differs from split part {part} ({len(keys)} vs {len(want)})")
        if keys & seen:
            fails.append("same tree in two view files")
        seen |= keys


def csv_checks(f, text, fails):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return
    header = rows[0]
    if {h.lower() for h in header} & ID_HEADERS:
        fails.append(f"{rel(f)}: id-like column header")
    est_i = header.index("estate") if "estate" in header else None
    extra = [i for i, h in enumerate(header) if h not in ALLOWED_COLS]
    for row in rows[1:]:
        ids = SEALED_IDS.get(row[est_i], set()) if est_i is not None and est_i < len(row) else ALL_SEALED_IDS
        for i in extra:
            if i < len(row) and row[i] in ids:
                fails.append(f"{rel(f)}: sealed id in column '{header[i]}'")
                return


def token_checks(f, text, fails):
    for line in text.splitlines():
        for e, ids in SEALED_IDS.items():
            if e in line and any(t in ids for t in INT.findall(line)):
                fails.append(f"{rel(f)}: estate name and sealed id on one line")
                return


def line_checks(f, text, fails):
    for line in text.splitlines():
        if line.strip() in SEALED_LINES:
            fails.append(f"{rel(f)}: exact sealed master line")
            return
        toks = COORD.findall(line)
        hits = [a for a in toks if a in LATS_OF]
        if hits and any(w in line for w in CLASS_WORDS):
            for a in hits:
                if LATS_OF[a] & set(toks):
                    fails.append(f"{rel(f)}: sealed coordinate pair with a class word")
                    return


def scan(roots, fails, notes):
    for root in roots:
        if not root.exists():
            notes.append(f"scan: {rel(root)} absent, skipped")
            continue
        files = [p for p in sorted(root.rglob("*")) if p.is_file()]
        notes.append(f"scan: {len(files)} files under {rel(root)}")
        for f in files:
            if f.name in ARTEFACTS or "verifier_context" in f.parts:
                fails.append(f"{rel(f)}: sealed artefact name")
            text = f.read_text(errors="replace")
            if f.suffix.lower() == ".csv":
                csv_checks(f, text, fails)
            else:
                token_checks(f, text, fails)
            line_checks(f, text, fails)


def mode_checks(fails):
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


def run_control():
    """Seed a leak (every V row with raw id, estate, coordinates and class) and require the checks to catch it.

    The L and T files are copied from the real view, so the only failure left is the seeded V leak.
    """
    control = SEALED / "control_leaky"
    view_dir, tasks_dir = control / "agent_view", control / "tasks"
    for d in (control, view_dir, tasks_dir):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
    leak = REF[REF.part == "V"][["estate", "id", "Long", "Lat", "class"]]
    for path in (view_dir / "V_unlabelled.csv", tasks_dir / "V_leaky.csv"):
        leak.to_csv(path, index=False)
        os.chmod(path, 0o600)
    for name in (VIEW_FILES["L"], VIEW_FILES["T"]):
        shutil.copy(AGENT_VIEW / name, view_dir / name)
        os.chmod(view_dir / name, 0o600)

    me = str(Path(__file__).resolve())
    runs = {
        "view + scan": [sys.executable, me, "--view", str(view_dir), "--scan", str(tasks_dir)],
        "scan only": [sys.executable, me, "--no-view", "--scan", str(tasks_dir)],
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
        shutil.rmtree(control)  # the seeded leak must not outlive the test

    print("CONTROL CAUGHT" if caught else "CONTROL MISSED: the check passed a seeded leak")
    return 0 if caught else 1


def main():
    ap = argparse.ArgumentParser(description="Sealed-ID check for agent views and task/job trees.")
    ap.add_argument("--view", type=Path, default=AGENT_VIEW)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--scan", type=Path, nargs="*", default=[AGENT_ROOT / "tasks", AGENT_ROOT / "jobs"])
    ap.add_argument("--control", action="store_true", help="seeded-leak control: the checks must catch it")
    args = ap.parse_args()
    if args.control:
        sys.exit(run_control())

    fails, notes = [], []
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
