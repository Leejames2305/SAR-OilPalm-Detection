#!/usr/bin/env python3
"""Convert the estate label CSVs to KML for Google Earth Web.

    CSVs format:
    id,Long,Lat,Class        # Class: Healthy / Middle / Unhealthy

Usage:
    python convert_label_to_kml.py                    # defaults below
    python convert_label_to_kml.py --output-dir misc/scripts/output
"""

import argparse
from pathlib import Path

import pandas as pd

from Shared_KML_utils import CLASS_STYLE_RGB, write_kml

# --- Defaults ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "data" / "Label-status"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Matches the Palong/Serting basemap label CSVs; excludes AHCensus.csv,
# which is in a different (projected, wkt_geom) format -- see convert_ah_census.py.
INPUT_GLOB = "*_Basemap Classification.csv"

REQUIRED_COLS = {"id", "Long", "Lat", "Class"}


def convert_csv(path: Path, output_dir: Path) -> tuple[Path, dict]:
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected column(s) {sorted(missing)}")

    out_path = output_dir / f"{path.stem}.kml"
    write_kml(df, out_path, doc_name=path.stem)
    return out_path, df["Class"].value_counts().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="directory containing the label CSVs")
    parser.add_argument("--pattern", default=INPUT_GLOB, help="glob for input CSVs inside --input-dir")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="directory for the KML files")
    args = parser.parse_args()

    csv_paths = sorted(args.input_dir.glob(args.pattern))
    if not csv_paths:
        raise SystemExit(f"No CSVs matching {args.pattern!r} under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    known = set(CLASS_STYLE_RGB)
    for path in csv_paths:
        out_path, counts = convert_csv(path, args.output_dir)
        print(f"{path.name} -> {out_path.name}  ({sum(counts.values())} rows)")
        print(f"  Class counts : {counts}")
        unknown = set(counts) - known
        if unknown:
            print(f"  Note         : unstyled class(es) {sorted(unknown)} "
                  f"import with a default grey style")
    print(f"\nImport in Google Earth Web: Projects -> Import -> Open as KML file")


if __name__ == "__main__":
    main()
