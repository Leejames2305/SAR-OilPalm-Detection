#!/usr/bin/env python3
"""Convert AHCensus label data to the Palong-style label format.

Source (data/Label-status/AHCensus.csv):
    wkt_geom,Status            # WKT points in a projected CRS (UTM 48N, EPSG:32648)
                               # Status: 1 = Healthy, 2 = Unhealthy, 3 = Vacant (dropped)

Target:
    id,Long,Lat,Class          # WGS84 decimal degrees (EPSG:4326)
                               # Class: Healthy / Unhealthy

Usage:
    python convert_ah_census.py                       # defaults below
    python convert_ah_census.py --src-epsg 24548      # e.g. swap to Kertau / UTM 48N
    python convert_ah_census.py --no-kml              # skip the KML export
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from kml_utils import write_kml

# --- Defaults ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "data" / "Label-status" / "AHCensus.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "output" / "AH_Census Classification.csv"

SRC_EPSG = 32648          # WGS84 / UTM zone 48N (AHCensus coords are metres)
DST_EPSG = 4326           # WGS84 geographic (Long/Lat), matches Palong/Serting CSVs
COORD_DECIMALS = 9        # Palong CSV carries ~9 decimal places

# Status -> Class mapping (Status 3 = Vacant is intentionally excluded)
STATUS_TO_CLASS = {
    1: "Healthy",
    2: "Unhealthy",
}

WKT_POINT_RE = re.compile(r"Point\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", re.IGNORECASE)
# The raw file carries a trailing legend ("1= Healthy", ...) in spare columns,
# which usecols=[0, 1] + value coercion discards.


def parse_wkt_points(series: pd.Series) -> list[tuple[float, float]]:
    points = []
    for geom in series:
        match = WKT_POINT_RE.fullmatch(str(geom).strip())
        if not match:
            raise ValueError(f"Unrecognised wkt_geom value: {geom!r}")
        points.append((float(match.group(1)), float(match.group(2))))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="AHCensus CSV path")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Converted CSV path")
    parser.add_argument("--src-epsg", type=int, default=SRC_EPSG, help="EPSG code of the source projected CRS")
    parser.add_argument("--kml", dest="kml", type=Path, default=OUTPUT_CSV.with_suffix(".kml"),
                        help="KML output path for Google Earth Web")
    parser.add_argument("--no-kml", dest="kml", action="store_const", const=None,
                        help="skip the KML export")
    args = parser.parse_args()

    df = pd.read_csv(args.input, usecols=[0, 1])
    df.columns = ["wkt_geom", "Status"]
    df["Status"] = pd.to_numeric(df["Status"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Status"])
    df["Status"] = df["Status"].astype(int)

    n_total = len(df)
    n_vacant = int((df["Status"] == 3).sum())
    n_unknown = int((~df["Status"].isin([*STATUS_TO_CLASS, 3])).sum())
    df = df[df["Status"].isin(STATUS_TO_CLASS)].reset_index(drop=True)

    src_pts = parse_wkt_points(df["wkt_geom"])
    xs = [p[0] for p in src_pts]
    ys = [p[1] for p in src_pts]
    lon, lat = warp_transform(CRS.from_epsg(args.src_epsg), CRS.from_epsg(DST_EPSG), xs, ys)

    out = pd.DataFrame(
        {
            "id": range(1, len(df) + 1),
            "Long": [round(v, COORD_DECIMALS) for v in lon],
            "Lat": [round(v, COORD_DECIMALS) for v in lat],
            "Class": df["Status"].map(STATUS_TO_CLASS),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    if args.kml is not None:
        args.kml.parent.mkdir(parents=True, exist_ok=True)
        write_kml(out, args.kml, "AHCensus Labels")

    print(f"Source          : {args.input}  (EPSG:{args.src_epsg} -> EPSG:{DST_EPSG})")
    print(f"Rows read       : {n_total}")
    print(f"Vacant dropped  : {n_vacant}")
    if n_unknown:
        print(f"Unknown status  : {n_unknown} (dropped, unrecognised Status value)")
    print(f"Class counts    : {out['Class'].value_counts().to_dict()}")
    print(f"Long range      : {out['Long'].min():.6f} .. {out['Long'].max():.6f}")
    print(f"Lat range       : {out['Lat'].min():.6f} .. {out['Lat'].max():.6f}")
    print(f"Written         : {args.output}  ({len(out)} rows)")
    if args.kml is not None:
        print(f"KML (Earth Web) : {args.kml}")


if __name__ == "__main__":
    main()
