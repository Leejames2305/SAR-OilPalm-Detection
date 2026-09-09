#!/usr/bin/env python3
"""Shared KML writer for label-point datasets (id, Long, Lat, Class).

Generates a KML document with one folder per Class and per-class placemark
styling (coloured circle icons). Points are expected in WGS84 decimal degrees
(EPSG:4326), ready for import into Google Earth Web (Projects -> Import).
"""

from xml.sax.saxutils import escape

import pandas as pd

KML_NS = "http://www.opengis.net/kml/2.2"
ICON_HREF = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"


def _abgr(rgb: str, alpha: str = "ff") -> str:
    """Convert '#rrggbb' (or 'rrggbb') to the KML abgr colour format."""
    rgb = rgb.lstrip("#")
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"{alpha}{b}{g}{r}".lower()


# Class -> KML style (rgb hex). Extend here when a dataset adds a new class.
#   Healthy   = green, Middle = orange, Unhealthy = red
CLASS_STYLE_RGB = {
    "Healthy": {"icon": "44cc22", "fill": "d7f0dd"},
    "Middle": {"icon": "f5a623", "fill": "fdeddd"},
    "Unhealthy": {"icon": "dd2222", "fill": "dde0f5"},
}


def write_kml(df: pd.DataFrame, path, doc_name: str, class_style: dict | None = None) -> None:
    """Write points (columns: id, Long, Lat, Class) to `path` as KML.

    Points with a Class missing from the style map still get a folder, but use
    a default white style and are reported via the returned folder names.
    """
    style_map = class_style or CLASS_STYLE_RGB
    classes = [c for c in df["Class"].dropna().unique()]

    styles = "\n".join(
        f"""
    <Style id="style_{escape(cls)}">
      <IconStyle><color>{_abgr(st['icon'])}</color><scale>0.9</scale>
        <Icon><href>{ICON_HREF}</href></Icon>
      </IconStyle>
      <LabelStyle><scale>0</scale></LabelStyle>
      <BalloonStyle><text><![CDATA[<b>$[name]</b><br/>Class: {escape(cls)}]]></text></BalloonStyle>
    </Style>"""
        for cls in classes
        for st in [style_map.get(cls, {"icon": "888888", "fill": "ffffff"})]
    )

    folders = []
    for cls in classes:
        group = df[df["Class"] == cls]
        st = style_map.get(cls, {"icon": "888888", "fill": "ffffff"})
        placemarks = "\n".join(
            f"""
        <Placemark>
          <name>{escape(str(row['id']))}</name>
          <styleUrl>#style_{escape(cls)}</styleUrl>
          <ExtendedData>
            <Data name="id"><value>{escape(str(row['id']))}</value></Data>
            <Data name="Class"><value>{escape(cls)}</value></Data>
          </ExtendedData>
          <Point><coordinates>{row['Long']:.9f},{row['Lat']:.9f},0</coordinates></Point>
        </Placemark>"""
            for _, row in group.iterrows()
        )
        folders.append(
            f"""
    <Folder>
      <name>{escape(cls)} ({len(group)})</name>
      <open>0</open>{placemarks}
    </Folder>"""
        )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}">
  <Document>
    <name>{escape(doc_name)}</name>{styles}{''.join(folders)}
  </Document>
</kml>
"""
    path.write_text(kml, encoding="utf-8")
