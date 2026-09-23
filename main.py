import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    # Global imports available to every cell. Native and optional
    # dependencies (rasterio, sklearn, xgboost, torch, google-cloud-storage)
    # are imported lazily inside their own sections so Section 1 can run
    # even where those packages are not installed.
    import marimo as mo
    import glob
    import importlib.util
    import io
    import json
    import os
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Oil Palm Disease Classification using ALOS-2 SAR

    Reactive Marimo port of the original `main.ipynb`. Runs on **Molab**
    and locally via the VSCode Marimo extension.

    **Pipeline map**

    - **Section 0** (below): global configuration. Every control is reactive:
      change a value and all dependent cells update.
    - **Section 1**: environment, data inventory, GCS sync via SA-key upload.
    - **Section 2**: CRS-aware SAR sampling per location, with visual QA.
    - **Section 3**: per-location datasets, GSMOTE toggle, spatial-block CV.
    - **Section 4**: classifier registry (RF, XGBoost, TabFM) and shared
      evaluation harness.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Section 0 — Global configuration

    Single source of truth for the whole notebook. Downstream cells read these
    controls, so dragging a slider re-runs sampling, QA, splits and results.
    Expensive steps additionally sit behind run buttons so nothing heavy fires
    on an accidental drag.
    """)
    return


@app.cell
def _():
    loc_select = mo.ui.multiselect(
        options=["Census", "Serting", "Palong"],
        value=["Census", "Serting", "Palong"],
        label="Locations (each location is modelled independently)",
    )
    window_slider = mo.ui.slider(
        start=1, stop=7, step=2, value=3,
        label="Sampling window N x N (default 3x3)",
    )
    stats_select = mo.ui.multiselect(
        options=["mean", "std", "min", "max", "p25", "p50"],
        value=["mean", "std", "min", "max", "p25", "p50"],
        label="Window statistics (POC finding: max, min and percentiles carry the signal)",
    )
    test_size_slider = mo.ui.slider(
        start=10, stop=40, step=5, value=20,
        label="Holdout test size (percent)",
    )
    seed_number = mo.ui.number(
        value=42, label="Random seed"
    )
    gsmote_toggle = mo.ui.checkbox(
        value=True, label="Apply GSMOTE inside training folds (toggle, default on)"
    )
    clf_select = mo.ui.multiselect(
        options=["RandomForest", "XGBoost", "TabFM"],
        value=["RandomForest", "XGBoost", "TabFM"],
        label="Classifiers (registry lives in Section 4: add one entry to plug in a new model)",
    )
    mo.vstack(
        [
            loc_select,
            window_slider,
            stats_select,
            test_size_slider,
            seed_number,
            gsmote_toggle,
            clf_select,
        ]
    )
    return loc_select, seed_number, test_size_slider, window_slider


@app.cell
def _(loc_select, seed_number, test_size_slider, window_slider):
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"
    SCENE_DIR = DATA_DIR / "SAR-scenes"
    LABEL_DIR = DATA_DIR / "Labels"
    PROCESSED_DIR = DATA_DIR / "Processed"
    for _d in (SCENE_DIR, LABEL_DIR, PROCESSED_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # Per-location file mapping. The scene prefix is shared by all products.
    LOCATION_META = {
        "Census": {
            "year": 2026,
            "label": "Census-Classification_2026.csv",
            "prefix": "ALOS2-Subset_Census_260610_Cal_ML_Spk",
        },
        "Serting": {
            "year": 2022,
            "label": "Serting-Classification_2022.csv",
            "prefix": "ALOS2-Subset_Serting_260610_Cal_ML_Spk",
        },
        "Palong": {
            "year": 2022,
            "label": "Palong-Classification_2022.csv",
            "prefix": "ALOS2-Subset_Palong_260610_Cal_ML_Spk",
        },
    }
    SCENE_PRODUCTS = {
        "intensity": "_TC.tif",
        "halpha": "_HAlphaDecomp_TC.tif",
        "yama": "_YamaDecomp_TC.tif",
    }

    # Positional band order: subset GeoTIFFs carry no band names, so the order
    # below is authoritative. Intensity was confirmed as [HH, HV, VH, VV].
    INTENSITY_ORDER = ["HH", "HV", "VH", "VV"]
    HALPHA_ORDER = ["H", "A", "alpha"]
    YAMA_ORDER = ["Pd", "Pv", "Ps", "Pc"]

    # Yama scenes are dB by SNAP export default and are kept as is.
    YAMA_IN_DB = True
    # Intensity scenes are EPSG:32648 while decomps are EPSG:4326, so every
    # sampler must reproject tree coordinates per scene.
    CRS_AWARE_SAMPLING = True

    RANDOM_STATE = int(seed_number.value)
    TEST_SIZE = float(test_size_slider.value) / 100.0
    WINDOW = int(window_slider.value)
    ACTIVE_LOCATIONS = [loc for loc in loc_select.value if loc in LOCATION_META]
    return LABEL_DIR, LOCATION_META, SCENE_DIR, SCENE_PRODUCTS


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Section 1 — Environment, data inventory, GCS sync

    Local-first: if `data/SAR-scenes/` and `data/Labels/` are present (the
    normal case for a repo checkout), nothing needs downloading and the sync
    cells below stay idle. On a fresh runtime such as Molab, upload the
    service-account key and press sync to pull the missing files from GCS.
    """)
    return


@app.cell
def _(LABEL_DIR, LOCATION_META, SCENE_DIR, SCENE_PRODUCTS):
    _scene_rows = []
    for _loc, _meta in LOCATION_META.items():
        for _product, _suffix in SCENE_PRODUCTS.items():
            _name = _meta["prefix"] + _suffix
            _path = SCENE_DIR / _name
            _scene_rows.append(
                {
                    "location": _loc,
                    "product": _product,
                    "file": _name,
                    "found": bool(_path.exists()),
                    "size_mb": round(_path.stat().st_size / 1048576, 2)
                    if _path.exists()
                    else 0.0,
                }
            )
    scene_inventory = pd.DataFrame(_scene_rows)

    _label_rows = []
    for _loc, _meta in LOCATION_META.items():
        _path = LABEL_DIR / _meta["label"]
        if _path.exists():
            _df = pd.read_csv(_path)
            _counts = _df["Class"].value_counts().to_dict()
            _total = int(len(_df))
            _unhealthy = int(_counts.get("Unhealthy", 0))
            _label_rows.append(
                {
                    "location": _loc,
                    "year": _meta["year"],
                    "trees": _total,
                    "healthy": int(_counts.get("Healthy", 0)),
                    "middle": int(_counts.get("Middle", 0)),
                    "unhealthy": _unhealthy,
                    "pos_rate": round(_unhealthy / _total, 4) if _total else 0.0,
                }
            )
        else:
            _label_rows.append(
                {
                    "location": _loc,
                    "year": _meta["year"],
                    "trees": 0,
                    "healthy": 0,
                    "middle": 0,
                    "unhealthy": 0,
                    "pos_rate": 0.0,
                }
            )
    label_balance = pd.DataFrame(_label_rows)
    return label_balance, scene_inventory


@app.cell
def _(label_balance, scene_inventory):
    mo.vstack(
        [
            mo.md("**SAR scenes on disk**"),
            mo.ui.table(scene_inventory),
            mo.md("**Label balance on disk** (pos_rate is the no-skill PR-AUC baseline)"),
            mo.ui.table(label_balance),
        ]
    )
    return


@app.cell
def _():
    sa_key_file = mo.ui.file(
        label="Service-account JSON key (only needed when files are missing locally)",
    )
    bucket_text = mo.ui.text(
        value="sar-oilpalm", label="GCS bucket name"
    )
    sync_button = mo.ui.run_button(label="Sync missing files from GCS")
    mo.vstack([sa_key_file, bucket_text, sync_button])
    return bucket_text, sa_key_file, sync_button


@app.cell
def _(
    LABEL_DIR,
    LOCATION_META,
    SCENE_DIR,
    SCENE_PRODUCTS,
    bucket_text,
    sa_key_file,
    sync_button,
):
    mo.stop(
        not sync_button.value,
        mo.md("Press **Sync missing files from GCS** to run this cell."),
    )
    _uploads = sa_key_file.value or []
    mo.stop(
        len(_uploads) == 0,
        mo.md("Upload the service-account JSON key above, then press sync again."),
    )

    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        mo.stop(
            True,
            mo.md(
                "The `google-cloud-storage` package is not installed. "
                "Install it with `uv pip install google-cloud-storage` and re-run."
            ),
        )

    def _upload_bytes(_item):
        if isinstance(_item, (list, tuple)) and len(_item) == 2:
            return str(_item[0]), bytes(_item[1])
        _name = getattr(_item, "name", "sa-key.json")
        _contents = getattr(_item, "contents", None)
        if _contents is None and hasattr(_item, "read"):
            _contents = _item.read()
        if isinstance(_contents, str):
            _contents = _contents.encode("utf-8")
        return str(_name), bytes(_contents)

    _key_name, _key_bytes = _upload_bytes(_uploads[0])
    _info = json.loads(_key_bytes.decode("utf-8"))
    _creds = service_account.Credentials.from_service_account_info(_info)
    _client = storage.Client(credentials=_creds, project=_info.get("project_id"))
    _bucket = _client.bucket(bucket_text.value.strip())

    _wanted = []
    for _loc, _meta in LOCATION_META.items():
        for _product, _suffix in SCENE_PRODUCTS.items():
            _name = _meta["prefix"] + _suffix
            _wanted.append(
                ("data/SAR-scenes/" + _name, SCENE_DIR / _name)
            )
        _wanted.append(
            ("data/Labels/" + _meta["label"], LABEL_DIR / _meta["label"])
        )

    _report = []
    for _blob_name, _local in _wanted:
        if _local.exists():
            _report.append({"file": _local.name, "action": "already on disk"})
            continue
        _local.parent.mkdir(parents=True, exist_ok=True)
        _bucket.blob(_blob_name).download_to_filename(str(_local))
        _report.append({"file": _local.name, "action": "downloaded from GCS"})
    sync_report = pd.DataFrame(_report)
    mo.ui.table(sync_report)
    return


@app.cell
def _():
    _candidates = [
        "rasterio",
        "geopandas",
        "sklearn",
        "xgboost",
        "torch",
        "tabfm",
        "google.cloud.storage",
        "imbalanced_learn",
    ]
    deps_status = pd.DataFrame(
        [
            {
                "package": _name,
                "available": importlib.util.find_spec(_name.split(chr(46))[0]) is not None,
            }
            for _name in _candidates
        ]
    )
    return (deps_status,)


@app.cell
def _(deps_status):
    mo.vstack(
        [
            mo.md("**Optional dependencies** (each section guards its own imports)"),
            mo.ui.table(deps_status),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Scale, CRS and band-order notes (authoritative for Sections 2 to 4)**

    - Intensity scenes are **linear power**. Yama decomposition scenes are
      **dB by SNAP export default** and are kept as is.
    - Intensity scenes are `EPSG:32648` (UTM 48N); decomposition scenes are
      `EPSG:4326`. Section 2 reprojects tree coordinates per scene.
    - Subset GeoTIFFs carry no band names. Positional order used throughout:
      intensity `[HH, HV, VH, VV]`, HAlpha `[H, A, alpha]`,
      Yama `[Pd, Pv, Ps, Pc]`.
    - Notebook outputs are kept viewable in the repo: refresh the snapshot
      with `marimo export html main.py -o __marimo__/main.html` after a run.
    """)
    return


if __name__ == "__main__":
    app.run()
