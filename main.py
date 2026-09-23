import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    # Global imports available to every cell. A bootstrap step installs any
    # missing core package with uv (pip fallback) before the imports below
    # run, so fresh runtimes such as a Molab server work out of the box.
    # Heavy ML packages (torch, tabfm) install lazily in Section 4 instead.
    import marimo as mo
    import glob
    import importlib.util
    import io
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    BOOTSTRAP_PACKAGES = {
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scipy": "scipy",
        "rasterio": "rasterio",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "imblearn": "imbalanced-learn",
        "google.cloud.storage": "google-cloud-storage",
    }

    def _ensure_packages(packages):
        missing = [k for k in packages if importlib.util.find_spec(k.split(".")[0]) is None]
        if not missing:
            print("[bootstrap] all required packages present.")
            return []
        specs = [packages[k] for k in missing]
        print("[bootstrap] installing missing packages: " + ", ".join(specs))
        if shutil.which("uv") is not None:
            cmd = ["uv", "pip", "install", "--system"] + specs
        else:
            print("[bootstrap] uv not found, falling back to pip.")
            cmd = [sys.executable, "-m", "pip", "install"] + specs
        subprocess.check_call(cmd)
        still = [k for k in missing if importlib.util.find_spec(k.split(".")[0]) is None]
        if still:
            print("[bootstrap] WARNING, still missing after install: " + ", ".join(still))
        else:
            print("[bootstrap] install complete.")
        return still

    BOOTSTRAP_MISSING = _ensure_packages(BOOTSTRAP_PACKAGES)

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


@app.cell(hide_code=True)
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
    return (
        clf_select,
        gsmote_toggle,
        loc_select,
        seed_number,
        stats_select,
        test_size_slider,
        window_slider,
    )


@app.cell(hide_code=True)
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
    return (
        ACTIVE_LOCATIONS,
        HALPHA_ORDER,
        INTENSITY_ORDER,
        LABEL_DIR,
        LOCATION_META,
        PROCESSED_DIR,
        RANDOM_STATE,
        SCENE_DIR,
        SCENE_PRODUCTS,
        TEST_SIZE,
        WINDOW,
        YAMA_ORDER,
    )


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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Section 2 — SAR sampling and visual QA

    For every tree in each location label CSV, a window (default 3x3,
    configurable in Section 0) is sampled from all three scene products:

    - **Intensity** (`EPSG:32648`): tree lon/lat is reprojected per scene, then
      the selected window statistics are computed over **linear power**.
      Only positive finite pixels count as valid.
    - **HAlpha / Yama** (`EPSG:4326`): sampled in lon/lat directly as
      window mean and std. Yama scenes are **dB by SNAP export default** and
      are kept as is, so negative values are valid there.
    - `Middle` trees are dropped. Each location is written to its own CSV in
      `data/Processed/`, named by window and statistic set. Existing CSVs are
      reused unless forced.
    """)
    return


@app.cell(hide_code=True)
def _(
    HALPHA_ORDER,
    INTENSITY_ORDER,
    LABEL_DIR,
    LOCATION_META,
    PROCESSED_DIR,
    SCENE_DIR,
    SCENE_PRODUCTS,
    YAMA_ORDER,
):
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
        from rasterio.windows import Window
    except ImportError:
        mo.stop(
            True,
            mo.md("The `rasterio` package is missing and the bootstrap install failed. Install it manually, then re-run."),
        )

    STAT_FUNCS = {
        "mean": lambda a: float(np.mean(a)),
        "std": lambda a: float(np.std(a)),
        "min": lambda a: float(np.min(a)),
        "max": lambda a: float(np.max(a)),
        "p25": lambda a: float(np.percentile(a, 25)),
        "p50": lambda a: float(np.percentile(a, 50)),
    }

    def _read_window(src, band_idx, row, col, window):
        half = window // 2
        win = Window(max(0, col - half), max(0, row - half), window, window)
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width == 0 or win.height == 0:
            return np.array([], dtype=float)
        arr = src.read(band_idx + 1, window=win).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr[np.isfinite(arr)]

    def _dataset_tag(window, stats):
        return "w" + str(window) + "_" + "_".join(stats)

    def dataset_path_for(loc, window, stats):
        return PROCESSED_DIR / ("sampled_" + loc + "_" + _dataset_tag(window, stats) + ".csv")

    def sample_location(loc, window, stats):
        meta = LOCATION_META[loc]
        label_path = LABEL_DIR / meta["label"]
        if not label_path.exists():
            raise FileNotFoundError("Label file not found: " + str(label_path) + ". Run the Section 1 sync first.")
        labels = pd.read_csv(label_path)
        labels = labels[labels["Class"].isin(["Healthy", "Unhealthy"])].copy()
        labels["y"] = (labels["Class"] == "Unhealthy").astype(int)
        n = len(labels)
        lons = labels["Long"].to_numpy(dtype=float)
        lats = labels["Lat"].to_numpy(dtype=float)
        out = labels[["id", "Long", "Lat", "Class", "y"]].copy()

        prefix = meta["prefix"]
        paths = {}
        for product in SCENE_PRODUCTS:
            paths[product] = SCENE_DIR / (prefix + SCENE_PRODUCTS[product])
        for product in paths:
            if not paths[product].exists():
                raise FileNotFoundError("Scene not found: " + str(paths[product]) + ". Run the Section 1 sync first.")

        with rasterio.open(paths["intensity"]) as src:
            if src.count != len(INTENSITY_ORDER):
                raise ValueError("Intensity band count is " + str(src.count) + ", expected " + str(len(INTENSITY_ORDER)) + ". Check the positional band order.")
            xs, ys = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
            for j in range(len(INTENSITY_ORDER)):
                pol = INTENSITY_ORDER[j]
                cols = {}
                for s in stats:
                    cols[s] = np.full(n, np.nan)
                for i in range(n):
                    row, col = src.index(xs[i], ys[i])
                    pix = _read_window(src, j, row, col, window)
                    pix = pix[pix > 0]
                    if pix.size == 0:
                        continue
                    for s in stats:
                        cols[s][i] = STAT_FUNCS[s](pix)
                for s in stats:
                    out[pol + "_" + s] = cols[s]

        prods = (("halpha", HALPHA_ORDER), ("yama", YAMA_ORDER))
        for prod in prods:
            pname = prod[0]
            order = prod[1]
            with rasterio.open(paths[pname]) as src:
                if src.count != len(order):
                    raise ValueError("Decomp band count is " + str(src.count) + ", expected " + str(len(order)) + ". Check the positional band order.")
                for j in range(len(order)):
                    feat = order[j]
                    mu = np.full(n, np.nan)
                    sd = np.full(n, np.nan)
                    for i in range(n):
                        row, col = src.index(lons[i], lats[i])
                        pix = _read_window(src, j, row, col, window)
                        if pix.size == 0:
                            continue
                        mu[i] = float(np.mean(pix))
                        sd[i] = float(np.std(pix))
                    out[feat + "_mean"] = mu
                    out[feat + "_std"] = sd
        return out

    return dataset_path_for, sample_location


@app.cell(hide_code=True)
def _():
    sample_button = mo.ui.run_button(label="Run sampling (per location, cached to CSV)")
    force_checkbox = mo.ui.checkbox(value=False, label="Force resample even when the CSV already exists")
    mo.vstack([sample_button, force_checkbox])
    return force_checkbox, sample_button


@app.cell(hide_code=True)
def _(
    ACTIVE_LOCATIONS,
    WINDOW,
    dataset_path_for,
    force_checkbox,
    sample_button,
    sample_location,
    stats_select,
):
    mo.stop(not sample_button.value, mo.md("Press **Run sampling** to build the per-location datasets."))
    mo.stop(len(ACTIVE_LOCATIONS) == 0, mo.md("Select at least one location in Section 0."))
    stats = [s for s in ["mean", "std", "min", "max", "p25", "p50"] if s in list(stats_select.value)]
    mo.stop(len(stats) == 0, mo.md("Select at least one window statistic in Section 0."))
    sampled_paths = {}
    rows = []
    for loc in ACTIVE_LOCATIONS:
        path = dataset_path_for(loc, WINDOW, stats)
        if path.exists() and not force_checkbox.value:
            df = pd.read_csv(path)
            action = "loaded from CSV"
        else:
            df = sample_location(loc, WINDOW, stats)
            df.to_csv(path, index=False)
            action = "sampled and saved"
        sampled_paths[loc] = str(path)
        rows.append(
            {
                "location": loc,
                "trees": len(df),
                "unhealthy": int((df["y"] == 1).sum()),
                "features": len(df.columns) - 5,
                "action": action,
                "file": path.name,
            }
        )
    sampling_summary = pd.DataFrame(rows)
    mo.ui.table(sampling_summary)
    return (sampled_paths,)


@app.cell(hide_code=True)
def _(ACTIVE_LOCATIONS, sampled_paths):
    mo.stop(len(ACTIVE_LOCATIONS) == 0, mo.md("Select at least one location in Section 0."))
    tabs = {}
    for loc in ACTIVE_LOCATIONS:
        if loc not in sampled_paths:
            continue
        df = pd.read_csv(sampled_paths[loc])
        feat_cols = [c for c in df.columns if c not in ("id", "Long", "Lat", "Class", "y")]
        h = df[df["y"] == 0]
        u = df[df["y"] == 1]
        miss = pd.DataFrame(
            {
                "feature": feat_cols,
                "missing_rate": [round(float(df[c].isna().mean()), 4) for c in feat_cols],
                "mean_healthy": [round(float(h[c].mean()), 5) for c in feat_cols],
                "mean_unhealthy": [round(float(u[c].mean()), 5) for c in feat_cols],
            }
        )
        tabs[loc] = mo.ui.table(miss)
    mo.stop(len(tabs) == 0, mo.md("Run the sampling step above first."))
    mo.ui.tabs(tabs)
    return


@app.cell(hide_code=True)
def _(ACTIVE_LOCATIONS, sampled_paths):
    mo.stop(len(ACTIVE_LOCATIONS) == 0, mo.md("Select at least one location in Section 0."))
    import matplotlib.pyplot as plt

    locs = [loc for loc in ACTIVE_LOCATIONS if loc in sampled_paths]
    mo.stop(len(locs) == 0, mo.md("Run the sampling step above first."))
    fig, axes = plt.subplots(len(locs), 2, figsize=(12, 4 * len(locs)), squeeze=False)
    for r in range(len(locs)):
        loc = locs[r]
        df = pd.read_csv(sampled_paths[loc])
        core = [c for c in ["HH_mean", "HV_mean", "VH_mean", "VV_mean"] if c in df.columns]
        ax = axes[r][0]
        if core:
            h_vals = df[df["y"] == 0][core].to_numpy(dtype=float).ravel()
            u_vals = df[df["y"] == 1][core].to_numpy(dtype=float).ravel()
            h_vals = h_vals[np.isfinite(h_vals)]
            u_vals = u_vals[np.isfinite(u_vals)]
            ax.boxplot([h_vals, u_vals], labels=["Healthy", "Unhealthy"])
            ax.set_yscale("log")
        ax.set_title(loc + " core backscatter means (log scale)")
        ax2 = axes[r][1]
        sc = ax2.scatter(df["Long"], df["Lat"], c=df["y"], cmap="RdYlGn_r", s=8)
        fig.colorbar(sc, ax=ax2)
        ax2.set_title(loc + " tree map (1 is Unhealthy)")
        ax2.set_xlabel("Long")
        ax2.set_ylabel("Lat")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Section 3 — Datasets, GSMOTE, splits and evaluation harness

    - Each location sampled CSV becomes `X` (all sampled feature columns)
      and `y` (`Unhealthy` is 1). `Middle` was already dropped at sampling.
    - **GSMOTE** (paper defaults: `k=5`, truncation `1.0`, deformation `0.0`,
      `combined`) is applied **inside training folds only**, on standardised
      features. Geometry is verbatim from the POC report. There is no sweep:
      the POC showed a flat surface.
    - Default protocol is the honest one: stratified holdout plus
      `StratifiedGroupKFold` over KMeans spatial blocks (low-positive blocks
      merged into their nearest sufficiently-positive neighbour).
    - Median imputation plus standardisation happen inside the harness on
      train-fit statistics only. The feature set is now compact (38 curated
      statistics), so the old dedup and MI-select stages were retired: the
      POC showed selection only ties within noise under honest CV.
    """)
    return


@app.cell(hide_code=True)
def _():
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils import check_random_state

    GSMOTE_DEFAULTS = {
        "k_neighbors": 5,
        "truncation_factor": 1.0,
        "deformation_factor": 0.0,
        "selection_strategy": "combined",
    }

    def make_geometric_sample(center, surface_point, truncation_factor, deformation_factor, random_state):
        if np.array_equal(center, surface_point):
            return center
        radius = np.linalg.norm(center - surface_point)
        normal_samples = random_state.normal(size=center.size)
        point_on_unit_sphere = normal_samples / np.linalg.norm(normal_samples)
        point = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere
        parallel_unit_vector = (surface_point - center) / np.linalg.norm(surface_point - center)
        if truncation_factor > 0 and np.dot(point, parallel_unit_vector) < truncation_factor - 1:
            point = point - 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector
        if truncation_factor < 0 and np.dot(point, parallel_unit_vector) > truncation_factor + 1:
            point = point - 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector
        parallel_point_position = np.dot(point, parallel_unit_vector) * parallel_unit_vector
        perpendicular_point_position = point - parallel_point_position
        point = parallel_point_position + (1 - deformation_factor) * perpendicular_point_position
        return center + radius * point

    class GeometricSMOTE:
        def __init__(self, sampling_strategy="auto", k_neighbors=5, truncation_factor=1.0, deformation_factor=0.0, selection_strategy="combined", random_state=None):
            if selection_strategy not in ("combined", "majority", "minority"):
                raise ValueError("selection_strategy must be combined, majority or minority")
            if not -1.0 <= truncation_factor <= 1.0:
                raise ValueError("truncation_factor must be in [-1.0, 1.0]")
            if not 0.0 <= deformation_factor <= 1.0:
                raise ValueError("deformation_factor must be in [0.0, 1.0]")
            self.sampling_strategy = sampling_strategy
            self.k_neighbors = k_neighbors
            self.truncation_factor = truncation_factor
            self.deformation_factor = deformation_factor
            self.selection_strategy = selection_strategy
            self.random_state = random_state

        def _targets(self, classes, counts):
            n_majority = max(counts.values())
            strat = self.sampling_strategy
            if isinstance(strat, str):
                if strat == "auto":
                    strat = "not majority"
                if strat == "minority":
                    return {min(counts, key=counts.get): n_majority}
                if strat in ("not minority", "not majority", "all"):
                    t = {c: n_majority for c in classes}
                    if strat == "not minority":
                        t.pop(min(counts, key=counts.get), None)
                    elif strat == "not majority":
                        t.pop(max(counts, key=counts.get), None)
                    return t
                raise ValueError("unsupported sampling_strategy " + str(strat))
            if isinstance(strat, dict):
                return {c: int(v) for c, v in strat.items()}
            raise TypeError("sampling_strategy must be a str or a dict")

        def fit_resample(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            rng = check_random_state(self.random_state)
            classes, counts_arr = np.unique(y, return_counts=True)
            counts = {c: int(n) for c, n in zip(classes, counts_arr)}
            targets = self._targets(classes, counts)
            X_res, y_res = X.copy(), y.copy()
            for cls in sorted(targets, key=str):
                n_new = int(targets[cls]) - counts.get(cls, 0)
                if n_new <= 0:
                    continue
                X_pos = X[y == cls]
                if len(X_pos) < 2:
                    X_new = X_pos[rng.randint(0, len(X_pos), size=n_new)].copy()
                else:
                    strategy = "minority" if len(X_pos) == len(X) else self.selection_strategy
                    k_eff = int(max(1, min(int(self.k_neighbors), len(X_pos) - 1)))
                    if strategy in ("minority", "combined"):
                        nns_pos = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_pos)
                        points_pos = nns_pos.kneighbors(X_pos, return_distance=False)[:, 1:]
                        idx = rng.randint(0, points_pos.size, size=n_new)
                        rows = np.floor_divide(idx, points_pos.shape[1])
                        cols = np.mod(idx, points_pos.shape[1])
                    if strategy in ("majority", "combined"):
                        X_neg = X[y != cls]
                        nn_neg = NearestNeighbors(n_neighbors=1).fit(X_neg)
                        points_neg = nn_neg.kneighbors(X_pos, return_distance=False)
                        if strategy == "majority":
                            idx = rng.randint(0, points_neg.size, size=n_new)
                            rows = np.floor_divide(idx, points_neg.shape[1])
                            cols = np.mod(idx, points_neg.shape[1])
                    X_new = np.zeros((n_new, X.shape[1]), dtype=float)
                    for ind in range(n_new):
                        row, col = int(rows[ind]), int(cols[ind])
                        center = X_pos[row]
                        if strategy == "minority":
                            surface_point = X_pos[points_pos[row, col]]
                        elif strategy == "majority":
                            surface_point = X_neg[points_neg[row, col]]
                        else:
                            surface_point_pos = X_pos[points_pos[row, col]]
                            surface_point_neg = X_neg[points_neg[row, 0]]
                            radius_pos = np.linalg.norm(center - surface_point_pos)
                            radius_neg = np.linalg.norm(center - surface_point_neg)
                            if radius_pos > radius_neg:
                                surface_point = surface_point_neg
                            else:
                                surface_point = surface_point_pos
                        X_new[ind] = make_geometric_sample(center, surface_point, self.truncation_factor, self.deformation_factor, rng)
                X_res = np.vstack((X_res, X_new))
                y_res = np.hstack((y_res, np.array([cls] * n_new)))
            return X_res, y_res

    return GSMOTE_DEFAULTS, GeometricSMOTE


@app.cell(hide_code=True)
def _(ACTIVE_LOCATIONS, WINDOW, dataset_path_for, stats_select):
    mo.stop(len(ACTIVE_LOCATIONS) == 0, mo.md("Select at least one location in Section 0."))
    stats = [s for s in ["mean", "std", "min", "max", "p25", "p50"] if s in list(stats_select.value)]
    mo.stop(len(stats) == 0, mo.md("Select at least one window statistic in Section 0."))
    X_LOC = {}
    y_LOC = {}
    coords_LOC = {}
    FEATURE_COLS = None
    rows = []
    for loc in ACTIVE_LOCATIONS:
        path = dataset_path_for(loc, WINDOW, stats)
        mo.stop(not path.exists(), mo.md("Dataset missing for **" + loc + "**: run the Section 2 sampling step first."))
        df = pd.read_csv(path)
        feats = [c for c in df.columns if c not in ("id", "Long", "Lat", "Class", "y")]
        if FEATURE_COLS is None:
            FEATURE_COLS = feats
        X_LOC[loc] = df[feats].copy()
        y_LOC[loc] = df["y"].astype(int).copy()
        coords_LOC[loc] = df[["Long", "Lat"]].copy()
        rows.append({"location": loc, "trees": len(df), "unhealthy": int((df["y"] == 1).sum()), "features": len(feats)})
    mo.ui.table(pd.DataFrame(rows))
    return X_LOC, coords_LOC, y_LOC


@app.cell(hide_code=True)
def _(ACTIVE_LOCATIONS, RANDOM_STATE, TEST_SIZE, X_LOC, coords_LOC, y_LOC):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    N_SPATIAL_BLOCKS = 6
    SPATIAL_MIN_POS = 10
    N_SPATIAL_FOLDS = 3

    SPLITS = {}
    SPATIAL_FOLDS = {}
    rows = []
    for loc in ACTIVE_LOCATIONS:
        X = X_LOC[loc]
        y = y_LOC[loc]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        SPLITS[loc] = {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte}
        coords = coords_LOC[loc].to_numpy(dtype=float)
        yv = y.to_numpy()
        km = KMeans(n_clusters=N_SPATIAL_BLOCKS, random_state=RANDOM_STATE, n_init=10)
        raw = km.fit_predict(coords)
        n_blk = int(raw.max()) + 1
        pos = {b: int((yv[raw == b] == 1).sum()) for b in range(n_blk)}
        cents = {b: coords[raw == b].mean(axis=0) for b in range(n_blk)}
        parent = {b: b for b in range(n_blk)}
        for b in range(n_blk):
            if pos[b] >= SPATIAL_MIN_POS:
                continue
            best = None
            bd = float("inf")
            for b2 in range(n_blk):
                if b2 == b or pos[b2] < SPATIAL_MIN_POS:
                    continue
                d = float(np.linalg.norm(cents[b] - cents[b2]))
                if d < bd:
                    best, bd = b2, d
            if best is not None:
                parent[b] = best
        merged = np.array([parent[r] for r in raw])
        n_unique = len(np.unique(merged))
        sgkf = StratifiedGroupKFold(n_splits=max(2, min(N_SPATIAL_FOLDS, n_unique)), shuffle=True, random_state=RANDOM_STATE)
        SPATIAL_FOLDS[loc] = (sgkf, merged)
        rows.append({"location": loc, "train": len(Xtr), "test": len(Xte), "spatial_blocks": n_unique, "spatial_folds": sgkf.n_splits})
    mo.ui.table(pd.DataFrame(rows))
    return SPATIAL_FOLDS, SPLITS


@app.cell(hide_code=True)
def _(GSMOTE_DEFAULTS, GeometricSMOTE):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    def compute_metrics(y_true, y_pred, y_proba, train_time=0.0, pred_time=0.0, threshold=0.5):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        if np.ndim(y_proba) == 1:
            pos_proba = np.asarray(y_proba, dtype=float)
        else:
            pos_proba = np.asarray(y_proba, dtype=float)[:, 1]
        yt = np.asarray(y_true)
        return {
            "accuracy": float(accuracy_score(yt, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, y_pred)),
            "precision_unhealthy": float(precision_score(yt, y_pred, zero_division=0)),
            "recall_unhealthy": float(recall_score(yt, y_pred, zero_division=0)),
            "f1_unhealthy": float(f1_score(yt, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(yt, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, pos_proba)),
            "pr_auc": float(average_precision_score(yt, pos_proba)),
            "brier_score": float(brier_score_loss(yt, pos_proba)),
            "cohen_kappa": float(cohen_kappa_score(yt, y_pred)),
            "runtime_train_s": float(train_time),
            "runtime_pred_s": float(pred_time),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "threshold_used": float(threshold),
        }

    def find_best_threshold(y_true, y_proba):
        if np.ndim(y_proba) == 1:
            pos_proba = np.asarray(y_proba, dtype=float)
        else:
            pos_proba = np.asarray(y_proba, dtype=float)[:, 1]
        prec_list, rec_list, thresholds = precision_recall_curve(y_true, pos_proba)
        n = len(thresholds)
        prec = np.asarray(prec_list[:n])
        rec = np.asarray(rec_list[:n])
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        return float(thresholds[best_idx]), float(f1s[best_idx])

    def _prepare_fold(X_tr, y_tr, X_te, use_gsmote, seed):
        med = np.nanmedian(np.asarray(X_tr, dtype=float), axis=0)
        Xtr = np.where(np.isnan(np.asarray(X_tr, dtype=float)), med, np.asarray(X_tr, dtype=float))
        Xte = np.where(np.isnan(np.asarray(X_te, dtype=float)), med, np.asarray(X_te, dtype=float))
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        ytr = np.asarray(y_tr)
        if use_gsmote:
            Xtr_s, ytr = GeometricSMOTE(random_state=seed, **GSMOTE_DEFAULTS).fit_resample(Xtr_s, ytr)
        return Xtr_s, ytr, Xte_s

    def _proba_of(model, X):
        p = model.predict_proba(X)
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            return p
        return p[:, 1]

    def run_holdout(model_factory, X_tr, y_tr, X_te, y_te, use_gsmote, seed):
        Xtr_s, ytr_s, Xte_s = _prepare_fold(X_tr, y_tr, X_te, use_gsmote, seed)
        model = model_factory()
        t0 = time.time()
        model.fit(Xtr_s, ytr_s)
        train_time = time.time() - t0
        train_proba = _proba_of(model, Xtr_s)
        threshold, _ = find_best_threshold(ytr_s, train_proba)
        t0 = time.time()
        test_proba = _proba_of(model, Xte_s)
        pred_time = time.time() - t0
        y_pred = (test_proba >= threshold).astype(int)
        return compute_metrics(np.asarray(y_te), y_pred, test_proba, train_time, pred_time, threshold)

    def run_spatial_cv(model_factory, X, y, splitter, groups, use_gsmote, seed):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y)
        ga = np.asarray(groups)
        out = []
        for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(Xa, ya, groups=ga)):
            m = run_holdout(model_factory, Xa[tr_idx], ya[tr_idx], Xa[te_idx], ya[te_idx], use_gsmote, seed + fold_idx)
            m["fold"] = fold_idx
            out.append(m)
        return pd.DataFrame(out)

    return run_holdout, run_spatial_cv


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Section 4 — Classifier registry and results

    New classifiers plug in through one dictionary (`CLASSIFIERS` below):
    add an entry and it appears in the Section 0 multiselect flow, the ML
    loop, the tables and the plots with no other edits.

    - **RandomForest**: 500 trees, train-count class weights; `tuned` config
      from a small `average_precision` grid search.
    - **XGBoost**: `hist` on CUDA with CPU fallback, `scale_pos_weight`
      from train counts; `tuned` config from a small grid.
    - **TabFM**: zero-shot foundation model (`standard` and `ensemble`).
      Needs a Hugging Face token (env `HF_TOKEN`, the input below, or the
      cached token file) and loads on CUDA when available. `torch` and
      `tabfm` install automatically on first load only.
    - The GSMOTE toggle from Section 0 applies inside every training fold.

    Read spatial-block CV first (honest). Holdout rows are shown only as a
    leakage-ceiling reference.
    """)
    return


@app.cell(hide_code=True)
def _():
    hf_token_text = mo.ui.text(value="", label="Hugging Face token (only needed for TabFM)")
    tabfm_button = mo.ui.run_button(label="Load TabFM model")
    mo.vstack([hf_token_text, tabfm_button])
    return hf_token_text, tabfm_button


@app.cell(hide_code=True)
def _(clf_select, hf_token_text, tabfm_button):
    import gc

    tabfm_model = None
    tabfm_status = "idle"
    want_tabfm = "TabFM" in list(clf_select.value)
    if want_tabfm and tabfm_button.value:
        tabfm_status = "loading"
        try:
            for pkg, spec in (("torch", "torch"), ("tabfm", "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git")):
                if importlib.util.find_spec(pkg) is None:
                    if shutil.which("uv") is not None:
                        subprocess.check_call(["uv", "pip", "install", "--system", spec])
                    else:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
            import torch

            token = os.environ.get("HF_TOKEN", "")
            if not token and hf_token_text.value.strip():
                token = hf_token_text.value.strip()
            token_path = os.path.expanduser("~/.cache/huggingface/token")
            if not token and os.path.exists(token_path):
                with open(token_path) as f:
                    token = f.read().strip()
            if token:
                os.environ["HF_TOKEN"] = token
                try:
                    os.makedirs(os.path.dirname(token_path), exist_ok=True)
                    with open(token_path, "w") as f:
                        f.write(token)
                except Exception:
                    pass
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.85)
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

            tabfm_model = tabfm_v1_0_0.load(model_type="classification", device=device)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tabfm_status = "loaded on " + device
        except Exception as exc:
            tabfm_model = None
            tabfm_status = "failed: " + repr(exc)
    elif want_tabfm:
        tabfm_status = "not loaded (press the load button above)"
    else:
        tabfm_status = "not selected"
    mo.md("**TabFM status:** " + tabfm_status)
    return (tabfm_model,)


@app.cell(hide_code=True)
def _(RANDOM_STATE, tabfm_model, y_LOC):
    from sklearn.ensemble import RandomForestClassifier

    try:
        from xgboost import XGBClassifier
        HAVE_XGB = True
    except ImportError:
        XGBClassifier = None
        HAVE_XGB = False

    def _pos_weight(loc):
        yv = np.asarray(y_LOC[loc])
        neg = int((yv == 0).sum())
        pos = int((yv == 1).sum())
        return neg / max(pos, 1)

    def _rf_build(loc, params=None):
        kw = {
            "n_estimators": 500,
            "class_weight": {0: 1.0, 1: _pos_weight(loc)},
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        if params:
            kw.update(params)
        return RandomForestClassifier(**kw)

    RF_GRID = {
        "n_estimators": [300],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    def _xgb_build(loc, params=None):
        kw = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "scale_pos_weight": _pos_weight(loc),
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        if params:
            kw.update(params)
        try:
            return XGBClassifier(device="cuda", **kw)
        except Exception:
            return XGBClassifier(device="cpu", **kw)

    XGB_GRID = {
        "n_estimators": [300],
        "max_depth": [3, 6],
        "learning_rate": [0.1],
    }

    def _tabfm_standard():
        if tabfm_model is None:
            raise RuntimeError("TabFM model is not loaded. Load it in the TabFM cell first.")
        from tabfm import TabFMClassifier

        return TabFMClassifier(model=tabfm_model, batch_size=1, cache_context=True, maybe_quantize_kv_cache=True, keep_cache_on_device=False, n_estimators=4)

    def _tabfm_ensemble():
        if tabfm_model is None:
            raise RuntimeError("TabFM model is not loaded. Load it in the TabFM cell first.")
        from tabfm import TabFMClassifier

        return TabFMClassifier.ensemble(model=tabfm_model, batch_size=1, cache_context=True, maybe_quantize_kv_cache=True, keep_cache_on_device=False, n_estimators=4)

    CLASSIFIERS = {
        "RandomForest": {"configs": ("default", "tuned"), "grid": RF_GRID, "build": _rf_build, "gs_n_jobs": -1},
        "XGBoost": {"configs": ("default", "tuned"), "grid": XGB_GRID, "build": _xgb_build, "gs_n_jobs": 1},
        "TabFM": {"configs": ("standard", "ensemble"), "builders": {"standard": _tabfm_standard, "ensemble": _tabfm_ensemble}},
    }
    if not HAVE_XGB:
        del CLASSIFIERS["XGBoost"]
    return (CLASSIFIERS,)


@app.cell(hide_code=True)
def _():
    ml_button = mo.ui.run_button(label="Run ML (selected classifiers, all locations)")
    ml_button
    return (ml_button,)


@app.cell(hide_code=True)
def _(
    ACTIVE_LOCATIONS,
    CLASSIFIERS,
    PROCESSED_DIR,
    RANDOM_STATE,
    SPATIAL_FOLDS,
    SPLITS,
    X_LOC,
    clf_select,
    gsmote_toggle,
    ml_button,
    run_holdout,
    run_spatial_cv,
    tabfm_model,
    y_LOC,
):
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    mo.stop(not ml_button.value, mo.md("Press **Run ML** to train and evaluate."))
    mo.stop(len(ACTIVE_LOCATIONS) == 0, mo.md("Select at least one location in Section 0."))
    RESULTS_DIR = PROCESSED_DIR / "results"
    PLOTS_DIR = PROCESSED_DIR / "plots"
    for d in (RESULTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    use_gsmote = bool(gsmote_toggle.value)
    selected = [c for c in ("RandomForest", "XGBoost", "TabFM") if c in list(clf_select.value) and c in CLASSIFIERS]
    mo.stop(len(selected) == 0, mo.md("No runnable classifier selected. Enable one in Section 0 (XGBoost needs the package, TabFM needs its model loaded)."))
    holdout_rows = []
    cv_rows = []
    for loc in ACTIVE_LOCATIONS:
        split = SPLITS[loc]
        Xtr, Xte = split["X_train"], split["X_test"]
        ytr, yte = split["y_train"], split["y_test"]
        med = np.nanmedian(np.asarray(Xtr, dtype=float), axis=0)

        def _fill(A):
            Aa = np.asarray(A, dtype=float)
            return np.where(np.isnan(Aa), med, Aa)

        tune_scaler = StandardScaler().fit(_fill(Xtr))
        Xtr_p = tune_scaler.transform(_fill(Xtr))
        ytr_a = np.asarray(ytr)
        for model_name in selected:
            if model_name == "TabFM" and tabfm_model is None:
                print("[" + loc + " / TabFM] skipped: model not loaded.")
                continue
            spec = CLASSIFIERS[model_name]
            if model_name == "TabFM":
                jobs = [(cfg, spec["builders"][cfg]) for cfg in spec["configs"]]
            else:
                jobs = [("default", lambda spec=spec, loc=loc: spec["build"](loc, None))]
                if spec.get("grid"):
                    gs = GridSearchCV(spec["build"](loc, None), spec["grid"], cv=3, scoring="average_precision", n_jobs=spec["gs_n_jobs"])
                    gs.fit(Xtr_p, ytr_a)
                    print("[" + loc + " / " + model_name + "] tuned params: " + str(gs.best_params_) + " (CV avg_precision " + str(round(float(gs.best_score_), 4)) + ")")
                    best = dict(gs.best_params_)
                    jobs.append(("tuned", lambda spec=spec, loc=loc, best=best: spec["build"](loc, best)))
            for cfg_name, factory in jobs:
                print("[" + loc + " / " + model_name + " / " + cfg_name + "] holdout and spatial CV, GSMOTE=" + str(use_gsmote))
                m = run_holdout(factory, Xtr, ytr, Xte, yte, use_gsmote, RANDOM_STATE)
                m.update({"location": loc, "model": model_name, "config": cfg_name, "eval_type": "holdout", "fold": -1, "gsmote": use_gsmote})
                holdout_rows.append(m)
                splitter, groups = SPATIAL_FOLDS[loc]
                cv_df = run_spatial_cv(factory, X_LOC[loc], y_LOC[loc], splitter, groups, use_gsmote, RANDOM_STATE)
                for _, r in cv_df.iterrows():
                    d = dict(r)
                    d.update({"location": loc, "model": model_name, "config": cfg_name, "eval_type": "spatial_cv", "gsmote": use_gsmote})
                    cv_rows.append(d)
                try:
                    import gc as _gc

                    _gc.collect()
                    try:
                        import torch as _torch

                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except ImportError:
                        pass
                except Exception:
                    pass
    results_df = pd.DataFrame(holdout_rows + cv_rows)
    for loc in ACTIVE_LOCATIONS:
        sub = results_df[results_df["location"] == loc]
        if len(sub):
            sub.to_csv(RESULTS_DIR / ("results_" + loc + ".csv"), index=False)
    print("Saved per-location results to " + str(RESULTS_DIR) + ", GSMOTE=" + str(use_gsmote))
    mo.ui.table(results_df[["location", "model", "config", "eval_type", "pr_auc", "f1_unhealthy", "recall_unhealthy", "roc_auc"]])
    return PLOTS_DIR, results_df


@app.cell(hide_code=True)
def _(results_df, y_LOC):
    DISPLAY_COLS = ["pr_auc", "f1_unhealthy", "recall_unhealthy", "precision_unhealthy", "roc_auc", "f1_macro", "balanced_accuracy"]
    tabs = {}
    for loc in sorted(results_df["location"].unique()):
        yv = np.asarray(y_LOC[loc])
        pos_rate = float((yv == 1).mean())
        sp = results_df[(results_df["location"] == loc) & (results_df["eval_type"] == "spatial_cv")]
        ho = results_df[(results_df["location"] == loc) & (results_df["eval_type"] == "holdout")]
        parts = [mo.md("**" + loc + "** (no-skill PR-AUC " + str(round(pos_rate, 4)) + ")")]
        if len(sp):
            cols = [c for c in DISPLAY_COLS if c in sp.columns]
            summ = sp.groupby(["model", "config"])[cols].mean().round(4).reset_index()
            summ["pr_auc_std"] = sp.groupby(["model", "config"])["pr_auc"].std().round(4).to_numpy()
            summ["pr_auc_lift"] = (summ["pr_auc"] - pos_rate).round(4)
            parts.append(mo.md("Honest spatial-block CV (mean over folds)"))
            parts.append(mo.ui.table(summ))
        if len(ho):
            show = ["model", "config"] + [c for c in DISPLAY_COLS if c in ho.columns]
            parts.append(mo.md("Random holdout (leakage ceiling, reference only)"))
            parts.append(mo.ui.table(ho[show].round(4)))
        tabs[loc] = mo.vstack(parts)
    mo.stop(len(tabs) == 0, mo.md("No results yet. Press **Run ML**."))
    mo.ui.tabs(tabs)
    return


@app.cell(hide_code=True)
def _(PLOTS_DIR, results_df):
    import matplotlib.pyplot as plt

    mo.stop(len(results_df) == 0, mo.md("No results yet. Press **Run ML**."))
    sp = results_df[results_df["eval_type"] == "spatial_cv"].copy()
    ho = results_df[results_df["eval_type"] == "holdout"].copy()
    locs = sorted(sp["location"].unique())
    mo.stop(len(locs) == 0, mo.md("No spatial CV rows to plot."))
    sp["label"] = sp["model"] + " / " + sp["config"]
    means = sp.groupby(["location", "label"])["pr_auc"].mean().unstack("label")
    stds = sp.groupby(["location", "label"])["pr_auc"].std().unstack("label")
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(means.columns)), 5))
    means.plot(kind="bar", yerr=stds, ax=ax, capsize=3, colormap="Set2")
    ax.set_title("Spatial-block CV PR-AUC by location (honest)")
    ax.set_ylabel("PR-AUC")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_spatial_pr_auc.png", dpi=150)
    for loc in sorted(ho["location"].unique()):
        rows = ho[ho["location"] == loc]
        best = rows.sort_values("pr_auc", ascending=False).groupby("model").head(1)
        n = len(best)
        if n == 0:
            continue
        fig2, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        for a in range(n):
            r = best.iloc[a]
            ax2 = axes[0][a]
            cm = [[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]]
            ax2.imshow(cm, cmap="Blues")
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(["Healthy", "Unhealthy"])
            ax2.set_yticklabels(["Healthy", "Unhealthy"])
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            ax2.set_title(str(r["model"]) + " / " + str(r["config"]))
            for ii in range(2):
                for jj in range(2):
                    ax2.text(jj, ii, str(cm[ii][jj]), ha="center", va="center")
        fig2.suptitle(loc + " holdout confusion (reference)")
        fig2.tight_layout()
        fig2.savefig(PLOTS_DIR / ("confusion_" + loc + ".png"), dpi=150)
        plt.close(fig2)
    print("Plots saved to " + str(PLOTS_DIR))
    fig
    return


@app.cell(hide_code=True)
def _(results_df):
    try:
        from scipy.stats import wilcoxon
        HAVE_W = True
    except ImportError:
        HAVE_W = False
    sp = results_df[results_df["eval_type"] == "spatial_cv"].copy()
    ho = results_df[results_df["eval_type"] == "holdout"].copy()
    for loc in sorted(results_df["location"].unique()):
        print("=" * 80)
        print("  " + loc)
        print("=" * 80)
        sl = sp[sp["location"] == loc]
        if HAVE_W and sl["model"].nunique() > 1:
            best = {}
            for model_name in sorted(sl["model"].unique()):
                mc = sl[sl["model"] == model_name]
                cfg = mc.groupby("config")["pr_auc"].mean().idxmax()
                best[model_name] = (cfg, mc[mc["config"] == cfg]["pr_auc"].to_numpy())
            models = sorted(best)
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    m1 = models[i]
                    m2 = models[j]
                    cfg1, s1 = best[m1]
                    cfg2, s2 = best[m2]
                    if len(s1) == len(s2) and len(s1) > 1:
                        stat, p = wilcoxon(s1, s2)
                        if p < 0.001:
                            sig = "***"
                        elif p < 0.01:
                            sig = "**"
                        elif p < 0.05:
                            sig = "*"
                        else:
                            sig = "ns"
                        print("  " + m1 + " (" + cfg1 + ") vs " + m2 + " (" + cfg2 + "): p=" + str(round(float(p), 4)) + " (" + sig + ")")
        for met in ["pr_auc", "f1_unhealthy", "recall_unhealthy", "roc_auc"]:
            hl = ho[ho["location"] == loc]
            if len(hl) and met in hl.columns:
                b = hl.loc[hl[met].idxmax()]
                print("  Best holdout " + met + ": " + str(b["model"]) + " (" + str(b["config"]) + ") = " + str(round(float(b[met]), 4)) + " (ceiling reference)")
    return


if __name__ == "__main__":
    app.run()
