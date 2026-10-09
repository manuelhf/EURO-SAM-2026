# Extracted from eurosam-ouseburn-gnn/Euro_SAM_v1.ipynb, cell(s) 4 (driver lines at the end moved to run_network.py) and 6 (as a function).
# The scientific code is unchanged; settings are module globals that the
# DAFNI entry point overrides from environment variables.

"""
Ouseburn sewer network generation with SWMManywhere (verified against v0.2.4)
==============================================================================
Reads settings from ouseburn_config.yml (single source of truth), runs the
synthesis + a short SWMM simulation, then checks the result is usable for the
graph-learning demo (connectivity, outfalls, slopes, diameters).

Outputs (in ./ouseburn_output/ouseburn/bbox_N/model_M/):
    model_M.inp              SWMM model
    nodes.geoparquet         junctions + outfalls
    edges.geoparquet         conduits
    subcatchments.geoparquet sub-catchments
    graph.parquet            networkx graph
    results.parquet          simulated flooding / flow / depth (long format)

Run:
    pip install swmmanywhere matplotlib
    python ouseburn_swmmanywhere.py

Needs internet on first run: OSM (streets, rivers), Overture/Google-Microsoft
buildings, NASADEM elevation via Microsoft Planetary Computer.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests

from swmmanywhere.logging import set_verbose
from swmmanywhere.swmmanywhere import load_config, swmmanywhere

# Works with `!python script.py`, `%run`, or code pasted into a cell
# (falls back to the current working directory, e.g. your Drive folder).
try:
    CONFIG_PATH = Path(__file__).with_name("ouseburn_config.yml")
except NameError:
    CONFIG_PATH = Path("ouseburn_config.yml")

# Public Overpass (OpenStreetMap) servers, tried in order. The main one,
# overpass-api.de, often refuses connections from busy shared IPs (e.g. Colab).
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]


def use_overpass(url: str) -> None:
    """Point osmnx (1.x or 2.x) at an Overpass server."""
    if hasattr(ox.settings, "overpass_url"):        # osmnx >= 2.0
        ox.settings.overpass_url = url
    else:                                           # osmnx 1.x
        ox.settings.overpass_endpoint = url
    ox.settings.overpass_rate_limit = False  # mirrors may lack /status endpoint
    ox.settings.requests_timeout = 300


def _is_overpass_error(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}".lower()
    return isinstance(e, requests.exceptions.RequestException) or any(
        k in text for k in ("overpass", "insufficientresponse", "interpreter")
    )


def _remove_failed_model_dirs(config: dict) -> None:
    """Delete model_N folders left behind by failed attempts (no .inp inside)."""
    proj = Path(config["base_dir"]) / config["project"]
    for d in proj.glob("bbox_*/model_*"):
        if d.is_dir() and not any(d.glob("*.inp")):
            shutil.rmtree(d, ignore_errors=True)


# -----------------------------------------------------------------------------
# 1. Load + sanity-check config
# -----------------------------------------------------------------------------
def load_project_config(path: Path = CONFIG_PATH) -> dict:
    raw_base = Path(load_config(path, validation=False)["base_dir"])
    raw_base.mkdir(parents=True, exist_ok=True)  # load_config requires it to exist
    config = load_config(path)                   # validates schema + overrides

    lon0, lat0, lon1, lat1 = config["bbox"]
    lat = (lat0 + lat1) / 2
    w = (lon1 - lon0) * 111_320 * math.cos(math.radians(lat))
    h = (lat1 - lat0) * 110_574
    print(f"BBox {config['bbox']}  ≈ {w:.0f} m × {h:.0f} m")
    return config


# -----------------------------------------------------------------------------
# 2. Run SWMManywhere
# -----------------------------------------------------------------------------
def run_generation(config: dict) -> Path:
    set_verbose(True)  # also required for results.parquet to be written
    last_err = None
    for url in OVERPASS_MIRRORS:
        use_overpass(url)
        print(f"\nOSM data via {url}")
        try:
            out_path, _metrics = swmmanywhere(dict(config))  # (path, metrics|None)
            break
        except Exception as e:
            if not _is_overpass_error(e):
                raise
            print(f"[!] {url} failed: {type(e).__name__} — trying next server")
            last_err = e
            _remove_failed_model_dirs(config)
    else:
        raise RuntimeError(
            "All Overpass servers failed. Wait 10-15 min, or in Colab use "
            "Runtime → Disconnect and delete runtime to get a new IP, then retry."
        ) from last_err
    if out_path.suffix != ".inp":
        raise RuntimeError(
            f"No pipes were generated (got {out_path.name}). Usually the bbox has "
            "no river inside it, or too few streets — check the bbox on a map."
        )
    print(f"\n✓ Model written to: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# 3. Load outputs + network checks
# -----------------------------------------------------------------------------
def _read(path_stem: Path) -> gpd.GeoDataFrame | None:
    for ext in (".geoparquet", ".geojson"):
        p = path_stem.with_suffix(ext)
        if p.exists():
            return gpd.read_parquet(p) if ext == ".geoparquet" else gpd.read_file(p)
    return None


def load_outputs(inp_path: Path) -> dict:
    d = inp_path.parent
    out = {k: _read(d / k) for k in ("nodes", "edges", "subcatchments")}
    res = d / "results.parquet"
    out["results"] = pd.read_parquet(res) if res.exists() else None
    return out


def check_network(inp_path: Path, out: dict) -> None:
    nodes, edges = out["nodes"], out["edges"]
    print("\n" + "=" * 60 + "\nNetwork checks\n" + "=" * 60)

    river = inp_path.parents[2] / "download" / "river.json"
    if river.exists() and river.stat().st_size < 200:
        print("[!] river.json is (nearly) empty — no watercourse in the bbox")

    if nodes is None or edges is None:
        print("[!] nodes/edges files missing"); return
    print(f"Nodes: {len(nodes)}   Edges: {len(edges)}   CRS: {edges.crs}")
    print(f"Node columns: {list(nodes.columns)}")
    print(f"Edge columns: {list(edges.columns)}")

    # Connectivity: each weakly connected component drains to its own outfall
    if {"u", "v"} <= set(edges.columns):
        G = nx.DiGraph(list(zip(edges["u"], edges["v"])))
        comps = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
        outfalls = [n for n in G if G.out_degree(n) == 0]
        print(f"Components: {len(comps)} (sizes {comps[:8]}{' …' if len(comps) > 8 else ''})")
        print(f"Sink/outfall nodes: {len(outfalls)}")
        if comps and comps[0] < 0.5 * G.number_of_nodes():
            print("[!] Largest component < 50% of nodes — network is fragmented; "
                  "consider raising outfall_derivation.outfall_length")

    for col in ("length", "diameter", "surface_slope", "chahinian_slope"):
        if col in edges.columns:
            s = pd.to_numeric(edges[col], errors="coerce").dropna()
            if len(s):
                print(f"{col:>16}: min {s.min():.4g}  median {s.median():.4g}  max {s.max():.4g}")
    if "diameter" in edges.columns:
        print(f"Diameters used: {sorted(edges['diameter'].dropna().round(3).unique())}")

    if (r := out["results"]) is not None:
        print(f"\nResults: {len(r)} rows, variables {sorted(r['variable'].unique())}"
              if "variable" in r.columns else f"\nResults columns: {list(r.columns)}")
        if {"variable", "value"} <= set(r.columns):
            fl = r[r["variable"] == "flooding"]
            if len(fl):
                print(f"Nodes flooding at any time: {fl.loc[fl['value'] > 0, 'id'].nunique()}"
                      if "id" in fl.columns else "")


# -----------------------------------------------------------------------------
# 4. Quick map
# -----------------------------------------------------------------------------
def plot_network(out: dict, save_to: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping"); return
    fig, ax = plt.subplots(figsize=(8, 10))
    if out["subcatchments"] is not None:
        out["subcatchments"].plot(ax=ax, facecolor="#d4eaf7", edgecolor="#7fa8c9", lw=0.4)
    if out["edges"] is not None:
        e = out["edges"]
        lw = 0.5 + 4 * e["diameter"] / e["diameter"].max() if "diameter" in e else 1.5
        e.plot(ax=ax, color="#1a6ea8", linewidth=lw)
    if out["nodes"] is not None:
        out["nodes"].plot(ax=ax, color="#e05c2a", markersize=6, zorder=3)
    ax.set_aspect("equal"); ax.set_title("SWMManywhere — lower Ouseburn synthetic network")
    plt.tight_layout(); plt.savefig(save_to, dpi=150)
    print(f"[plot] saved to {save_to}")


# Re-running this cell no longer builds a new model_N every time:
# it reuses the newest model for the bbox in ouseburn_config.yml.
REGENERATE = False   # True = run SWMManywhere again (new model_N folder)


def existing_model(config: dict) -> Path | None:
    from swmmanywhere.filepaths import check_bboxes
    proj = Path(config["base_dir"]) / config["project"]
    n = check_bboxes(tuple(config["bbox"]), proj) if proj.exists() else False
    if not n:
        return None
    models = [p / f"{p.name}.inp" for p in (proj / f"bbox_{n}").glob("model_*")
              if (p / f"{p.name}.inp").exists()]
    return max(models, key=lambda p: int(p.parent.name.split("_")[-1])) if models else None


# -----------------------------------------------------------------------------
# 1b. Components, outfalls and the river (notebook cell 6; plt.show() -> file)
# -----------------------------------------------------------------------------
def component_report(m: Path, save_to: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from swmmanywhere.utilities import load_graph

    print("Model:", m)

    nodes, edges = gpd.read_parquet(m/"nodes.geoparquet"), gpd.read_parquet(m/"edges.geoparquet")
    print(edges["edge_type"].value_counts())
    print(edges.groupby("edge_type")["diameter"].describe()[["count", "50%", "max"]])

    # River: bbox_N/download/river.json
    R = load_graph(m.parent/"download"/"river.json")
    river = gpd.GeoSeries([d["geometry"] for *_, d in R.edges(data=True)], crs=edges.crs)

    # Components (largest first) and outfalls (nodes with no outgoing pipe)
    G = nx.DiGraph(list(zip(edges.u, edges.v)))
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    comp = {n: i for i, c in enumerate(comps) for n in c}
    edges["comp"] = edges.u.map(comp)
    sinks = nodes[nodes.id.isin([n for n in G if G.out_degree(n) == 0])].copy()
    sinks["comp"] = sinks.id.map(comp)
    sinks["comp_size"] = sinks.comp.map(lambda i: len(comps[i]))
    sinks["dist_to_river_m"] = sinks.geometry.apply(lambda p: river.distance(p).min()).round(1)
    print("\nOutfalls:")
    print(sinks[["id", "comp", "comp_size", "surface_elevation", "dist_to_river_m"]]
          .sort_values("comp").to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 11))
    river.plot(ax=ax, color="deepskyblue", lw=3, alpha=.6)
    edges.plot(ax=ax, column="comp", cmap="tab10" if len(comps) <= 10 else "tab20", lw=1.5)
    sinks.plot(ax=ax, color="k", marker="v", markersize=60, zorder=3)
    for _, r in sinks.iterrows():
        ax.annotate(f"C{r.comp} ({r.comp_size})", (r.geometry.x, r.geometry.y),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(f"{m.parent.name}/{m.name}: {len(comps)} components, {len(sinks)} outfalls (▼)")
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close(fig)
    print(f"[plot] saved to {save_to}")
    return sinks[["id", "comp", "comp_size", "surface_elevation", "dist_to_river_m"]].sort_values("comp")
