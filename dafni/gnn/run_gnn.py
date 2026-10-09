"""DAFNI entry point - model 3: graph learning for sparse sewer monitoring.

Runs notebook Euro_SAM_GNN: IDW vs GNN-topo vs GNN-hydro with event-wise
cross-validation, paired bootstrap GNN-hydro vs GNN-topo, figures, and the
"typical case" example map redrawn from the saved fold models.

Inputs
  /data/inputs/**/ one folder with nodes.csv, edges.csv, scenarios.csv and
  timeseries.npz (a dataslot, or the output of the scenarios model in a DAFNI
  workflow); .zip files are unpacked first.
Parameters: see model_definition.yaml (names = notebook settings).
Outputs (/data/outputs) - the notebook's ouseburn_gnn/ folder:
  results_pooled.csv, results_by_fold.csv, hydro_vs_topo_bootstrap.csv,
  results_by_density.png, example_likelihood_map.png,
  GNN-topo_foldK.pt, GNN-hydro_foldK.pt, config.json, run_parameters.json
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]   # image: /app; repo: dafni/common
import dafni_io as io  # noqa: E402
import ouseburn_gnn as gnn  # noqa: E402

NEEDED = ("nodes.csv", "edges.csv", "scenarios.csv", "timeseries.npz")


def configure() -> dict:
    s = dict(
        SEED=io.env_int("SEED", 0, 0),
        N_FOLDS=io.env_int("N_FOLDS", 4, 2, 20),
        EPOCHS=io.env_int("EPOCHS", 60, 1, 1000),
        PATIENCE=io.env_int("PATIENCE", 10, 1, 1000),
        N_BOOT=io.env_int("N_BOOT", 2000, 10, 100000),
        HEADLINE_DENSITY=io.env_float("HEADLINE_DENSITY", 0.05),
    )
    if not any(abs(d - s["HEADLINE_DENSITY"]) < 1e-9 for d in gnn.DENSITIES):
        raise ValueError(f"HEADLINE_DENSITY must be one of {gnn.DENSITIES}")
    for k, v in s.items():          # module globals read by the notebook functions
        setattr(gnn, k, v)
    return s


def check_events(dataset: Path) -> None:
    """Every fold needs at least one training event (test group + VAL_EVENTS held out)."""
    n = pd.read_csv(dataset / "scenarios.csv").event_id.nunique()
    if n - math.ceil(n / gnn.N_FOLDS) - gnn.VAL_EVENTS < 1:
        raise ValueError(f"{n} rain events are too few for N_FOLDS={gnn.N_FOLDS} "
                         f"(+{gnn.VAL_EVENTS} validation event per fold)")


def main() -> None:
    out = io.OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    settings = configure()
    replot = io.env_bool("REPLOT_EXAMPLE", True)
    work = Path(tempfile.mkdtemp(prefix="ouseburn_gnn_"))
    try:
        dataset = io.find_dir_with(NEEDED, [io.INPUTS, *io.unpack_zips(io.INPUTS, work)])
        check_events(dataset)
        gnn.DATASET_DIR, gnn.OUT_DIR = dataset, out
        io.write_json(out / "run_parameters.json",
                      dict(settings, REPLOT_EXAMPLE=replot, DEVICE=gnn.DEVICE,
                           input_sha256={n: io.sha256(dataset / n) for n in NEEDED}))
        print(f"Dataset: {dataset}\nSettings: {settings}")

        pooled, spread, comp = gnn.run_experiment()          # notebook cell 5
        cols = ["f1", "avg_precision", "false_alarm_normal", "loc_within_2_hops"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\nPOOLED over all folds - test disturbances, unmonitored manholes:")
            print(pooled.pivot(index="density", columns="method", values=cols).round(3))
            print("\nGNN-hydro vs GNN-topo - paired bootstrap over test disturbances:")
            print(comp.round(3))
        if replot:                                           # notebook cell 7
            gnn.replot_example()
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
