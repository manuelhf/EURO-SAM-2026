"""DAFNI entry point - model 2: hydraulic redesign, sensors and SWMM scenarios.

Runs notebook Euro_SAM_v1 steps 2-4 on a SWMManywhere model: keep the main
drainage systems, redesign pipes to standards, rank manholes for monitoring,
optionally run the hydraulic diagnostic, then simulate normal + disturbed runs.

Inputs
  /data/inputs/**/ exactly one *.inp (a dataslot, or the output of the network
  model in a DAFNI workflow); .zip files are unpacked first.
Parameters: see model_definition.yaml (names = notebook settings).
Outputs (/data/outputs) - the notebook's ouseburn_dataset/ folder:
  nodes.csv, edges.csv, scenarios.csv, timeseries.npz, checks.txt,
  sensors_map.png, example_blockage.png, example_inflow.png,
  diagnose.txt (if RUN_DIAGNOSE), run_parameters.json
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]   # image: /app; repo: dafni/common
import dafni_io as io  # noqa: E402
import ouseburn_scenarios as sc  # noqa: E402


def stage_model(work: Path) -> Path:
    """Copy the single input .inp to work/<stem>/<stem>.inp (layout the notebook expects)."""
    roots = [io.INPUTS, *io.unpack_zips(io.INPUTS, work / "unzipped")]
    found = io.find_files("*.inp", roots)
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one .inp file under {io.INPUTS}; found {found}")
    model_dir = work / found[0].stem
    model_dir.mkdir(parents=True)
    shutil.copy2(found[0], model_dir / found[0].name)
    return model_dir


def configure() -> dict:
    s = dict(
        SEED=io.env_int("SEED", 42, 0),
        N_RAIN_EVENTS=io.env_int("N_RAIN_EVENTS", 12, 1, 500),
        DISTURBANCES_PER_EVENT=io.env_int("DISTURBANCES_PER_EVENT", 8, 1, 100),
        BLOCKAGE_SHARE=io.env_float("BLOCKAGE_SHARE", 0.5, 0, 1),
        DURATION_H=io.env_int("DURATION_H", 3, 1, 48),
        MIN_COMPONENT=io.env_int("MIN_COMPONENT", 50, 1),
        SENSOR_FRACTION=io.env_float("SENSOR_FRACTION", 0.05),
        SENSOR_STRATEGY=io.env_str("SENSOR_STRATEGY", "practice", ("practice", "spread")),
        SENSOR_SPACING_M=io.env_float("SENSOR_SPACING_M", 150, 1),
        REDESIGN=io.env_bool("REDESIGN", True),
        DESIGN_RAIN_MMH=io.env_float("DESIGN_RAIN_MMH", 50, 1),
        MAX_MANHOLE_DEPTH_M=io.env_float("MAX_MANHOLE_DEPTH_M", 8.0, 1),
        RAIN_PEAK_MMH=io.env_floats("RAIN_PEAK_MMH", (0.0, 12.0), n=2),
        BLOCKAGE_FACTOR=io.env_floats("BLOCKAGE_FACTOR", (0.10, 0.40), n=2),
        INFLOW_LPS=io.env_floats("INFLOW_LPS", (3.0, 25.0), n=2),
        QA_MAX_ERROR_PCT=io.env_float("QA_MAX_ERROR_PCT", 10, 0),
    )
    # headline density must be one of the nested sets (notebook comment)
    match = [d for d in sc.SENSOR_DENSITIES if abs(d - s["SENSOR_FRACTION"]) < 1e-9]
    if not match:
        raise ValueError(f"SENSOR_FRACTION must be one of {sc.SENSOR_DENSITIES}")
    s["SENSOR_FRACTION"] = match[0]
    for k in ("RAIN_PEAK_MMH", "BLOCKAGE_FACTOR", "INFLOW_LPS"):
        if s[k][0] > s[k][1]:
            raise ValueError(f"{k} must be 'min,max', got {s[k]}")
    if not (0 < s["BLOCKAGE_FACTOR"][0] and s["BLOCKAGE_FACTOR"][1] <= 1):
        raise ValueError(f"BLOCKAGE_FACTOR must lie in (0, 1], got {s['BLOCKAGE_FACTOR']}")
    for k, v in s.items():          # module globals read by the notebook functions
        setattr(sc, k, v)
    return s


def main() -> None:
    out = io.OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    settings = configure()
    run_diagnose = io.env_bool("RUN_DIAGNOSE", True)
    work = Path(tempfile.mkdtemp(prefix="ouseburn_scenarios_"))
    try:
        sc.MODEL_DIR = stage_model(work)
        sc.DATASET_DIR = out
        inp = sc.MODEL_DIR / f"{sc.MODEL_DIR.name}.inp"
        io.write_json(out / "run_parameters.json",
                      dict(settings, RUN_DIAGNOSE=run_diagnose, input_model=inp.name,
                           input_sha256=io.sha256(inp)))
        print(f"Settings: {settings}")
        if run_diagnose:                                   # notebook cell 10
            with io.tee_stdout(out / "diagnose.txt"):
                sc.diagnose()
        sc.main()                                          # notebook cell 12
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
