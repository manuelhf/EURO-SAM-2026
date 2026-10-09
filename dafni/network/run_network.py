"""DAFNI entry point - model 1: synthetic sewer network (SWMManywhere).

Runs notebook Euro_SAM_v1 step 1 (and the 1b component check):
SWMManywhere synthesis for the configured bounding box + network checks.

Inputs
  /data/inputs/**/ *.yml|*.yaml  optional SWMManywhere config (dataslot);
                                 default: ouseburn_config.yml from the repo
Parameters (environment variables)
  BBOX   "lon_min,lat_min,lon_max,lat_max" (WGS84); empty = bbox in the config
Outputs (/data/outputs)
  model_N.inp, nodes/edges/subcatchments.geoparquet, graph.parquet,
  results.parquet       SWMManywhere model folder (flattened)
  network_map.png, components_map.png, outfalls.csv, network_checks.txt,
  ouseburn_config.yml   the config actually used, run_parameters.json

Needs internet: OpenStreetMap (Overpass), building footprints and NASADEM
elevation via Microsoft Planetary Computer.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]   # image: /app; repo: dafni/common
import dafni_io as io  # noqa: E402
import ouseburn_network as net  # noqa: E402

# image: /app/ouseburn_config.yml; repo checkout: eurosam-ouseburn-gnn/ouseburn_config.yml
DEFAULT_CONFIG = next(p for p in (HERE / "ouseburn_config.yml",
                                  HERE.parent.parent / "eurosam-ouseburn-gnn" / "ouseburn_config.yml")
                      if p.exists())
# the model files listed in the notebook (SWMManywhere's verbose mode also
# leaves one .geojson per graph step, which are not kept)
MODEL_FILES = ("*.inp", "nodes.*", "edges.*", "subcatchments.*", "graph.parquet",
               "results.parquet", "network_map.png")


def pick_config() -> Path:
    found = io.find_files("*.yml", [io.INPUTS]) + io.find_files("*.yaml", [io.INPUTS])
    if len(found) > 1:
        raise ValueError(f"More than one config file in the inputs: {found}")
    return found[0] if found else DEFAULT_CONFIG


def main() -> None:
    out = io.OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ouseburn_network_"))
    os.chdir(work)                      # osmnx caches in ./cache

    src = pick_config()
    cfg = yaml.safe_load(src.read_text())
    cfg["base_dir"] = str(work / "ouseburn_output")
    bbox = io.env_floats("BBOX", tuple(cfg["bbox"]), n=4)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"BBOX must be lon_min,lat_min,lon_max,lat_max; got {bbox}")
    cfg["bbox"] = list(bbox)
    cfg_path = work / "ouseburn_config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    io.write_json(out / "run_parameters.json",
                  dict(config_source=str(src), config_sha256=io.sha256(src), bbox=bbox))

    with io.tee_stdout(out / "network_checks.txt"):
        print(f"Config: {src}")
        # notebook cell 4, driver lines (REGENERATE is irrelevant: fresh container)
        config = net.load_project_config(cfg_path)
        inp = net.run_generation(config)
        outputs = net.load_outputs(inp)
        net.check_network(inp, outputs)
        net.plot_network(outputs, inp.parent / "network_map.png")
        # notebook cell 6
        print()
        net.component_report(inp.parent, out / "components_map.png").to_csv(
            out / "outfalls.csv", index=False)

    for pattern in MODEL_FILES:
        for f in inp.parent.glob(pattern):
            shutil.copy2(f, out / f.name)
    # keep the used config with the outputs, base_dir back to the repo default
    cfg["base_dir"] = "./ouseburn_output"
    (out / "ouseburn_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"\n✓ Network model written to {out} ({inp.name})")


if __name__ == "__main__":
    main()
