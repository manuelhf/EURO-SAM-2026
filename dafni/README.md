# Running the Ouseburn pipeline on DAFNI

The two notebooks are split into three [DAFNI](https://www.dafni.ac.uk/) models, which can be chained into one
DAFNI workflow:

| Model | Folder | Notebook source | Input | Output (`/data/outputs`) |
|---|---|---|---|---|
| 1. Network | `network/` | `Euro_SAM_v1` cells 4 and 6 | optional SWMManywhere config (`.yml`) | `model_N.inp`, geoparquet files, checks, maps |
| 2. Scenarios | `scenarios/` | `Euro_SAM_v1` cells 8, 10 and 12 | one `.inp` | the notebook's `ouseburn_dataset/` files |
| 3. Graph learning | `gnn/` | `Euro_SAM_GNN` cells 3, 5 and 7 | `nodes.csv`, `edges.csv`, `scenarios.csv`, `timeseries.npz` | the notebook's `ouseburn_gnn/` files |

Each folder has:

- `ouseburn_*.py`: the notebook code, extracted unchanged (the settings stay module globals).
- `run_*.py`: the DAFNI entry point. It reads parameters from environment variables, finds inputs anywhere
  under `/data/inputs` (unpacking any `.zip`), sets the notebook globals and calls the notebook's functions.
- `Dockerfile`, plus `requirements.in` (direct dependencies) and `requirements.txt` (the Linux lockfile).
- `model_definition.yaml`: the DAFNI model definition (schema `v1beta3`).

`common/dafni_io.py` is shared by the three images. The notebooks themselves are not modified.

## Getting the images

The [DAFNI images](../.github/workflows/dafni-images.yml) GitHub Action builds each image for `linux/amd64`
and saves it with `docker save | gzip`.

- **Every run (push to `main`, pull request, manual run):** produces one artifact per model, named
  `ouseburn-<model>-dafni`. Each artifact holds `ouseburn-<model>-<tag>.tar.gz` and `model_definition.yaml`.
  GitHub downloads artifacts as a `.zip`; unzip it to get the `.tar.gz`.
- **Tag `v*` (e.g. `v1.0.0`):** the same files are also attached to the GitHub release.
- **Manual run with "pipeline_test" ticked:** after the builds, the three containers run in sequence with
  small settings (4 events × 2 disturbances, 2 folds × 2 epochs), the way a DAFNI workflow would.

To build locally from the repository root:

```bash
docker build --platform linux/amd64 -f dafni/gnn/Dockerfile -t ouseburn-gnn:latest .
```

```bash
docker save ouseburn-gnn:latest | gzip > ouseburn-gnn.tar.gz
```

## Uploading to DAFNI

In the DAFNI web app go to *Model Catalogue → Add model* and upload the `.tar.gz` together with its
`model_definition.yaml`. Or use the [DAFNI CLI](https://github.com/dafnifacility/cli):

```bash
dafni upload model dafni/gnn/model_definition.yaml ouseburn-gnn-v1.0.0.tar.gz -m "EURO-SAM 2026 release"
```

Then upload the inputs as DAFNI datasets, or chain the models in a workflow. Outputs of an earlier step
appear under `/data/inputs/<step-name>`, and the entry points search all of `/data/inputs`:

1. **Network**: no input needed (it uses the repository's `ouseburn_config.yml`). Optionally set `BBOX` or
   attach a different config. It downloads OpenStreetMap (Overpass), building footprints and NASADEM
   elevation (Microsoft Planetary Computer) at run time; DAFNI models do have internet access. About
   8 minutes on DAFNI.
2. **Scenarios**: input is the network output (or a dataset with one `.inp`). Defaults reproduce the
   reported 108 runs; the notebook's "stronger-disturbance variant" is `RAIN_PEAK_MMH=0,25`,
   `BLOCKAGE_FACTOR=0.05,0.25`. About 70 minutes on DAFNI.
3. **Graph learning**: input is the scenarios output (or the `ouseburn_dataset` folder as a dataset).
   Defaults are the notebook's 4-fold event-wise cross-validation, 60 epochs and 2000 bootstrap resamples.
   It runs on CPU; about 1 hour on DAFNI with the defaults.

### Building the workflow in DAFNI

These points come from the first runs of the full workflow (October 2026):

- **Data only flows through "Choose steps to receive data from".** In the workflow builder, the arrows
  drawn between steps only set the order in which they run. A step receives the previous step's
  `/data/outputs` only if that step is ticked in its **Choose steps to receive data from** setting. If a
  step is linked by an arrow alone, it starts with an empty `/data/inputs` and stops with "Expected exactly
  one … under ['/data/inputs'] … found []". Set it for **scenarios** (← network) and **graph-learning**
  (← scenarios).
- **Outputs are only kept by a Publish step.** Add a *Publish* step after scenarios and after graph
  learning, with file rows such as `outputs/**/*` for the step whose files you want. Two publish rows must
  not pick up the same file name (all three models write `run_parameters.json`).
- **The network step depends on outside services.** If Overpass or Microsoft's Planetary Computer is
  briefly unavailable, the step fails with an error such as
  `pystac_client.exceptions.APIError: {"error":"Service is unavailable."}`. Rerun the workflow once the
  service is back. To avoid the dependency, start the workflow at scenarios and give it the published
  network `.inp` as a dataset.
- **GNN results vary slightly between machines.** Graph-learning training on DAFNI's CPUs is not
  bit-identical to a Colab GPU run: F1 and average precision agreed within 0.01–0.03, but the small
  topology-vs-hydraulic difference in source localisation changed sign. In the DAFNI run the paired
  bootstrap marks that difference as not significant (`hydro_vs_topo_bootstrap.csv`).

The dataslots are optional in the definitions because DAFNI requires a default dataset UUID for a required
dataslot. Once the datasets are on DAFNI, you can add their UUIDs as `default:` and set `required: true`.
A run without input stops with a message naming the files it expected.

## Parameters

Parameter names are the notebook setting names. The full list, with ranges and descriptions, is in each
`model_definition.yaml`. Ranges such as `RAIN_PEAK_MMH` are written as `"min,max"`. DAFNI passes every
parameter as a string; booleans arrive as `True`/`False`.

## Running locally without Docker

The entry points also run from a checkout. `INPUTS_DIR` and `OUTPUTS_DIR` replace `/data/inputs` and
`/data/outputs`:

```bash
INPUTS_DIR=run/in OUTPUTS_DIR=run/network python dafni/network/run_network.py
```

```bash
INPUTS_DIR=run/network OUTPUTS_DIR=run/scenarios N_RAIN_EVENTS=4 DISTURBANCES_PER_EVENT=2 python dafni/scenarios/run_scenarios.py
```

```bash
INPUTS_DIR=run/scenarios OUTPUTS_DIR=run/gnn N_FOLDS=2 EPOCHS=2 N_BOOT=20 python dafni/gnn/run_gnn.py
```

On Apple-silicon Macs, `swmm-toolkit` 0.17.0 (pyswmm's SWMM engine) ships dylibs with broken signatures,
and macOS kills Python on `import pyswmm`. Re-sign them once inside your environment:
`codesign --force -s - <site-packages>/swmm/toolkit/*.dylib <site-packages>/swmm/toolkit/*.so`.
The Linux images are not affected.

## Updating

- If the notebooks change, re-extract the cells into the `ouseburn_*.py` files. Each file's header lists
  which cells it contains.
- After editing a `requirements.in`, regenerate the lockfiles (shared packages are pinned to the network
  lock, so all three images use the same numpy, pandas, networkx and SWMM engine):

```bash
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 dafni/network/requirements.in -o dafni/network/requirements.txt
```

  For `scenarios`, add `-c dafni/network/requirements.txt`. For `gnn`, also add
  `--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`.
- `python dafni/validate_definitions.py` checks the three definitions (this also runs in CI).
