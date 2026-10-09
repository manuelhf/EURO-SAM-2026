# Hydraulically-informed graph learning for sewer monitoring under data sparsity

Tetiana Starovoit · Manuel Herrera — Newcastle University
EURO-SAM workshop, Marseille, 12–13 October 2026 ([abstract](docs/EURO-SAM_2026_abstract.pdf))

Utilities can monitor only a few percent of manholes. This project asks whether those few sensors,
combined with a graph model of how the network is connected and how water flows through it,
can tell us which unmonitored manholes are affected by a disturbance.

## What is in this repository

| Path | Contents |
|---|---|
| `Euro_SAM_v1.ipynb` | Steps 1–4: synthetic network (SWMManywhere), hydraulic redesign and checks, sensor placement, 108 SWMM scenarios → `ouseburn_dataset/` |
| `Euro_SAM_GNN.ipynb` | Step 5: graph learning (interpolation vs GNN-topology vs GNN-hydraulic), 4-fold event-wise cross-validation → `ouseburn_gnn/` |
| `ouseburn_config.yml` | SWMManywhere configuration (bounding box, overrides, processing steps) |
| `docs/` | Workshop abstract |

Both notebooks are saved **with their outputs**, so the maps, logs and result tables of the reported run
can be read directly on GitHub without re-running anything.

Generated data folders (created by the notebooks, at the top level):

| Folder | Created by | Commit to GitHub? |
|---|---|---|
| `ouseburn_output/` | notebook v1, cell 3 (SWMManywhere) | Yes, except `*/download/` caches (excluded by `.gitignore`) |
| `ouseburn_dataset/` | notebook v1, last cell | Yes (nodes, edges, scenarios, time series, checks, maps) |
| `ouseburn_gnn/` | GNN notebook | Yes (results tables, figures, trained fold models) |

## How to run (Google Colab)

1. Put this repository's files in a Google Drive folder (by default `My Drive/Colab Notebooks/Euro-SAM 2026`).
2. Open `Euro_SAM_v1.ipynb` in Colab. In the first cell, set `PROJECT` to that folder, then run all cells.
   - The first network build downloads OpenStreetMap, elevation and building data (5–15 min); later runs reuse them.
   - If the main OpenStreetMap server refuses connections, the code switches to mirror servers automatically.
   - The network cell reuses an existing model; set `REGENERATE = True` to rebuild it.
   - Run the **Diagnose** cell before the full scenario run (≈1 min); the full run takes roughly 15–40 min.
3. Open `Euro_SAM_GNN.ipynb`, choose *Runtime → Change runtime type → T4 GPU*, set `PROJECT`, and run all cells
   (about 10 min on a T4). The last cell redraws the example likelihood map from the saved models without retraining.

Settings for each step sit at the top of its settings cell (sensor density, number of rain events,
disturbance ranges, redesign rules, training options).

## Running on DAFNI

[`dafni/`](dafni/README.md) packages the notebooks as three DAFNI models: network, scenarios and graph learning.
Each has a script, a Dockerfile and a DAFNI model definition. The *DAFNI images* GitHub Action builds each
image as a downloadable `.tar.gz`.

## Headline results (reported run)

Network: 590 manholes in two drainage systems (lower Ouseburn, Newcastle), redesigned to standards;
SWMM continuity error median 1.9 %, max 3.0 % over 108 runs; no flooding in normal operation.

At 5 % of manholes monitored (30 sensors), pooled over 4 event-wise cross-validation folds, unmonitored manholes:

| Method | F1 (affected) | Average precision | Source within 2 pipes | False alarms (normal runs) |
|---|---|---|---|---|
| Interpolation (IDW) | 0.23 | 0.14 | 3 % | 0 |
| GNN – topology | 0.78 | 0.77 | 13 % | 0 |
| GNN – hydraulic | **0.83** | **0.80** | **20 %** | 0 |

## Working together

- **Code and notebooks:** edit in Colab, then *File → Save a copy in GitHub* (or upload the notebook here).
  Notebooks with outputs merge badly, so agree who is editing which notebook, and work on a branch with a
  pull request for larger changes.
- **Data:** the Drive folder is the shared working copy; commit the data folders above when a run is final,
  so results are versioned alongside the code that produced them.

## Notes for the paper (known limitations and planned work)

- The comparison GNN-hydraulic vs GNN-topology changes several things at once (flow direction, pipe gates,
  expected-state inputs, ~1.4× parameters); planned: ablation one ingredient at a time, size-matched baseline.
- Both models already use the model's expected behaviour through the sensor deviations, so the comparison is conservative.
- F1 measures the *affected area*; source localisation (within 2 pipes) is the stricter test — report F1 per disturbance type.
- Bootstrap intervals treat disturbances as independent; planned: resample whole rain events, several training seeds per fold.
- 25 pumping stations are an artefact of the 30 m elevation data used for the layout; planned: depth-limit sensitivity
  (`MAX_MANHOLE_DEPTH_M = 12`) and 1 m Environment Agency LiDAR terrain.
- Node counts: 592 nodes = 590 manholes + 2 outfall nodes; 590 links = 588 between manholes + 2 outfall links.

## Built with

[SWMManywhere](https://github.com/ImperialCollegeLondon/SWMManywhere) · [pyswmm](https://github.com/pyswmm/pyswmm) / EPA SWMM 5 ·
geopandas · networkx · osmnx · PyTorch · scikit-learn · Google Colab

## Licence

No licence has been chosen yet; until then, all rights are reserved by the authors. Keep the repository private
or add a licence (e.g. MIT for the code) before making it public.
