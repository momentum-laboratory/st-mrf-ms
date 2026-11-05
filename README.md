# Quantitative semisolid MT and rNOE imaging in a multiple sclerosis mouse model

[![DOI](https://zenodo.org/badge/1337288812.svg)](https://doi.org/10.5281/zenodo.23989929)

Analysis code for *"Quantitative Semisolid Magnetization Transfer and Relayed
Nuclear Overhauser Effect Imaging in a Multiple Sclerosis Mouse Model Using Deep
Magnetic Resonance Fingerprinting"* (Ben Chaim, Rivlin, Perlman),
*Chem. Biomed. Imaging* **2026**.
[doi:10.1021/cbmi.6c00179](https://doi.org/10.1021/cbmi.6c00179)

The framework quantifies the proton exchange dynamics of three molecular pools —
semisolid MT, rNOE(-3.5 ppm) and rNOE(-1.6 ppm) — from saturation transfer MR
fingerprinting data, using a serial pair of neural networks. It is validated on
brain-lipid phantoms and applied longitudinally to a cuprizone mouse model of
demyelination and remyelination.

## Pipelines

Two stages, each a separately trained network, run in series (paper Figure 1):
stage 1 predicts semisolid MT from the MT protocol conditioned on T1/T2; stage 2
predicts both rNOE pool fractions from the two rNOE protocols conditioned on
T1/T2 and the stage-1 MT parameters.

`main.ipynb` is the entry point and runs both studies:

| Study | Produces |
|---|---|
| `in_vitro` | Pool-fraction maps for the seven tubes, then `phantom_fs_maps_with_gt` — the maps beside a ground-truth panel |
| `in_vivo` | Pool-fraction maps for all 41 scans (12 mice x 4 timepoints), then `maps_grid_<mouse>_all_contrasts` — T1, T2 and the three pool fractions, one row per timepoint |

The notebook only orchestrates, and shows both figures inline as it goes.
`utils/quantification.py` spells out each stage step by step — which arrays go
in, in what order — while `utils/inference.py` holds the arithmetic they share.


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Inference runs on GPU if one is available and on CPU otherwise.

## Data

Everything each pipeline reads lives under `data/`, as plain NumPy arrays.

```
data/in_vivo/<mouse>/week<N>/       data/in_vitro/
    t1.npy                              t1_mean.npy    per-tube mean T1 (ms)
    t2.npy                              t2.npy         T2 map (ms)
    mt.npy        30 frames             mt.npy         semi-solid MT ST-MRF, 30 frames
    noe_3p5.npy   31 frames (M0 + 30)   noe_3p5.npy    rNOE -3.5 ppm ST-MRF, 30 frames
    noe_1p6.npy   31 frames (M0 + 30)   noe_1p6.npy    rNOE -1.6 ppm ST-MRF, 30 frames
    brain_mask.npy                      tube_masks.npy the 7 tube ROIs, (7, rows, cols)
```

Mice are numbered `mouse1`–`mouse12` and timepoints are weeks 0, 4, 7 and 12.
`utils/scans.py` lists all 41 scans, and its `ANIMALS` table maps each mouse back
to the cage, ear-tag and identifier used during acquisition — which is how the
raw scan deposit is organised.

Also included: the trained networks (`trained_nn/<study>/`) and the outputs
of a full run (`results/<study>/`).

## Running

```bash
jupyter lab main.ipynb
```

Run the cells top to bottom. Both figures render inline as they are produced,
and the last section says where everything was written. To look at a different
animal, change `mouse` in the figure cell to any of `mouse1`-`mouse12`.

## Layout

```
main.ipynb          the entry point; runs both studies, shows both figures
config.py           paths, network settings and display ranges, both studies

utils/
    quantification.py   each study's two stages, and its figure
    inference.py        running a trained network over acquired data — the
                        shared core; also array I/O and range (de)normalization
    networks.py         the reconstruction network architecture
    plots.py            rendering quantitative parameter maps
    scans.py            the in vivo cohort, and each animal's provenance

data/<study>/       inputs
trained_nn/<study>/ trained networks
results/<study>/    maps and figures
```

## Citation

If you use this code in a scientific publication, please cite:

> Ben Chaim, R.; Rivlin, M.; Perlman, O. Quantitative Semisolid Magnetization
> Transfer and Relayed Nuclear Overhauser Effect Imaging in a Multiple Sclerosis
> Mouse Model Using Deep Magnetic Resonance Fingerprinting. *Chem. Biomed. Imaging*
> **2026**. [doi:10.1021/cbmi.6c00179](https://doi.org/10.1021/cbmi.6c00179)
