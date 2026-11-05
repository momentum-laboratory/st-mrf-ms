"""Quantifying both studies, and building each one's figure.

The two stages of the framework (paper Figure 1) are the same operation in both
studies — see `inference`. What differs is only which arrays
are fed in, and how each study's figure is laid out. This file holds those two
study-specific parts, sharing everything else.

    build_stage       load a stage's network and its normalization bounds
    in_vitro_*        the brain-lipid phantom study
    in_vivo_*         the cuprizone mouse study
"""

import time

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr

import config as cfg
import torch

from .inference import (
    load_array,
    load_mask,
    load_maps,
    load_network,
    normalize_trajectories,
    read_bounds,
    run,
    save_maps,
    scale_pair,
    to_maps,
)
from .networks import Network
from .plots import b_viridis, b_winter, brain_crop, map_grid, map_row, masked_cmap
from .scans import MICE, SCANS


def load_stage(stage, device):
    """Load a stage's network and the bounds table saved alongside it."""
    net, saved = load_network(stage.weights, Network, device,
                              input_dim=stage.input_dim, n_hidden=stage.n_hidden,
                              n_neurons=stage.n_neurons)
    return net, saved[stage.minmax_key]


# ===========================================================================
# In vitro — brain-lipid phantoms
# ===========================================================================

def in_vitro_reference_data():
    """The per-tube mean T1 map, the T2 map, and the seven tube ROIs."""
    c = cfg.InVitro
    return load_array(c.t1_mean), load_array(c.t2), load_mask(c.tube_masks)


def in_vitro_quantify(device):
    """Run both stages over the phantom."""
    c = cfg.InVitro
    t1_map, t2_map, _ = in_vitro_reference_data()
    # The networks take relaxation times in seconds.
    t1_s, t2_s = t1_map / 1000, t2_map / 1000

    # ---- stage 1: semi-solid MT, from the MT protocol conditioned on T1/T2 --
    print('\n[stage 1] semi-solid MT')
    net, minmax = load_stage(c.MT, device)

    rows, cols, mrf = normalize_trajectories(
        load_array(c.MT.mrf), c.MT.sched_iter, device)
    t1t2 = scale_pair(t1_s, t2_s, read_bounds(minmax, 't1w', 't2w'),
                      rows, cols, device)

    prediction = run(net, torch.hstack([t1t2, mrf]))
    mt_maps = to_maps(prediction, read_bounds(minmax, 'mt_fs', 'mt_ksw'),
                      c.MT.output_keys, rows, cols)

    mt_maps['t1w'] = t1_map      # kept alongside the predictions, in ms
    mt_maps['t2w'] = t2_map
    save_maps(mt_maps, c.MT.maps_path)
    print(f'  saved -> {c.MT.maps_path}')

    # ---- stage 2: both rNOE pools, conditioned on T1/T2 and stage-1 MT ------
    print('\n[stage 2] joint rNOE(-3.5) + rNOE(-1.6)')
    net, minmax = load_stage(c.NOE, device)

    rows, cols, noe35 = normalize_trajectories(
        load_array(c.NOE.noe35), c.NOE.sched_iter, device)
    _, _, noe16 = normalize_trajectories(
        load_array(c.NOE.noe16), c.NOE.sched_iter, device)

    t1t2 = scale_pair(t1_s, t2_s, read_bounds(minmax, 't1', 't2'),
                      rows, cols, device)
    mt = scale_pair(mt_maps['fs'], mt_maps['ksw'],
                    read_bounds(minmax, 'mt_fs', 'mt_ksw'), rows, cols, device)

    prediction = run(net, torch.hstack([t1t2, mt, noe35, noe16]))
    noe_maps = to_maps(prediction, read_bounds(minmax, *c.NOE.minmax_output_keys),
                       c.NOE.output_keys, rows, cols)

    save_maps(noe_maps, c.NOE.maps_path)
    print(f'  saved -> {c.NOE.maps_path}')

    return mt_maps, noe_maps


def in_vitro_report(mt_maps, noe_maps, rois):
    """Per-tube pool fraction beside the tube's known lipid concentration.

    Prints each pool's correlation with concentration — the in vitro result the
    paper reports — and returns the per-tube table behind it.

    :return: DataFrame, one row per tube, values as mean +/- SD in percent
    """
    ground_truth = np.array(cfg.InVitro.GROUND_TRUTH_LIPID_PCT)
    pools = {
        'MT f_ss (%)':        mt_maps['fs'],
        'rNOE(-3.5) f_s (%)': noe_maps['noe_minus_3p5ppm_fs'],
        'rNOE(-1.6) f_s (%)': noe_maps['noe_minus_1p6ppm_fs'],
    }

    table = pd.DataFrame({'lipid (%)': ground_truth})
    print('\nCorrelation with lipid concentration')
    for label, fmap in pools.items():
        # Pool fractions are stored as fractions and reported in percent.
        means = np.array([fmap[roi].mean() for roi in rois]) * 100
        sds = np.array([fmap[roi].std() for roi in rois]) * 100
        table[label] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(means, sds)]

        r, p = pearsonr(ground_truth, means)
        print(f'  {label:22s} r = {r:.3f}   p = {p:.2e}')

    return table.set_index('lipid (%)')


def in_vitro_figure(mt_maps, noe_maps, rois):
    """Pool-fraction maps beside a ground-truth panel labelling each tube."""
    c = cfg.InVitro
    cmap = masked_cmap(b_viridis)
    display_mask = np.any(rois, axis=0)

    def label_tubes(ax):
        for roi, conc in zip(rois, c.GROUND_TRUTH_LIPID_PCT):
            rows, cols = np.where(roi)
            ax.text(cols.mean(), rows.mean(), f'{conc}%',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    panels = [{'data': np.where(display_mask, 1.0, np.nan),
               'cmap': masked_cmap(ListedColormap(['#999999'])), 'clim': (0, 1),
               'title': 'Ground truth (%)', 'colorbar': False,
               'annotate': label_tubes}]
    for fmap, title, clim in (
            (mt_maps['fs'], r'MT  $f_{ss}$ (%)', c.Figure.clim_mt),
            (noe_maps['noe_minus_3p5ppm_fs'],
             r'rNOE$_{-3.5\,\rm ppm}$  $f_{s}$ (%)', c.Figure.clim_noe),
            (noe_maps['noe_minus_1p6ppm_fs'],
             r'rNOE$_{-1.6\,\rm ppm}$  $f_{s}$ (%)', c.Figure.clim_noe)):
        panels.append({'data': fmap * 100, 'mask': display_mask,
                       'cmap': cmap, 'clim': clim, 'title': title})

    map_row(panels, figsize=(17, 4.5),
            suptitle='Brain-lipid phantom — volume-fraction maps with ground truth',
            save_path=c.figures / c.Figure.name)


# ===========================================================================
# In vivo — cuprizone mice
# ===========================================================================

# One column per contrast, in display order.
IN_VIVO_COLUMNS = [
    {'key': 't1', 'label': 'T$_1$ Maps',
     'cmap': 'hot', 'clim': (1000, 2000), 'ticks': np.arange(1000, 2200, 200)},
    {'key': 't2', 'label': 'T$_2$ Maps',
     'cmap': b_winter, 'clim': (30, 90), 'ticks': np.arange(30, 100, 10)},
    {'key': 'mt_fs', 'label': 'MT $f_{ss}$ (%)',
     'cmap': b_viridis, 'clim': (0, 16), 'ticks': np.arange(0, 18, 2)},
    {'key': 'noe_minus_3p5ppm_fs', 'label': 'rNOE -3.5 ppm $f_s$ (%)',
     'cmap': b_viridis, 'clim': (1, 2.5), 'ticks': np.arange(1, 3.0, 0.5)},
    {'key': 'noe_minus_1p6ppm_fs', 'label': 'rNOE -1.6 ppm $f_s$ (%)',
     'cmap': b_viridis, 'clim': (1, 2.5), 'ticks': np.arange(1, 3.0, 0.5)},
]

# Pool fractions are stored as fractions and displayed as percentages.
PERCENT_KEYS = ['mt_fs', 'noe_minus_3p5ppm_fs', 'noe_minus_1p6ppm_fs']


def in_vivo_scans(mouse_filter=None):
    """The scans to process, optionally restricted to one animal."""
    scans = [s for s in SCANS if mouse_filter in (None, s[0])]
    if not scans:
        raise SystemExit(f'unknown mouse {mouse_filter!r}; available: {", ".join(MICE)}')
    return scans


def _in_vivo_relaxation(mouse, week):
    """T1 and T2 for one scan, in seconds (what the networks expect)."""
    return (load_array(cfg.InVivo.array_path(mouse, week, 't1')) / 1000,
            load_array(cfg.InVivo.array_path(mouse, week, 't2')) / 1000)


def in_vivo_quantify(device, mouse_filter=None):
    """Run both stages over the cohort, saving the maps for every scan."""
    c = cfg.InVivo

    # The networks are the same for every scan, so load them once.
    mt_net, mt_minmax = load_stage(c.MT, device)
    noe_net, noe_minmax = load_stage(c.NOE, device)

    scans = in_vivo_scans(mouse_filter)
    for mouse, week, scan_name in scans:
        t0 = time.time()
        t1, t2 = _in_vivo_relaxation(mouse, week)

        # ---- stage 1: semi-solid MT, conditioned on T1/T2 ------------------
        rows, cols, mrf = normalize_trajectories(
            load_array(c.array_path(mouse, week, c.MT.mrf)),
            c.MT.sched_iter, device)
        t1t2 = scale_pair(t1, t2, read_bounds(mt_minmax, 't1w', 't2w'),
                          rows, cols, device)

        prediction = run(mt_net, torch.hstack([t1t2, mrf]))
        mt_maps = to_maps(prediction, read_bounds(mt_minmax, 'mt_fs', 'mt_ksw'),
                          c.MT.output_keys, rows, cols)
        save_maps(mt_maps, c.MT.maps_path(scan_name))

        # ---- stage 2: both rNOE pools, conditioned on T1/T2 and stage-1 MT -
        rows, cols, noe35 = normalize_trajectories(
            load_array(c.array_path(mouse, week, c.NOE.noe35)),
            c.NOE.sched_iter, device)
        _, _, noe16 = normalize_trajectories(
            load_array(c.array_path(mouse, week, c.NOE.noe16)),
            c.NOE.sched_iter, device)

        t1t2 = scale_pair(t1, t2, read_bounds(noe_minmax, 't1', 't2'),
                          rows, cols, device)
        mt = scale_pair(mt_maps['fs'], mt_maps['ksw'],
                        read_bounds(noe_minmax, 'mt_fs', 'mt_ksw'),
                        rows, cols, device)

        prediction = run(noe_net, torch.hstack([t1t2, mt, noe35, noe16]))
        noe_maps = to_maps(prediction,
                           read_bounds(noe_minmax, *c.NOE.minmax_output_keys),
                           c.NOE.output_keys, rows, cols)
        save_maps(noe_maps, c.NOE.maps_path(scan_name))

        print(f'{scan_name:18s} '
              f'MT {np.nanmean(mt_maps["fs"]) * 100:5.2f}%  |  '
              f'rNOE(-3.5) {np.nanmean(noe_maps["noe_minus_3p5ppm_fs"]) * 100:.2f}%  |  '
              f'rNOE(-1.6) {np.nanmean(noe_maps["noe_minus_1p6ppm_fs"]) * 100:.2f}%  '
              f'({time.time() - t0:.1f} s)')

    print(f'\nProcessed {len(scans)} scans.')
    print(f'  stage 1 maps -> {c.MT.maps_dir}')
    print(f'  stage 2 maps -> {c.NOE.maps_dir}')


def in_vivo_figure(mouse=None):
    """Per-timepoint map grid for one animal: rows are weeks, columns contrasts."""
    c = cfg.InVivo
    mouse = mouse or c.Figure.representative_mouse
    scans = sorted(in_vivo_scans(mouse), key=lambda s: int(s[1]))

    rows = []
    for _, week, scan_name in scans:
        whole_brain = load_mask(c.roi_path(mouse, week, 'whole_brain'))
        mt = load_maps(c.MT.maps_path(scan_name))
        noe = load_maps(c.NOE.maps_path(scan_name))
        data = {
            't1': load_array(c.array_path(mouse, week, 't1')),
            't2': load_array(c.array_path(mouse, week, 't2')),
            'mt_fs': mt['fs'],
            'noe_minus_3p5ppm_fs': noe['noe_minus_3p5ppm_fs'],
            'noe_minus_1p6ppm_fs': noe['noe_minus_1p6ppm_fs'],
        }
        for key in PERCENT_KEYS:
            data[key] = data[key] * 100
        rows.append({'label': f'Week {week}', 'data': data,
                     'mask': whole_brain, 'crop': brain_crop(whole_brain)})

    map_grid(rows, IN_VIVO_COLUMNS, suptitle=mouse,
             save_path=c.figures / c.Figure.name(mouse))
