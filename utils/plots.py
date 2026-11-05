"""Rendering quantitative parameter maps.

Two layouts cover both studies:

    map_row    one row of maps, each with its own colorbar — used for the
               phantom tubes, where every panel has a different range
    map_grid   a grid with one colorbar per column beneath the last row — used
               for the per-timepoint mouse figure, where a column shares a range

Both build on :func:`draw_map`, which renders a single masked panel.

The two black-background colormaps (`b_viridis`, `b_winter`) were previously
imported from the external `molecular-mrf` repository and are defined here
instead, so this package has no external repository dependency.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Style shared by both figures.
FIGURE_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def apply_style():
    plt.style.use('default')
    plt.rcParams.update(FIGURE_STYLE)


def _black_background_cmap(name):
    """Copy a named colormap, forcing its lowest value to black.

    Used so masked-out (zero) background pixels render black rather than the
    colormap's own minimum color.
    """
    base = plt.get_cmap(name)
    colors = base(np.arange(base.N))
    colors[0, 0:3] = 0
    return mcolors.LinearSegmentedColormap.from_list(f'b_{name}', colors)


b_viridis = _black_background_cmap('viridis')
b_winter = _black_background_cmap('winter')


def masked_cmap(base, bad_color='#eeeeee'):
    """A copy of `base` that renders NaN (outside the ROI) as `bad_color`."""
    cmap = base if hasattr(base, 'copy') else plt.get_cmap(base)
    cmap = cmap.copy()
    cmap.set_bad(color=bad_color)
    return cmap


def draw_map(ax, data, mask=None, cmap=None, clim=None, title=None,
             outside='black'):
    """Render one parameter map.

    :param data: 2-D array
    :param mask: optional ROI; pixels outside it are suppressed
    :param outside: 'black' multiplies by the mask, so background falls to the
                    colormap's black floor; 'blank' sets it to NaN so the
                    colormap's bad color shows instead
    :return: the image handle, for attaching a colorbar
    """
    if mask is not None:
        data = np.where(mask, data, np.nan) if outside == 'blank' else data * mask

    vmin, vmax = (None, None) if clim is None else clim
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    if title:
        ax.set_title(title, pad=9, fontsize=12)
    ax.axis('off')
    return im


def map_row(panels, suptitle=None, save_path=None, figsize=None):
    """One row of maps, each with its own colorbar on the right.

    :param panels: list of dicts, each with 'data' and optionally 'mask',
                   'cmap', 'clim', 'title', 'outside', 'colorbar' (default True)
                   and 'annotate' (a callable given the axis, for extra labels)
    :param save_path: if given, saved here as both .pdf and .png
    :return: the figure
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4.25 * n, 4.5))
    fig.subplots_adjust(wspace=0.18)
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        im = draw_map(ax, panel['data'], panel.get('mask'),
                      cmap=panel.get('cmap'), clim=panel.get('clim'),
                      title=panel.get('title'),
                      outside=panel.get('outside', 'blank'))
        if panel.get('colorbar', True):
            cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.07)
            fig.colorbar(im, cax=cax).ax.tick_params(labelsize=9)
        if panel.get('annotate'):
            panel['annotate'](ax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    _save(fig, save_path)
    return fig


def map_grid(rows, columns, suptitle=None, save_path=None):
    """A grid of maps: one row per acquisition, one column per contrast.

    Each column gets a single horizontal colorbar beneath the bottom row, since
    a column shares one display range.

    :param rows: list of dicts, each with
                 'label' (shown to the left of the row),
                 'data'  (dict keyed by column 'key'),
                 'mask'  (applied to every panel in the row),
                 'crop'  (slice applied to data and mask)
    :param columns: list of dicts, each with 'key', 'label', 'cmap', 'clim' and
                    optionally 'ticks'
    :param save_path: if given, saved here as both .pdf and .png
    :return: the figure
    """
    n_rows, n_cols = len(rows), len(columns)

    # Size each panel from the cropped map's aspect ratio, so panels are not
    # letterboxed.
    sample = rows[0]['data'][columns[0]['key']][rows[0]['crop']]
    img_h, img_w = sample.shape
    side = 5
    panel_w, panel_h = ((side * img_w / img_h, side) if img_h >= img_w
                        else (side, side * img_h / img_w))

    # A dedicated short row for the colorbars keeps them from bleeding into the
    # bottom row of images.
    cbar_h = panel_h * 0.16
    height_ratios = [panel_h] * n_rows + [cbar_h]

    fig = plt.figure(figsize=(panel_w * n_cols, sum(height_ratios)))
    gs = fig.add_gridspec(len(height_ratios), n_cols,
                          height_ratios=height_ratios, wspace=0.05, hspace=0.1)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.03)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.97)

    axs = np.empty((n_rows, n_cols), dtype=object)
    for j, col in enumerate(columns):
        for i, row in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax
            draw_map(ax, row['data'][col['key']][row['crop']],
                     row['mask'][row['crop']],
                     cmap=col['cmap'], clim=col['clim'])

    # Column headers, placed before the colorbars so positions are still exact.
    for j, col in enumerate(columns):
        box = axs[0, j].get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + 0.03, col['label'],
                 ha='center', va='bottom', fontsize=18, fontweight='bold')

    for i, row in enumerate(rows):
        box = axs[i, 0].get_position()
        fig.text(box.x0 - 0.03, (box.y0 + box.y1) / 2, row['label'],
                 ha='right', va='center', fontsize=16, rotation=90)

    for j, col in enumerate(columns):
        cax = fig.add_subplot(gs[n_rows, j])
        # Shrink within the cell so neighbouring tick labels do not collide.
        pos = cax.get_position()
        inset = 0.04
        cax.set_position([pos.x0 + inset * pos.width, pos.y0,
                          pos.width * (1 - 2 * inset), pos.height])
        sm = ScalarMappable(norm=Normalize(vmin=col['clim'][0], vmax=col['clim'][1]),
                            cmap=col['cmap'])
        cb = fig.colorbar(sm, cax=cax, orientation='horizontal',
                          ticks=col.get('ticks'))
        cb.ax.tick_params(labelsize=16)

    _save(fig, save_path)
    return fig


def brain_crop(mask, margin_frac=0.15):
    """Slice cropping to a mask's bounding box, padded proportionally to it."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad_r = int(round((rmax - rmin) * margin_frac))
    pad_c = int(round((cmax - cmin) * margin_frac))
    return np.s_[
        max(0, rmin - pad_r): min(mask.shape[0], rmax + pad_r),
        max(0, cmin - pad_c): min(mask.shape[1], cmax + pad_c),
    ]


def _save(fig, save_path):
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(save_path.with_suffix(f'.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {save_path.with_suffix(".pdf")} (+ .png)')
