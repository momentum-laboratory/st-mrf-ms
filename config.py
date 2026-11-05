"""Configuration for both studies.

Every path resolves relative to this file, so the pipelines run from any
checkout location with no setup:

    data/<study>/         inputs, as .npy arrays
    trained_nn/<study>/   the trained networks
    results/<study>/      quantitative maps and figures

Each study runs the same two stages, so each has an `MT` and an `NOE` section
carrying that stage's network, inputs and output paths. The two stages were
trained separately and store their normalization bounds in different layouts —
MT under split `min_<name>`/`max_<name>` keys, rNOE as `{key: [min, max]}`
pairs. `inference.read_bounds` accepts either.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR       = ROOT / 'data'
TRAINED_NN_DIR = ROOT / 'trained_nn'
RESULTS_DIR    = ROOT / 'results'


# ===========================================================================
# In vitro — brain-lipid phantoms
# ===========================================================================

class InVitro:
    """Seven tubes of porcine brain lipid extract, scanned once."""

    name = 'in_vitro'
    title = 'In vitro — brain-lipid phantoms'

    data    = DATA_DIR / 'in_vitro'
    nn      = TRAINED_NN_DIR / 'in_vitro'
    results = RESULTS_DIR / 'in_vitro'
    maps    = results / 'maps'
    figures = results / 'figures'

    # Lipid concentration of each tube (% w/v) — the ground truth.
    GROUND_TRUTH_LIPID_PCT = [0, 4.8, 8, 9.6, 12, 16, 24]

    # Inputs. `t1_mean` is the per-tube mean T1 rather than the per-pixel map:
    # pixel-level T1 fitting noise would otherwise propagate into the
    # predictions as spurious f_ss structure (Methods).
    t1_mean    = data / 't1_mean.npy'
    t2         = data / 't2.npy'
    tube_masks = data / 'tube_masks.npy'    # (7, rows, cols), boolean

    class MT:
        """Stage 1 — semi-solid MT (f_ss, k_sw), conditioned on T1 and T2.

        An alternative network trained over an extended T2 range exists in the
        project history and gives r = 0.976; the published r = 0.969 comes from
        this one. Do not swap it without re-checking the correlation.
        """
        weights     = TRAINED_NN_DIR / 'in_vitro' / 'mt.pt'
        mrf         = DATA_DIR / 'in_vitro' / 'mt.npy'
        maps_path   = RESULTS_DIR / 'in_vitro' / 'maps' / 'mt.mat'

        # 32 frames acquired; the first 2 (transient magnetization) are not stored.
        sched_iter  = 30
        input_dim   = sched_iter + 2   # MT MRF frames + T1 + T2
        n_hidden    = 2
        n_neurons   = 300

        minmax_key  = 'min_max_values'
        output_keys = ['fs', 'ksw']

    class NOE:
        """Stage 2 — joint rNOE(-3.5 ppm) and rNOE(-1.6 ppm) pool fractions.

        Conditioned on T1, T2 and the stage-1 MT parameters. The rNOE exchange
        rate is held fixed rather than inferred, so there are two outputs.
        """
        weights     = TRAINED_NN_DIR / 'in_vitro' / 'rnoe.pt'
        noe35       = DATA_DIR / 'in_vitro' / 'noe_3p5.npy'
        noe16       = DATA_DIR / 'in_vitro' / 'noe_1p6.npy'
        maps_path   = RESULTS_DIR / 'in_vitro' / 'maps' / 'rnoe.mat'

        # 32 frames acquired; the first 2 (transient magnetization) are not stored.
        sched_iter  = 30
        input_dim   = 30 * 2 + 4       # both rNOE series + T1, T2, MT f_ss, MT k_sw
        n_hidden    = 4
        n_neurons   = 300

        minmax_key         = 'min_max_values'
        minmax_output_keys = ['noe_minus3p5ppm_fs', 'noe_minus1p6ppm_fs']
        output_keys        = ['noe_minus_3p5ppm_fs', 'noe_minus_1p6ppm_fs']

    class Figure:
        """Pool-fraction maps beside a ground-truth panel.

        The rNOE ceiling is 1.2 rather than 1.0 because the 24 % tube reaches
        about 1.05 % f_s.
        """
        name     = 'phantom_fs_maps_with_gt'
        clim_mt  = (0, 6)         # MT f_ss (%)
        clim_noe = (0.4, 1.2)     # rNOE f_s (%)


# ===========================================================================
# In vivo — cuprizone mice
# ===========================================================================

class InVivo:
    """Twelve mice imaged at four timepoints; see `scans.py` for the cohort.

    Everything belonging to one scan sits in `data/in_vivo/<mouse>/week<N>/`:

        t1.npy, t2.npy             relaxation maps, in milliseconds
        mt.npy                     semi-solid MT ST-MRF series, 30 frames
        noe_3p5.npy, noe_1p6.npy   rNOE ST-MRF series, 31 frames each (M0 + 30)
        brain_mask.npy             whole-brain ROI

    The mask is a hand-drawn annotation rather than acquired data. It lives
    beside the inputs because it is per-scan and means nothing apart from the
    scan it was drawn on.
    """

    name = 'in_vivo'
    title = 'In vivo — cuprizone mice'

    data    = DATA_DIR / 'in_vivo'
    nn      = TRAINED_NN_DIR / 'in_vivo'
    results = RESULTS_DIR / 'in_vivo'
    maps    = results / 'maps'
    figures = results / 'figures'

    # Cuprizone runs weeks 0-7; week 12 follows five weeks of recovery.
    ALL_WEEKS = ['0', '4', '7', '12']

    ROI_FILES = {'whole_brain': 'brain_mask'}

    @staticmethod
    def array_path(mouse, week, name):
        """Path to one array of one scan, e.g. 'mt' or 'brain_mask'."""
        return InVivo.data / mouse / f'week{week}' / f'{name}.npy'

    @staticmethod
    def roi_path(mouse, week, roi_name):
        return InVivo.array_path(mouse, week, InVivo.ROI_FILES[roi_name])

    class MT:
        """Stage 1 — semi-solid MT (f_ss, k_sw), conditioned on T1 and T2."""
        weights     = TRAINED_NN_DIR / 'in_vivo' / 'mt.pt'
        maps_dir    = RESULTS_DIR / 'in_vivo' / 'maps' / 'mt'
        mrf         = 'mt'             # array name within the scan folder

        # 32 frames acquired; the first 2 (transient magnetization) are not stored.
        sched_iter  = 30
        input_dim   = sched_iter + 2   # MT MRF frames + T1 + T2
        n_hidden    = 2
        n_neurons   = 300

        minmax_key  = 'min_max_values'
        output_keys = ['fs', 'ksw']

        @classmethod
        def maps_path(cls, scan_name):
            return cls.maps_dir / f'{scan_name}_mt.mat'

    class NOE:
        """Stage 2 — joint rNOE(-3.5 ppm) and rNOE(-1.6 ppm) pool fractions.

        Conditioned on T1, T2 and the stage-1 MT parameters. The rNOE exchange
        rate is held fixed rather than inferred (Methods), so there are two
        outputs.
        """
        weights     = TRAINED_NN_DIR / 'in_vivo' / 'rnoe.pt'
        maps_dir    = RESULTS_DIR / 'in_vivo' / 'maps' / 'rnoe'
        noe35       = 'noe_3p5'
        noe16       = 'noe_1p6'

        # 32 frames acquired; the first is not stored, leaving an unsaturated
        # (M0) image plus 30 saturated images per pool.
        sched_iter  = 31
        input_dim   = 31 * 2 + 4       # both rNOE series + T1, T2, MT f_ss, MT k_sw
        n_hidden    = 2
        n_neurons   = 300

        minmax_key         = 'min_max_values'
        minmax_output_keys = ['noe_minus3p5ppm_fs', 'noe_minus1p6ppm_fs']
        output_keys        = ['noe_minus_3p5ppm_fs', 'noe_minus_1p6ppm_fs']

        @classmethod
        def maps_path(cls, scan_name):
            return cls.maps_dir / f'{scan_name}_rnoe.mat'

    class Figure:
        """Per-timepoint map grid for one animal."""
        representative_mouse = 'mouse8'

        @staticmethod
        def name(mouse):
            return f'maps_grid_{mouse}_all_contrasts'


# Display names used when reporting pool fractions.
METRIC_DISPLAY_NAMES = {
    'fs':                  'Semi-solid MT $f_{ss}$ (%)',
    'ksw':                 'Semi-solid MT $k_{sw}$ (s$^{-1}$)',
    'mt_fs':               'Semi-solid MT $f_{ss}$ (%)',
    'noe_minus_3p5ppm_fs': 'rNOE(-3.5 ppm) $f_s$ (%)',
    'noe_minus_1p6ppm_fs': 'rNOE(-1.6 ppm) $f_s$ (%)',
}
