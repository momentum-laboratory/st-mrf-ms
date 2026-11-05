"""
"""

import os
import random

import numpy as np
import scipy.io as sio
import torch


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------
#
# Inputs are stored in whichever lossless dtype is smallest for them — the
# ST-MRF series as uint16 (the acquired counts), the relaxation maps as float64,
# the ROI masks as bool. Everything downstream computes in float64.

def load_array(path):
    """Load a stored array as float64."""
    return np.load(path).astype(np.float64)


def load_mask(path):
    """Load an ROI mask as boolean."""
    return np.load(path).astype(bool)


def save_maps(quant_maps, path):
    """Write parameter maps to a .mat file, creating the directory if needed."""
    os.makedirs(path.parent, exist_ok=True)
    sio.savemat(path, quant_maps)


def load_maps(path):
    """Read parameter maps back from a .mat file."""
    return sio.loadmat(path)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_device(verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Device: {device}')
    return device


def set_seed(seed=2024, verbose=False):
    """Seed every RNG that can affect inference, and force deterministic cuDNN."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if verbose:
        print(f'Random seed set as {seed}')


def load_network(checkpoint_file_path, network_cls, device, **network_kwargs):
    """Build a network, load its trained parameters, and put it in eval mode.

    :return: (network, chkpt) where `chkpt` is the loaded checkpoint file, so
             the caller can also read the normalization bounds stored
             alongside the parameters.
    """
    chkpt = torch.load(checkpoint_file_path, map_location='cpu', weights_only=False)
    reco_net = network_cls(**network_kwargs).to(device)
    reco_net.load_state_dict(chkpt['model_state_dict'])
    reco_net.eval()
    return reco_net, chkpt


# ---------------------------------------------------------------------------
# Normalization bounds
# ---------------------------------------------------------------------------

def read_bounds(temp_data, *names):
    """Read the range each network parameter was normalized over during training.

    The two stages were saved with different conventions, so both are accepted:

        MT     min_t1w: 1.3,  max_t1w: 2.3     separate min_/max_ keys
        rNOE   t1: [1.2, 2.3]                  one [min, max] entry

    :param temp_data: the bounds mapping from the saved network (`chkpt['min_max_values']`)
    :param names: parameter names, in the order the network expects them
    :return: (min tensor, max tensor)
    """
    def pair(name):
        if name in temp_data:
            return temp_data[name]
        return temp_data[f'min_{name}'], temp_data[f'max_{name}']

    min_tensor, max_tensor = zip(*(pair(n) for n in names))
    return (torch.tensor(np.array(min_tensor, dtype=np.float64)),
            torch.tensor(np.array(max_tensor, dtype=np.float64)))


def normalize_range(original_array, original_min, original_max, new_min, new_max):
    """Rescale values from [original_min, original_max] into [new_min, new_max]."""
    a, b, c, d = original_min, original_max, new_min, new_max
    return (original_array - a) / (b - a) * (d - c) + c


def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    """Rescale network outputs from [new_min, new_max] back to physical units."""
    a, b, c, d = original_min, original_max, new_min, new_max
    return (normalized_array - c) / (d - c) * (b - a) + a


# ---------------------------------------------------------------------------
# Preparing the network's input
# ---------------------------------------------------------------------------

def normalize_trajectories(acquired_data_orig, sched_iter, device):
    """Turn an ST-MRF series into one L2-normalized trajectory per pixel.

    The network is trained on the *shape* of a trajectory rather than its
    absolute intensity, so each pixel's signal is divided by its own 2-norm.
    Empty background pixels have zero norm and would divide to NaN, so their
    norm is set to 1 (leaving them zero).

    :param acquired_data_orig: (sched_iter, c_acq_data, w_acq_data)
    :return: (c_acq_data, w_acq_data, tensor of shape (c_acq_data*w_acq_data, sched_iter))
    """
    _, c_acq_data, w_acq_data = np.shape(acquired_data_orig)
    acquired_data = np.reshape(acquired_data_orig, (sched_iter, c_acq_data * w_acq_data), order='F')

    norms = np.linalg.norm(acquired_data, axis=0, ord=2)
    norms[norms == 0] = 1.0
    acquired_data = (acquired_data / norms).T

    return c_acq_data, w_acq_data, torch.from_numpy(acquired_data).to(device).float()


def scale_pair(input_data_orig1, input_data_orig2, bounds, c, w, device):
    """Flatten two maps into columns, stack them, and rescale to [0, 1].

    Used for the pairs the networks are conditioned on — (T1, T2) and, for the
    rNOE stage, (MT f_ss, MT k_sw). NaN/inf pixels (e.g. failed fits outside
    the brain mask) are zeroed before reshaping, so they don't propagate into
    the network input.

    :param bounds: (min_tensor, max_tensor) from :func:`read_bounds`
    :return: tensor of shape (c*w, 2)
    """
    min_tensor, max_tensor = bounds

    def column(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.reshape(arr, (1, c * w), order='F').T)

    input_data = torch.hstack((column(input_data_orig1), column(input_data_orig2)))
    return normalize_range(input_data, min_tensor, max_tensor, 0, 1).to(device)


def run(reco_net, acquired_data):
    """Forward pass, with gradients off."""
    with torch.no_grad():
        return reco_net(acquired_data.to(next(reco_net.parameters()).device).float())


def to_maps(prediction, bounds, output_keys, c_acq_data, w_acq_data):
    """Convert a network's 0-1 outputs back into named parameter maps.

    :param bounds: (min_tensor, max_tensor) tensors the outputs were normalized over
    :param output_keys: parameter names, in the network's output order
    :return: dict mapping each key to a (c_acq_data, w_acq_data) map
    """
    min_tensor, max_tensor = bounds
    prediction = un_normalize_range(prediction, min_tensor.to(prediction.device),
                                    max_tensor.to(prediction.device), 0, 1)
    values = prediction.cpu().numpy()
    return {key: np.reshape(values[:, i], (c_acq_data, w_acq_data), order='F')
            for i, key in enumerate(output_keys)}
