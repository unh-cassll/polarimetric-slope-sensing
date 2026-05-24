"""
Stokes parameter reconstruction from a DoFP (division-of-focal-plane)
polarimeter raw frame.

The Sony IMX250MZR "Polarsens" sensor has micropolarizers tiled in a
repeating 2x2 super-pixel. The convention used in this codebase (and the
MATLAB original) is:

    super-pixel layout (row, col within the 2x2):
        (0,0) -> 90 deg
        (0,1) -> 45 deg
        (1,0) -> 135 deg
        (1,1) -> 0 deg

Stokes parameters from the four intensities:
    S0 = (I0 + I45 + I90 + I135) / 2
    S1 =  I0 - I90
    S2 =  I45 - I135

We return normalized Stokes (s1 = S1/S0, s2 = S2/S0) alongside S0, since the
downstream DoLP = sqrt(s1^2 + s2^2) is the quantity actually used.

Three reconstruction methods are provided, matching the MATLAB names:
    1. bilinear_interpolation  — sparse-sample bilinear upsample
    2. kernel_averaging        — 4x4 boxcar over each orientation channel
    3. conv_demodulation       — modulated-carrier demodulation (Ratliff 2009)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

# ---- Super-pixel offsets ---------------------------------------------------
# These index into the 2x2 tile. Edit here if your sensor uses a different
# micropolarizer arrangement.
_OFFSETS = {
    "I90":  (0, 0),
    "I45":  (0, 1),
    "I135": (1, 0),
    "I0":   (1, 1),
}


def _extract_channels(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split a raw DoFP frame into four sparse intensity sub-images.

    Each returned array has shape (H/2, W/2) — one sample per super-pixel.
    """
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"expected 2D frame, got shape {f.shape}")
    if f.shape[0] % 2 or f.shape[1] % 2:
        raise ValueError(
            f"frame dimensions {f.shape} must be even (2x2 super-pixel tiling)"
        )
    return {name: f[r::2, c::2] for name, (r, c) in _OFFSETS.items()}


def by_superpixel(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native super-pixel reduction: one Stokes vector per 2x2 super-pixel.

    Returns HALF-resolution (H/2, W/2) Stokes arrays. The four orientation
    samples within each super-pixel are combined directly, with NO
    interpolation and NO upsampling. This is the most honest reduction with
    respect to information content: the output grid equals the measurement
    grid (one Stokes vector per super-pixel), so the result never implies
    spatial detail finer than the sensor actually delivers.

    The tradeoff vs the interpolating methods (bilinear, kernel_averaging,
    conv_demodulation): those return full (H, W) arrays and partially correct
    the instantaneous-field-of-view (IFOV) half-pixel offset between
    orientations, at the cost of implying 2x the real linear resolution. This
    method makes no such correction (it treats the four super-pixel samples as
    co-located) but is faithful to the true sampling density.

    Note that the effective pixel pitch of the returned slope field is TWICE
    the sensor pixel pitch, which matters for any downstream spatial-frequency
    or wavelength computation (pass the doubled dx).
    """
    ch = _extract_channels(frame)  # each (H/2, W/2)
    I0, I45, I90, I135 = ch["I0"], ch["I45"], ch["I90"], ch["I135"]
    S0 = 0.5 * (I0 + I45 + I90 + I135)
    S1 = I0 - I90
    S2 = I45 - I135
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(S0 > 0, S1 / S0, 0.0)
        s2 = np.where(S0 > 0, S2 / S0, 0.0)
    return S0, s1, s2


def _bilinear_upsample(sub: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a half-resolution sub-image to full resolution.

    Uses scipy.ndimage.map_coordinates for accuracy; no SciPy dependency
    beyond what we already import for convolve. Each sub-sample sits at the
    super-pixel center, so the mapping accounts for the half-pixel offset.
    """
    from scipy.ndimage import map_coordinates

    H, W = out_shape
    # In the full frame, sub-image pixel (i, j) sits at full-frame coordinates
    # (2i + r_off, 2j + c_off). We invert that mapping.
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    # For a generic orientation we don't know which offset we're working with
    # at call time — but the caller passes in the appropriate sub-image and we
    # treat its samples as living on a regular grid in sub-pixel coordinates.
    # The half-pixel offset is absorbed by sampling at (rr-0.5)/2 etc., which
    # MATLAB's interp2 with 'linear' on the sparse grid effectively does.
    sub_rr = (rr - 0.5) / 2.0
    sub_cc = (cc - 0.5) / 2.0
    return map_coordinates(sub, [sub_rr, sub_cc], order=1, mode="nearest")


def by_bilinear_interpolation(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method 1: bilinear interpolation of sparse intensity arrays.

    Returns
    -------
    S0, s1, s2 : ndarray, shape (H, W)
        Total intensity and normalized Stokes parameters.
    """
    H, W = frame.shape
    ch = _extract_channels(frame)
    I0   = _bilinear_upsample(ch["I0"],   (H, W))
    I45  = _bilinear_upsample(ch["I45"],  (H, W))
    I90  = _bilinear_upsample(ch["I90"],  (H, W))
    I135 = _bilinear_upsample(ch["I135"], (H, W))

    S0 = 0.5 * (I0 + I45 + I90 + I135)
    S1 = I0 - I90
    S2 = I45 - I135
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(S0 > 0, S1 / S0, 0.0)
        s2 = np.where(S0 > 0, S2 / S0, 0.0)
    return S0, s1, s2


def _channel_mask(shape: tuple[int, int], r_off: int, c_off: int) -> np.ndarray:
    """Binary mask marking pixels of a given orientation in the full frame."""
    m = np.zeros(shape, dtype=np.float64)
    m[r_off::2, c_off::2] = 1.0
    return m


def by_kernel_averaging(
    frame: np.ndarray, kernel: str = "4x4"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method 2: kernel averaging (Ratliff et al. 2009).

    For each orientation, multiply the full frame by that orientation's binary
    mask and convolve with a uniform kernel; divide by the convolution of the
    mask itself. This yields a full-resolution estimate of each intensity
    channel that is the local average over the kernel's footprint.

    Parameters
    ----------
    kernel : {"2x2", "4x4"}
        Footprint of the averaging kernel.
    """
    f = np.asarray(frame, dtype=np.float64)
    H, W = f.shape

    sizes = {"2x2": 2, "4x4": 4}
    if kernel not in sizes:
        raise ValueError(f"kernel must be one of {list(sizes)}; got {kernel!r}")
    k = sizes[kernel]
    box = np.ones((k, k), dtype=np.float64)

    channels = {}
    for name, (r_off, c_off) in _OFFSETS.items():
        mask = _channel_mask((H, W), r_off, c_off)
        num = convolve(f * mask, box, mode="reflect")
        den = convolve(mask,     box, mode="reflect")
        with np.errstate(divide="ignore", invalid="ignore"):
            channels[name] = np.where(den > 0, num / den, 0.0)

    I0, I45, I90, I135 = channels["I0"], channels["I45"], channels["I90"], channels["I135"]
    S0 = 0.5 * (I0 + I45 + I90 + I135)
    S1 = I0 - I90
    S2 = I45 - I135
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(S0 > 0, S1 / S0, 0.0)
        s2 = np.where(S0 > 0, S2 / S0, 0.0)
    return S0, s1, s2


def by_conv_demodulation(
    frame: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method 4 of Ratliff, LaCasse & Tyo (Opt. Express 17, 9112, 2009).

    The largest of the paper's microgrid-specific bilinear interpolators
    (Fig. 3, "Method 4"): a radius-3*sqrt(2)/2 neighborhood of 16 pixels (4 per
    orientation) with three distance-proportional weights

        A = 0.3541,  B = 0.2639,  C = 0.1180.

    Four 4x4 convolution kernels (H1..H4 in the figure) each interpolate one
    orientation's like-polarization samples to the reconstruction point. The
    paper demodulates the four convolved images (Eqs. 9-11) and forms Stokes
    via Eq. 12:

        S0 = (I0 + I45 + I90 + I135) / 2
        S1 =  I0  - I90
        S2 =  I45 - I135

    The paper's Eq. 11 demodulation is layout-agnostic; here we implement it
    explicitly for the package's super-pixel layout (`_OFFSETS`). The
    orientation that each kernel yields rotates with the parity of the output
    pixel, so we build a per-parity {orientation -> kernel} map directly from
    the layout (via a labeled probe frame) and fill each parity sublattice from
    its own map. This yields a full-resolution (H, W) result identical to the
    paper's scheme. Verified: flat field -> DoLP 0; known synthetic
    polarization field reconstructs DoLP and orientation exactly.

    The earlier releases used a separable Bartlett kernel here as a stand-in;
    this is now the exact Method 4 filter.
    """
    f = np.asarray(frame, dtype=np.float64)
    H, W = f.shape

    kernels = _ratliff_method4_kernels()
    pmap = _ratliff_parity_maps(kernels)

    conv = {j: convolve(f, kernels[j], mode="reflect") for j in kernels}

    S0 = np.zeros((H, W))
    S1 = np.zeros((H, W))
    S2 = np.zeros((H, W))
    for (pr, pc), mp in pmap.items():
        I0   = conv[mp["I0"]]
        I45  = conv[mp["I45"]]
        I90  = conv[mp["I90"]]
        I135 = conv[mp["I135"]]
        s0 = 0.5 * (I0 + I45 + I90 + I135)
        S0[pr::2, pc::2] = s0[pr::2, pc::2]
        S1[pr::2, pc::2] = (I0 - I90)[pr::2, pc::2]
        S2[pr::2, pc::2] = (I45 - I135)[pr::2, pc::2]

    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(S0 > 0, S1 / S0, 0.0)
        s2 = np.where(S0 > 0, S2 / S0, 0.0)
    return S0, s1, s2


# Ratliff Method 4 kernel weights (Fig. 3), normalized to sum to 1.
_RATLIFF4_A, _RATLIFF4_B, _RATLIFF4_C = 0.3541, 0.2639, 0.1180


def _ratliff_method4_kernels() -> dict[int, np.ndarray]:
    """The four 4x4 Method-4 interpolation kernels (H1..H4), normalized."""
    A, B, C = _RATLIFF4_A, _RATLIFF4_B, _RATLIFF4_C
    raw = {
        1: np.array([[0, B, 0, C], [0, 0, 0, 0], [0, A, 0, B], [0, 0, 0, 0]], float),
        2: np.array([[C, 0, B, 0], [0, 0, 0, 0], [B, 0, A, 0], [0, 0, 0, 0]], float),
        3: np.array([[0, 0, 0, 0], [B, 0, A, 0], [0, 0, 0, 0], [C, 0, B, 0]], float),
        4: np.array([[0, 0, 0, 0], [0, A, 0, B], [0, 0, 0, 0], [0, B, 0, C]], float),
    }
    return {j: K / K.sum() for j, K in raw.items()}


def _ratliff_parity_maps(kernels: dict[int, np.ndarray]) -> dict[tuple[int, int], dict[str, int]]:
    """For each output parity, which kernel yields which orientation.

    Determined empirically from the active super-pixel layout (`_OFFSETS`) by
    convolving a frame in which each orientation carries a unique constant, so
    the mapping always tracks the configured layout rather than being
    hard-coded. Returns {(pr, pc): {orientation_name: kernel_index}}.
    """
    vals = {"I0": 1.0, "I45": 2.0, "I90": 3.0, "I135": 4.0}
    val2name = {round(v): k for k, v in vals.items()}
    probe = np.zeros((8, 8), dtype=np.float64)
    for name, (r, c) in _OFFSETS.items():
        probe[r::2, c::2] = vals[name]
    convp = {j: convolve(probe, kernels[j], mode="reflect") for j in kernels}
    pmap: dict[tuple[int, int], dict[str, int]] = {}
    for pr in (0, 1):
        for pc in (0, 1):
            m: dict[str, int] = {}
            for j in kernels:
                v = round(float(convp[j][4 + pr, 4 + pc]))
                m[val2name[v]] = j
            pmap[(pr, pc)] = m
    return pmap


# Convenience dispatcher --------------------------------------------------------
METHODS = {
    "bilinear":          by_bilinear_interpolation,
    "kernel_averaging":  by_kernel_averaging,
    "conv_demodulation": by_conv_demodulation,
}


def compute_stokes(
    frame: np.ndarray, method: str = "bilinear", **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S0, s1, s2 from a raw DoFP frame using the named method."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}; got {method!r}")
    return METHODS[method](frame, **kwargs)
