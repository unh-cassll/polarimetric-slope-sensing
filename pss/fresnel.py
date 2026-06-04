"""
Fresnel inversion: build a lookup table mapping DoLP values to angle of
incidence (theta_i) on the rising branch of the Fresnel curve (i.e., up to
Brewster's angle).

This mirrors the MATLAB block:

    s = load('dolp_theta_vecs.mat');
    DOLP_vec = s.DOLP_full;
    theta_vec = s.theta_full;
    ind_max = find(DOLP_vec==max(DOLP_vec),1,'first');
    DOLP_full = linspace(0,1,10000)';
    theta_full = interp1(DOLP_vec(1:ind_max), theta_vec(1:ind_max), DOLP_full, 'pchip');

If the user has a real `.mat` file with the calibrated table, they can pass it
to `load_lookup_table`; otherwise we generate the table from the closed-form
Fresnel relation using `build_lookup_table`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


def fresnel_dolp(theta_deg: np.ndarray | float, n_water: float = 1.34) -> np.ndarray:
    """DoLP as a function of incidence angle, ideal Fresnel (unpolarized sky)."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    s = np.sin(th)
    c = np.cos(th)
    s2 = s * s
    n2 = n_water * n_water
    num = 2.0 * s2 * c * np.sqrt(n2 - s2)
    den = n2 - s2 - n2 * s2 + 2.0 * s2 * s2
    return num / den


def build_lookup_table(
    n_water: float = 1.34, n_points: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (DoLP, theta_i) lookup table on the rising Fresnel branch.

    Returns
    -------
    DOLP_full : ndarray, shape (n_points,)
        Monotonic DoLP grid on [0, 1].
    theta_full : ndarray, shape (n_points,)
        Incidence angle in degrees corresponding to each DoLP via PCHIP
        interpolation along the rising branch of the Fresnel curve.
    """
    theta_dense = np.linspace(0.0, 89.99, 200_000)
    dolp_dense = fresnel_dolp(theta_dense, n_water=n_water)
    # Rising branch only: from theta=0 up to the peak.
    i_peak = int(np.argmax(dolp_dense))
    theta_rise = theta_dense[: i_peak + 1]
    dolp_rise  = dolp_dense[: i_peak + 1]
    # PCHIP requires strictly increasing x. The rising branch is monotonic, but
    # the peak may have repeated values within floating-point precision; trim.
    keep = np.concatenate([[True], np.diff(dolp_rise) > 0])
    dolp_rise = dolp_rise[keep]
    theta_rise = theta_rise[keep]

    DOLP_full = np.linspace(0.0, 1.0, n_points)
    interp = PchipInterpolator(dolp_rise, theta_rise, extrapolate=False)
    theta_full = interp(DOLP_full)
    # Beyond the curve's maximum DoLP, interpolation returns NaN. Clamp those
    # to the peak's theta value so the index lookup behaves gracefully.
    peak_dolp = dolp_rise[-1]
    peak_theta = theta_rise[-1]
    theta_full = np.where(DOLP_full > peak_dolp, peak_theta, theta_full)
    # And at DoLP == 0 exactly, theta = 0.
    theta_full[0] = 0.0
    return DOLP_full, theta_full


def load_lookup_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a (DOLP_full, theta_full) table from a .mat file.

    The .mat file is expected to contain variables named 'DOLP_full' and
    'theta_full' matching the MATLAB convention.
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if "DOLP_full" not in mat or "theta_full" not in mat:
        # Some saved tables use the names with a different case or the
        # pre-resampled vectors. Try the alternates.
        if "DOLP_vec" in mat and "theta_vec" in mat:
            dolp_vec = np.asarray(mat["DOLP_vec"]).squeeze()
            theta_vec = np.asarray(mat["theta_vec"]).squeeze()
            i_peak = int(np.argmax(dolp_vec))
            DOLP_full = np.linspace(0.0, 1.0, 10_000)
            interp = PchipInterpolator(dolp_vec[: i_peak + 1], theta_vec[: i_peak + 1],
                                       extrapolate=False)
            theta_full = interp(DOLP_full)
            theta_full = np.nan_to_num(theta_full, nan=theta_vec[i_peak])
            theta_full[0] = 0.0
            return DOLP_full, theta_full
        raise KeyError(
            f"{path}: expected variables 'DOLP_full' and 'theta_full' "
            "(or 'DOLP_vec' and 'theta_vec') in the .mat file"
        )
    return (np.asarray(mat["DOLP_full"]).squeeze(),
            np.asarray(mat["theta_full"]).squeeze())


def dolp_to_aoi(
    dolp: np.ndarray, DOLP_full: np.ndarray, theta_full: np.ndarray
) -> np.ndarray:
    """Map a DoLP field to angle-of-incidence (degrees) via index lookup.

    Reproduces the MATLAB:
        DOLP_int = floor(DOLP*10000);
        DOLP_int(DOLP_int<1) = 1; DOLP_int(DOLP_int>10000) = 10000;
        AOI = theta_full(DOLP_int);
    """
    n = len(DOLP_full)
    idx = np.floor(np.asarray(dolp, dtype=np.float64) * n).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    return theta_full[idx]
