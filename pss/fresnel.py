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


def lut_from_curve(
    theta: np.ndarray, curve: np.ndarray, n_points: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Build an EMPIRICAL (DoLP, theta_i) lookup table from a measured curve.

    Given a measured DoLP-vs-incidence profile -- e.g. the per-row DoLP(theta)
    of a wide-FOV camera, which spans a large range of incidence angles in one
    frame -- reduce it to the strictly-increasing rising branch (up to the DoLP
    peak / Brewster's angle) and resample onto the same (DOLP_full, theta_full)
    grid that `build_lookup_table` produces, so the result drops straight into
    `compute_slope_field(lookup_table=...)` and `dolp_to_aoi`.

    The reduction (sort by theta, truncate at the DoLP peak, keep strictly
    increasing DoLP samples, PCHIP, clamp beyond the peak, force theta=0 at
    DoLP=0) mirrors the ideal-Fresnel table build; it is the fragile step, so it
    is deliberately faithful to the validated prototype.

    Parameters
    ----------
    theta : ndarray
        Incidence angle samples (deg). NaNs are dropped.
    curve : ndarray
        Measured DoLP at each `theta`. NaNs are dropped.
    n_points : int
        Size of the output DoLP grid (default 10000, matching the MATLAB).

    Returns
    -------
    DOLP_full : ndarray, shape (n_points,)
        Monotonic DoLP grid on [0, 1].
    theta_full : ndarray, shape (n_points,)
        Incidence angle (deg) for each DoLP via PCHIP along the rising branch.
    """
    theta = np.asarray(theta, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    m = np.isfinite(theta) & np.isfinite(curve)
    th, dl = theta[m], curve[m]
    if th.size < 2:
        raise ValueError("lut_from_curve needs at least two finite samples")
    o = np.argsort(th)
    th, dl = th[o], dl[o]
    # Rising branch only: from the start up to the DoLP peak.
    ipk = int(np.argmax(dl))
    th, dl = th[: ipk + 1], dl[: ipk + 1]
    # Keep strictly-increasing DoLP (PCHIP requires strictly increasing x).
    runmax = np.maximum.accumulate(np.concatenate([[-np.inf], dl[:-1]]))
    keep = dl > runmax + 1e-6
    keep[np.argmax(dl > 0)] = True
    th, dl = th[keep], dl[keep]
    if dl.size < 2:
        raise ValueError(
            "lut_from_curve found no usable rising branch (the DoLP-vs-theta "
            "curve does not increase with incidence). Common cause: the wrong "
            "`row_sign` in the wide-FOV geometry, which inverts the AOI-vs-row "
            "mapping so the measured curve falls instead of rising.")

    DOLP_full = np.linspace(0.0, 1.0, n_points)
    tf = PchipInterpolator(dl, th, extrapolate=False)(DOLP_full)
    # Clamp outside the MEASURED DoLP support: above the peak -> the peak's
    # theta; below the smallest measured DoLP -> its (small-angle) theta. The
    # two ends are filled separately -- filling both with the peak (as a single
    # nan fill would) injects a spurious large angle at low DoLP and breaks
    # monotonicity when the measured curve does not reach DoLP ~ 0.
    tf = np.where(DOLP_full > dl[-1], th[-1], tf)
    tf = np.where(DOLP_full < dl[0], th[0], tf)
    tf[0] = 0.0
    return DOLP_full, np.nan_to_num(tf, nan=th[-1])


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
            dl = dolp_vec[: i_peak + 1]
            th = theta_vec[: i_peak + 1]
            # PCHIP requires strictly increasing x; drop repeated DoLP samples.
            keep = np.concatenate([[True], np.diff(dl) > 0])
            dl, th = dl[keep], th[keep]
            DOLP_full = np.linspace(0.0, 1.0, 10_000)
            theta_full = PchipInterpolator(dl, th, extrapolate=False)(DOLP_full)
            # Clamp outside the measured DoLP support, both ends separately
            # (as in lut_from_curve): above the peak -> peak theta; below the
            # smallest measured DoLP -> its small-angle theta.
            theta_full = np.where(DOLP_full > dl[-1], th[-1], theta_full)
            theta_full = np.where(DOLP_full < dl[0], th[0], theta_full)
            theta_full[0] = 0.0
            return DOLP_full, np.nan_to_num(theta_full, nan=th[-1])
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

    Follows the MATLAB scheme:
        DOLP_int = floor(DOLP*10000);
        DOLP_int(DOLP_int<1) = 1; DOLP_int(DOLP_int>10000) = 10000;
        AOI = theta_full(DOLP_int);
    except that the 0-based index floor(d*n) is used directly (the exact
    1-based translation would be floor(d*n) - 1); the two differ by one
    table bin, ~0.005 deg, and this form is the less-biased of the two.

    Non-finite DoLP maps to NaN (floor(NaN).astype(int) is undefined and
    would otherwise silently index theta=0).
    """
    n = len(DOLP_full)
    dolp = np.asarray(dolp, dtype=np.float64)
    finite = np.isfinite(dolp)
    idx = np.floor(np.where(finite, dolp, 0.0) * n).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    aoi = np.asarray(theta_full[idx], dtype=np.float64)
    return np.where(finite, aoi, np.nan)
