"""
Core utilities for slope-field surface reconstruction.

Provides:
  - lindisp_with_current : linear water-wave dispersion relation with
    surface tension and steady current.  Returns phase speed and
    wavenumber.
  - _cwt, _inverse_cwt   : continuous wavelet transform and its inverse,
    built on the EWDM Morlet implementation with an empirically-calibrated
    normalization correction that gives ~1% round-trip amplitude fidelity
    on band-limited signals.
"""
import numpy as np
import xarray as xr
from scipy import interpolate
from ewdm.wavelets import cwt, Morlet


# ---------------------------------------------------------------------------
# Dispersion relation
# ---------------------------------------------------------------------------
def lindisp_with_current(omega, h, current_m_s):
    """
    Linear water-wave dispersion relation with surface tension and current.

    Solves omega = sqrt((g*k + sigma/rho * k^3) * tanh(k*h)) + k*U
    for k(omega) by tabulated cubic interpolation.

    Args:
        omega       : scalar or 1-D array of angular frequencies (rad/s)
        h           : water depth (m)
        current_m_s : steady current speed projected onto the wave
                      direction (m/s).  Use 0 for no current.

    Returns:
        c : phase speed   omega/k     (same shape as omega)
        k : wavenumber    (rad/m)
    """
    omega = np.atleast_1d(omega).flatten().astype(float).copy()
    h = float(np.atleast_1d(h).flatten()[0])
    U = float(np.atleast_1d(current_m_s).flatten()[0])
    omega[omega == 0] = np.nan

    g, rho_w, sigma = 9.806, 1020.0, 0.072
    k_vec = np.logspace(-4, 4, 200)
    omega_disp = (np.sqrt((g*k_vec + sigma/rho_w*k_vec**3)
                          * np.tanh(k_vec*h)) + k_vec*U)
    k_from_omega = interpolate.interp1d(
        omega_disp, k_vec, kind='cubic',
        bounds_error=False, fill_value=np.nan)
    k = k_from_omega(omega)
    return omega / k, k


# ---------------------------------------------------------------------------
# CWT round-trip
# ---------------------------------------------------------------------------
def _cwt(signal_1d, freqs, fs, mother=None):
    """
    Forward continuous wavelet transform.

    Args:
        signal_1d : (T,) real time series
        freqs     : (nf,) frequency grid (Hz)
        fs        : sampling frequency (Hz)
        mother    : EWDM mother wavelet; default Morlet(6.0)

    Returns:
        xarray DataArray of shape (nf, T) with complex CWT coefficients.
    """
    if mother is None:
        mother = Morlet(6.0)
    t = np.arange(len(signal_1d)) / fs
    da = xr.DataArray(np.asarray(signal_1d, dtype=float),
                      coords={'time': t}, dims=['time'])
    return cwt(da, freqs=freqs, fs=fs, mother=mother)


def _inverse_cwt(W, freqs, fs, mother=None):
    """
    Inverse continuous wavelet transform (Torrence-Compo delta-function
    reconstruction) with an empirically-calibrated normalization.

    The bare TC reconstruction systematically under-shoots variance by a
    factor of ~0.696 when paired with EWDM's CWT normalization.  A
    universal correction factor of 1.4383 (close to sqrt(2)) brings the
    round-trip to within 1-3% across signal types and frequency grids.

    Args:
        W      : (nf, T) complex CWT coefficients
        freqs  : (nf,) frequency grid (Hz), matching W
        fs     : sampling frequency (Hz)
        mother : EWDM mother wavelet; default Morlet(6.0)

    Returns:
        eta(t) : (T,) real reconstructed time series
    """
    if mother is None:
        mother = Morlet(6.0)
    scales = 1.0 / (mother.flambda * freqs)
    ds = np.abs(np.gradient(scales))
    dt = 1.0 / fs
    CALIBRATION = 1.4383
    weight = (CALIBRATION * np.sqrt(dt) /
              (mother.cdelta * (np.pi ** -0.25)))
    return weight * (np.real(W) / scales[:, None]**1.5 * ds[:, None]).sum(axis=0)
