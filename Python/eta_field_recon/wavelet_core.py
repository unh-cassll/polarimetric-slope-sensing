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
  - krogstad_eta_coeffs  : Krogstad signed-projection of the two slope CWT
    coefficient arrays onto the surface-elevation CWT coefficients, via the
    dispersion-relation wavenumber. This is the shared core of the long-wave
    inversion, used by both eta_field_recon.recon.reconstruct_eta_field and
    examples._data.mean_wave_timeseries.
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


# ---------------------------------------------------------------------------
# Krogstad signed-projection: slope CWT coefficients -> elevation CWT coeffs
# ---------------------------------------------------------------------------
def krogstad_eta_coeffs(Wsx, Wsy, k_disp):
    """Project slope-CWT coefficients onto elevation-CWT coefficients.

    Implements the Krogstad signed projection used by the long-wave (mean-
    wave) inversion. Given the continuous-wavelet transforms of the cross-look
    and along-look spatial-mean slopes (Wsx, Wsy) and the dispersion-relation
    wavenumber k(omega) on the same frequency grid, returns the elevation
    wavelet coefficients W_eta together with the per-(f, t) direction cosines.

    The projection direction at each (frequency, time) is

        cos_th = |Wsx| / sqrt(|Wsx|^2 + |Wsy|^2)
        sin_th = |Wsy| / sqrt(|Wsx|^2 + |Wsy|^2)

    with the sign of sin_th recovered from sign(Re(Wsy * conj(Wsx))). Using the
    relative phase between Wsy and Wsx this way resolves the 180-degree
    ambiguity that slope magnitude alone cannot.

    Guard: np.sign(0) == 0, which would ZERO sin_th whenever Wsx (or the
    relative phase) vanishes -- e.g. a wave traveling exactly along the
    along-look axis, where Wsx == 0. That would incorrectly annihilate the
    signal. When the relative phase is indeterminate there is no information
    to flip the sign, so it defaults to +1 (keep the magnitude) rather than 0
    (destroy it).

    The elevation coefficient is the slope projection divided by the
    wavenumber (slope = d(eta)/dx -> in the wavelet domain a factor of i*k):

        W_eta = 1j * (cos_th * Wsx + sin_th * Wsy) / k

    Non-finite entries (e.g. from k = NaN at omega = 0) are set to 0.

    Args:
        Wsx, Wsy : (nf, T) complex CWT coefficients of the cross-look and
                   along-look spatial-mean slopes.
        k_disp   : (nf,) dispersion-relation wavenumber (rad/m) on the CWT
                   frequency grid. Broadcast over the time axis.

    Returns:
        W_eta  : (nf, T) complex elevation CWT coefficients.
        cos_th : (nf, T) real direction cosine (cross-look projection).
        sin_th : (nf, T) real signed direction sine (along-look projection).
    """
    eps = 1e-30
    mag = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + eps
    rel_sign = np.sign(np.real(Wsy * np.conj(Wsx)))
    rel_sign = np.where(rel_sign == 0, 1.0, rel_sign)   # indeterminate -> +1
    cos_th = np.abs(Wsx) / mag
    sin_th = (np.abs(Wsy) / mag) * rel_sign
    # k may carry NaN (omega = 0) or inf; the division is expected to produce
    # non-finite entries there, which we immediately zero out. Silence the
    # warning since this is by-design, not an error.
    with np.errstate(divide="ignore", invalid="ignore"):
        W_eta = 1j * (cos_th * Wsx + sin_th * Wsy) / k_disp[:, None]
    W_eta = np.where(np.isfinite(W_eta), W_eta, 0.0)
    return W_eta, cos_th, sin_th
