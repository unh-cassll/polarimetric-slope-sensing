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


# Per-frequency reconstruction-gain calibration (opt-in; see _inverse_cwt).
# Keyed on (freqs, fs, T, mother) so the sinusoid sweep runs once per config.
_PER_SCALE_GAIN_CACHE = {}


def _mother_key(mother):
    """Hashable identity for the mother wavelet (Morlet(6.0) by default)."""
    if mother is None:
        return ("Morlet", 6.0)
    w0 = getattr(mother, "f0", getattr(mother, "omega0", 6.0))
    return (type(mother).__name__, float(w0))


def _fill_nonfinite_nearest(a):
    """Replace non-finite entries by linear interpolation over finite ones."""
    a = np.asarray(a, dtype=float)
    good = np.isfinite(a)
    if good.all():
        return a
    if not good.any():
        return np.ones_like(a)
    idx = np.arange(a.size)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def _bare_inverse(W, freqs, fs, mother):
    """Torrence-Compo delta reconstruction with NO amplitude calibration.

    Identical to the production inverse but with the leading constant set to
    1.0, so a residual per-scale gain can be measured and inverted.
    """
    if mother is None:
        mother = Morlet(6.0)
    scales = 1.0 / (mother.flambda * freqs)
    ds = np.abs(np.gradient(scales))
    dt = 1.0 / fs
    weight = np.sqrt(dt) / (mother.cdelta * (np.pi ** -0.25))
    return weight * (np.real(W) / scales[:, None] ** 1.5 * ds[:, None]).sum(axis=0)


def _per_scale_gain(freqs, fs, T, mother):
    """Per-frequency correction so a unit tone reconstructs at unit amplitude.

    For each f in `freqs`, synthesise a unit cosine of length T, take the
    forward CWT on the same grid, run the un-calibrated inverse, and measure
    the central-window amplitude gain G_bare(f).  The applied correction is
    g(f) = 1 / G_bare(f).  Phase- and amplitude-independent, so one cosine
    probe per frequency suffices.  Depends only on (freqs, fs, T, mother);
    result is cached.
    """
    freqs = np.asarray(freqs, dtype=float)
    key = (tuple(np.round(freqs, 10)), float(fs), int(T), _mother_key(mother))
    cached = _PER_SCALE_GAIN_CACHE.get(key)
    if cached is not None:
        return cached

    if mother is None:
        mother = Morlet(6.0)
    t = np.arange(T) / fs
    c = slice(int(0.2 * T), int(0.8 * T))      # central, cone-of-influence-light
    gains = np.empty(freqs.size, dtype=float)
    for i, f0 in enumerate(freqs):
        ref = np.cos(2.0 * np.pi * f0 * t)
        ref = ref - ref.mean()
        da = xr.DataArray(ref, coords={"time": t}, dims=["time"])
        W = cwt(da, freqs=freqs, fs=fs, mother=mother).values
        rec = _bare_inverse(W, freqs, fs, mother)
        denom = np.std(ref[c])
        gains[i] = (np.std(rec[c]) / denom) if denom > 0 else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(np.isfinite(gains) & (gains > 1e-6), 1.0 / gains, np.nan)
    corr = _fill_nonfinite_nearest(corr)
    corr = np.clip(corr, 0.2, 5.0)             # guard pathological band edges
    _PER_SCALE_GAIN_CACHE[key] = corr
    return corr


def _inverse_cwt(W, freqs, fs, mother=None, per_scale=False):
    """
    Inverse continuous wavelet transform (Torrence-Compo delta-function
    reconstruction).

    Two normalization modes:

    - per_scale=False (default): the original behaviour.  The bare TC
      reconstruction under-shoots variance by ~0.696 with EWDM's CWT
      normalization, so a single universal constant 1.4383 (~sqrt(2)) is
      applied, giving 1-3% round-trip fidelity across grids.

    - per_scale=True: replace the single constant with a per-frequency
      reconstruction gain calibrated, at call time, by round-tripping unit
      reference sinusoids on this (freqs, fs, mother) grid (see
      _per_scale_gain).  Parameter-free; flattens the round-trip to ~1% in
      the well-resolved interior of the band and removes the magic constant.

    Args:
        W         : (nf, T) complex CWT coefficients
        freqs     : (nf,) frequency grid (Hz), matching W
        fs        : sampling frequency (Hz)
        mother    : EWDM mother wavelet; default Morlet(6.0)
        per_scale : use the per-frequency calibration instead of 1.4383.

    Returns:
        eta(t) : (T,) real reconstructed time series
    """
    freqs = np.asarray(freqs, dtype=float)

    if not per_scale:
        if mother is None:
            mother = Morlet(6.0)
        scales = 1.0 / (mother.flambda * freqs)
        ds = np.abs(np.gradient(scales))
        dt = 1.0 / fs
        CALIBRATION = 1.4383
        weight = (CALIBRATION * np.sqrt(dt) /
                  (mother.cdelta * (np.pi ** -0.25)))
        return weight * (np.real(W) / scales[:, None]**1.5 * ds[:, None]).sum(axis=0)

    W = np.asarray(W)
    g = _per_scale_gain(freqs, fs, W.shape[1], mother)
    return _bare_inverse(W * g[:, None], freqs, fs, mother)


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
