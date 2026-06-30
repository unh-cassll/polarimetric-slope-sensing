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
  - aperture_transfer_function / aperture_transfer_gain : circular/square
    aperture amplitude transfer H(k) and its inverse 1/H, undoing the spatial
    low-pass of averaging slope over a finite footprint.
"""
import warnings

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.signal.windows import tukey
from scipy.special import j1
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
        c : phase speed   omega/k     (1-D, on the flattened omega)
        k : wavenumber    (rad/m)     (1-D, on the flattened omega)
    """
    omega = np.atleast_1d(omega).flatten().astype(float).copy()
    h = float(np.atleast_1d(h).flatten()[0])
    U = float(np.atleast_1d(current_m_s).flatten()[0])
    omega[omega == 0] = np.nan

    g, rho_w, sigma = 9.806, 1020.0, 0.072
    k_vec = np.logspace(-4, 4, 200)
    omega_disp = (np.sqrt((g*k_vec + sigma/rho_w*k_vec**3)
                          * np.tanh(k_vec*h)) + k_vec*U)
    # An opposing current (U<0) makes omega(k) non-monotonic: it rises to a
    # wave-blocking maximum, then falls. interp1d on a non-monotonic abscissa is
    # undefined, so keep the leading increasing branch and warn; omega above the
    # blocking maximum returns NaN k.
    inc = np.diff(omega_disp) > 0
    if not inc.all():
        cut = int(np.argmin(inc)) + 1
        warnings.warn(
            f"dispersion omega(k) non-monotonic (opposing current U={U:.2f} m/s "
            f"blocks waves above {omega_disp[:cut].max():.3f} rad/s); using the "
            f"increasing branch, higher omega -> NaN k.", UserWarning, stacklevel=2)
        k_vec = k_vec[:cut]
        omega_disp = omega_disp[:cut]
    if k_vec.size < 2:
        # Supercritical opposing current: no increasing branch to invert.
        return omega / np.nan, np.full_like(omega, np.nan)
    k_from_omega = interpolate.interp1d(
        omega_disp, k_vec, kind='cubic' if k_vec.size >= 4 else 'linear',
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


_CDELTA_WARNED = set()


def _mother_cdelta(mother):
    """Positive TC reconstruction constant. ewdm tabulates cdelta only for
    Morlet(6) (=0.776) and returns a -1 sentinel otherwise; passed through it
    sign-flips the inverse. Detect it, warn once, return a sign-safe magnitude
    (amplitude is then recovered by the per-scale calibration)."""
    if mother is None:
        mother = Morlet(6.0)
    cd = mother.cdelta
    if cd is not None and cd > 0:
        return cd
    key = _mother_key(mother)
    if key not in _CDELTA_WARNED:
        warnings.warn(
            f"ewdm has no tabulated cdelta for {key[0]}({key[1]}); using a "
            f"sign-safe fallback. Use per_scale=True for calibrated amplitude.",
            UserWarning, stacklevel=2)
        _CDELTA_WARNED.add(key)
    return abs(cd) if (cd is not None and cd != 0) else 1.0


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
    weight = np.sqrt(dt) / (_mother_cdelta(mother) * (np.pi ** -0.25))
    return weight * (np.real(W) / scales[:, None] ** 1.5 * ds[:, None]).sum(axis=0)


_GAIN_LO, _GAIN_HI = 0.2, 5.0       # trusted correction-magnitude band


def _per_scale_gain(freqs, fs, T, mother, coi_frac=0.5):
    """Per-frequency correction so a unit tone reconstructs at unit amplitude.

    For each f in `freqs`, synthesize a unit cosine of length T, take the
    forward CWT on the same grid, run the un-calibrated inverse, and measure
    the central-window signed gain G_bare(f).  The applied correction is
    g(f) = 1 / G_bare(f).  Depends only on (freqs, fs, T, mother, coi_frac);
    result is cached.

    Cone-of-influence guard: a band is uncalibratable and returns NaN (with a
    warning) -- never a silently clamped value -- when the wavelet e-folding
    time (~sqrt(2)*scale) exceeds coi_frac of the record, when fewer than two
    scales leave the inverse spacing undefined, or when the measured gain is
    non-finite or lands outside the trusted [_GAIN_LO, _GAIN_HI] magnitude band.
    """
    freqs = np.asarray(freqs, dtype=float)
    key = (tuple(np.round(freqs, 10)), float(fs), int(T),
           _mother_key(mother), float(coi_frac))
    cached = _PER_SCALE_GAIN_CACHE.get(key)
    if cached is not None:
        return cached

    if mother is None:
        mother = Morlet(6.0)

    # Usability mask: wavelet must fit the record, and >=2 scales are needed
    # to form the inverse's scale spacing.
    scales = 1.0 / (mother.flambda * freqs)
    usable = (np.sqrt(2.0) * scales) <= coi_frac * (T / fs)
    if freqs.size < 2:
        usable[:] = False

    t = np.arange(T) / fs
    c = slice(int(0.2 * T), int(0.8 * T))      # central, cone-of-influence-light
    gains = np.full(freqs.size, np.nan)
    for i in np.flatnonzero(usable):
        ref = np.cos(2.0 * np.pi * freqs[i] * t)
        ref = ref - ref.mean()
        da = xr.DataArray(ref, coords={"time": t}, dims=["time"])
        W = cwt(da, freqs=freqs, fs=fs, mother=mother).values
        rec = _bare_inverse(W, freqs, fs, mother)
        # Signed projection of rec onto ref recovers sign as well as magnitude,
        # so a sign-flipped inverse is corrected rather than left inverted.
        denom = float(np.dot(ref[c], ref[c]))
        gains[i] = (float(np.dot(rec[c], ref[c])) / denom) if denom > 0 else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(np.isfinite(gains) & (np.abs(gains) > 1e-6),
                        1.0 / gains, np.nan)
    # Out-of-range corrections are flagged NaN rather than clamped: clamping
    # would silently pass off a pathological band as a recovered one.
    corr = np.where(np.abs(corr) <= _GAIN_HI, corr, np.nan)
    corr = np.where(np.abs(corr) >= _GAIN_LO, corr, np.nan)

    n_bad = int(np.sum(~np.isfinite(corr)))
    if n_bad:
        warnings.warn(
            f"{n_bad}/{freqs.size} CWT band(s) uncalibratable (cone-of-influence "
            f"or out-of-range gain); returning NaN there instead of clamping.",
            UserWarning, stacklevel=2)
    _PER_SCALE_GAIN_CACHE[key] = corr
    return corr


def _inverse_cwt(W, freqs, fs, mother=None, per_scale=False):
    """
    Inverse continuous wavelet transform (Torrence-Compo delta-function
    reconstruction).

    Two normalization modes:

    - per_scale=False (default): the bare TC reconstruction under-shoots
      variance by ~0.696 with EWDM's CWT normalization, so a single universal
      constant 1.4383 (~sqrt(2)) is applied, giving 1-3% round-trip fidelity
      across grids.

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
    W = np.asarray(W)
    if freqs.size < 2:
        # np.gradient needs >=2 scales for the inverse spacing; a single band
        # is unreconstructable, so return zeros rather than crash.
        warnings.warn(
            "inverse CWT needs >= 2 frequency bands; returning zeros.",
            UserWarning, stacklevel=2)
        return np.zeros(W.shape[1])

    if not per_scale:
        if mother is None:
            mother = Morlet(6.0)
        scales = 1.0 / (mother.flambda * freqs)
        ds = np.abs(np.gradient(scales))
        dt = 1.0 / fs
        CALIBRATION = 1.4383
        weight = (CALIBRATION * np.sqrt(dt) /
                  (_mother_cdelta(mother) * (np.pi ** -0.25)))
        return weight * (np.real(W) / scales[:, None]**1.5 * ds[:, None]).sum(axis=0)

    g = _per_scale_gain(freqs, fs, W.shape[1], mother)
    # Uncalibratable bands carry NaN gain; drop them (zero contribution) so a
    # single bad band does not poison the whole reconstruction.
    g = np.where(np.isfinite(g), g, 0.0)
    return _bare_inverse(W * g[:, None], freqs, fs, mother)


# ---------------------------------------------------------------------------
# Aperture transfer function: spatial-mean slope over a finite footprint
# ---------------------------------------------------------------------------
_J1_FIRST_NULL = 3.8317059702075123     # first zero of J1 (circular-disc null)


def aperture_transfer_function(k, diameter_m, shape="circular"):
    """Amplitude transfer of the spatial-mean slope over a measurement aperture.

    Averaging a wave of wavenumber magnitude k over a finite footprint low-passes
    it by a wavenumber-dependent factor H(k) <= 1. For a circular disc of diameter
    D (radius R = D/2) this is the isotropic jinc

        H(k) = 2 J1(kR) / (kR),   H(0) = 1,

    independent of wave direction; it first vanishes at kR = 3.832, beyond which
    the aperture has nulled the wave. For a square frame of side L the factor is
    sinc(kx L/2) sinc(ky L/2) (direction-dependent); the RMS azimuthal average
    sqrt(<H^2>) is returned. Via the dispersion relation k(f) this becomes a
    frequency-dependent transfer the long-wave inversion can undo. NaN k yields
    NaN H.

    Args:
        k          : scalar or array of wavenumber magnitudes (rad/m)
        diameter_m : disc diameter (circular) or frame side L (square), metres
        shape      : "circular" (default) or "square"

    Returns:
        H : same shape as k, the amplitude transfer in [<=1].
    """
    k = np.asarray(k, dtype=float)
    if diameter_m is None or diameter_m <= 0:
        raise ValueError("diameter_m must be positive")
    if shape == "circular":
        x = k * (diameter_m / 2.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            # NaN k (e.g. blocked frequencies) passes through as NaN, not 1.0.
            return np.where(np.isfinite(x),
                            np.where(x > 1e-9, 2.0 * j1(x) / x, 1.0),
                            np.nan)
    if shape == "square":
        L = float(diameter_m)
        th = np.linspace(0.0, np.pi / 2, 64)            # quarter-symmetry
        kx = np.outer(k.ravel(), np.cos(th))
        ky = np.outer(k.ravel(), np.sin(th))
        H2 = np.mean((np.sinc(kx * L / (2 * np.pi))
                      * np.sinc(ky * L / (2 * np.pi))) ** 2, axis=1)
        return np.sqrt(H2).reshape(k.shape)
    raise ValueError(f"unknown aperture shape {shape!r}")


def aperture_transfer_gain(freqs, k_disp, diameter_m, shape="circular",
                           min_transfer=0.3):
    """Per-frequency correction 1/H(k(f)) that undoes aperture low-passing.

    The spatial-mean slope -- and the elevation reconstructed from it -- is
    suppressed by aperture_transfer_function H(k(f)); multiply the elevation CWT
    coefficients by this gain to restore it. Bands where |H| < min_transfer (at
    or beyond the aperture null, where 1/H would amplify noise without bound) are
    returned NaN with a warning -- nulled energy cannot be recovered.

    Args:
        freqs        : (nf,) CWT frequency grid (Hz); used only for the message.
        k_disp       : (nf,) dispersion wavenumber on `freqs` (rad/m).
        diameter_m   : aperture diameter (circular) or frame side (square), m.
        shape        : "circular" (default) or "square".
        min_transfer : smallest |H| inverted; below it the band returns NaN.

    Returns:
        gain : (nf,) real correction (1/H), NaN where |H| < min_transfer.
    """
    H = aperture_transfer_function(np.asarray(k_disp, dtype=float),
                                   diameter_m, shape=shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(np.abs(H) >= min_transfer, 1.0 / H, np.nan)
    n_bad = int(np.sum(~np.isfinite(gain)))
    if n_bad:
        warnings.warn(
            f"{n_bad}/{gain.size} band(s) at/beyond the aperture transfer null "
            f"(|H| < {min_transfer}); returning NaN (nulled energy is "
            f"unrecoverable).", UserWarning, stacklevel=2)
    return gain


