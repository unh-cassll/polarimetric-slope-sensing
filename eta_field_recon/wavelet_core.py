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


def _fill_nonfinite_linear(a):
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

    - per_scale=False (default): the original behavior.  The bare TC
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
                  (_mother_cdelta(mother) * (np.pi ** -0.25)))
        return weight * (np.real(W) / scales[:, None]**1.5 * ds[:, None]).sum(axis=0)

    W = np.asarray(W)
    g = _per_scale_gain(freqs, fs, W.shape[1], mother)
    # Uncalibratable bands carry NaN gain; drop them (zero contribution) so a
    # single bad band does not poison the whole reconstruction.
    g = np.where(np.isfinite(g), g, 0.0)
    return _bare_inverse(W * g[:, None], freqs, fs, mother)


# ---------------------------------------------------------------------------
# Krogstad signed-projection: slope CWT coefficients -> elevation CWT coeffs
# ---------------------------------------------------------------------------
_AXIS_LOCK_FRAC = 0.5       # |Wsx| < frac*|Wsy| -> along-look axis, lock sign


def krogstad_eta_coeffs(Wsx, Wsy, k_disp, skirt_gain=None):
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
    (destroy it). The sign(0)->+1 guard is correct on-axis; the real on-axis
    failure mode is not a wrong sign but sign VARIANCE across (f, t) when the
    cross-look channel is noise -- handled by the band-locked sign below.

    The elevation coefficient is the slope projection divided by the
    wavenumber. Under ewdm's CWT convention slope = d(eta)/dx maps to a
    factor of -i*k in the wavelet domain, so the inverse carries +1j
    (verified empirically: a unit cos(wt) slope input reconstructs +cos(wt),
    not -cos -- see skirt_correction's self-calibration):

        W_eta = +1j * (cos_th * Wsx + sin_th * Wsy) / k

    Non-finite entries (e.g. from k = NaN at omega = 0) are set to 0.

    Args:
        Wsx, Wsy : (nf, T) complex CWT coefficients of the cross-look and
                   along-look spatial-mean slopes.
        k_disp   : (nf,) dispersion-relation wavenumber (rad/m) on the CWT
                   frequency grid. Broadcast over the time axis.
        skirt_gain : optional (nf,) per-frequency correction for the
                   finite-bandwidth 1/k(omega) skirt reshaping (see
                   skirt_correction).  If given, W_eta is multiplied by
                   skirt_gain[:, None] before return.  Default None (unchanged).

    Returns:
        W_eta  : (nf, T) complex elevation CWT coefficients.
        cos_th : (nf, T) real direction cosine (cross-look projection).
        sin_th : (nf, T) real signed direction sine (along-look projection).
    """
    eps = 1e-30
    mag = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + eps
    cross = np.real(Wsy * np.conj(Wsx))
    rel_sign = np.sign(cross)
    rel_sign = np.where(rel_sign == 0, 1.0, rel_sign)   # indeterminate -> +1

    # Band-locked sign on the along-look axis: where cross-look is weak vs
    # along-look, the pointwise sign is noise and flips across (f, t), so the
    # inverse-CWT sum loses amplitude (the 90/270 deg Hs dip). Lock those points
    # to one dominant sign (preserves Hs, only flips the whole surface); off-axis
    # points keep the pointwise sign.
    weak = np.abs(Wsx) < _AXIS_LOCK_FRAC * np.abs(Wsy)
    dom = np.sign(cross.sum()) or 1.0
    rel_sign = np.where(weak, dom, rel_sign)

    cos_th = np.abs(Wsx) / mag
    sin_th = (np.abs(Wsy) / mag) * rel_sign
    # k may carry NaN (omega = 0) or inf; the division is expected to produce
    # non-finite entries there, which we immediately zero out. Silence the
    # warning since this is by-design, not an error.
    with np.errstate(divide="ignore", invalid="ignore"):
        # +1j (not -1j): ewdm CWT convention; a unit cos slope reconstructs
        # +cos elevation. -1j inverts the surface (verified vs lidar + synthetic).
        W_eta = +1j * (cos_th * Wsx + sin_th * Wsy) / k_disp[:, None]
    W_eta = np.where(np.isfinite(W_eta), W_eta, 0.0)
    if skirt_gain is not None:
        W_eta = W_eta * np.asarray(skirt_gain, dtype=float)[:, None]
    return W_eta, cos_th, sin_th


# ---------------------------------------------------------------------------
# Directional spread + spreading-bias Hs correction
# ---------------------------------------------------------------------------
# f = c0 + c1*x + c2*x^2, x = 1-r2. Calibrated on synthetic cos-2s seas through
# the per_scale+skirt path; mother-agnostic (r2 does not depend on omega0).
_SPREAD_HS_COEF = (0.9921, 0.1455, 0.0435)


def directional_spread(Wsx, Wsy, mask=None):
    """Energy-weighted second directional moment of a slope CWT field.

    From the slope cross-spectra Cxx=sum|Wsx|^2, Cyy=sum|Wsy|^2,
    Cxy=sum Re(Wsx conj Wsy): a2=(Cxx-Cyy)/(Cxx+Cyy), b2=2*Cxy/(Cxx+Cyy),
    r2=hypot(a2,b2). r2=1 for a unidirectional sea and falls with spread; it is
    independent of the mother wavelet. `mask` (nf,T bool) restricts the sum.
    Returns dict(r2, a2, b2, sigma) with sigma=sqrt(max(0,(1-r2)/2)) rad an
    approximate spread width.
    """
    Wsx = np.asarray(Wsx)
    Wsy = np.asarray(Wsy)
    if mask is not None:
        Wsx = Wsx[mask]
        Wsy = Wsy[mask]
    Cxx = float(np.sum(np.abs(Wsx) ** 2))
    Cyy = float(np.sum(np.abs(Wsy) ** 2))
    Cxy = float(np.sum(np.real(Wsx * np.conj(Wsy))))
    tot = Cxx + Cyy
    if tot <= 0:
        return dict(r2=np.nan, a2=np.nan, b2=np.nan, sigma=np.nan)
    a2 = (Cxx - Cyy) / tot
    b2 = 2.0 * Cxy / tot
    r2 = float(np.hypot(a2, b2))
    return dict(r2=r2, a2=a2, b2=b2, sigma=float(np.sqrt(max(0.0, (1 - r2) / 2))))


def spread_hs_factor(r2):
    """Multiplicative Hs correction for the directional-projection variance loss.

    The krogstad projection discards off-axis slope variance, so recovered Hs
    runs low for broad seas (monotonic in spread). Invert that from the measured
    second directional moment r2 (see directional_spread). Brings corrected Hs
    within ~3% of input for cos-2s spreads s>=2 on the per_scale+skirt path.
    Returns ~1 near r2=1 (unidirectional).
    """
    if not np.isfinite(r2):
        return 1.0
    x = 1.0 - float(r2)
    c0, c1, c2 = _SPREAD_HS_COEF
    return float(np.clip(c0 + c1 * x + c2 * x * x, 0.9, 1.5))


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
    sinc(kx L/2) sinc(ky L/2) (direction-dependent); the isotropic azimuthal
    average is returned. Via the dispersion relation k(f) this becomes a
    frequency-dependent transfer the long-wave inversion can undo.

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
            return np.where(x > 1e-9, 2.0 * j1(x) / x, 1.0)
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


# ---------------------------------------------------------------------------
# Multi-aperture long-wave blend: stitch disc-mean reconstructions in frequency
# ---------------------------------------------------------------------------
def aperture_crossover_frequency(power_small, power_large, freqs, band=None, smooth=3):
    """Maximum-overlap handoff frequency between a small- and large-disc long-wave
    reconstruction.

    The per-frequency power ratio P_small/P_large is > 1 at low f (the small disc
    averages fewer pixels, so its residual slope noise -- amplified by 1/k^2 --
    inflates the long wave) and > 1 again at high f (the large disc is suppressed
    by its aperture transfer), with a minimum in between where the two agree.
    That minimum is the natural handoff: trust the large disc below it, the small
    disc above. `band` (lo, hi) Hz restricts the search; `smooth` running-mean-
    smooths the ratio first for robustness.
    """
    freqs = np.asarray(freqs, dtype=float)
    r = np.asarray(power_small, float) / np.maximum(np.asarray(power_large, float), 1e-30)
    if smooth and smooth > 1:
        r = np.convolve(r, np.ones(int(smooth)) / int(smooth), mode="same")
    sel = (np.ones(freqs.shape, bool) if band is None
           else (freqs >= band[0]) & (freqs <= band[1]))
    idx = np.flatnonzero(sel)
    return float(freqs[idx[int(np.argmin(r[idx]))]])


def blend_aperture_coeffs(W_list, freqs, band=None, width_oct=0.35):
    """Frequency-blend long-wave elevation CWT coefficients from an aperture
    ladder, ordered LARGEST disc first -> smallest last.

    Each adjacent pair hands off at its maximum-overlap frequency (see
    aperture_crossover_frequency): the larger disc (low 1/k^2 noise) is kept
    below the handoff, the smaller disc (less aperture suppression, so it
    reaches the intermediate band) above it, joined by a logistic transition
    `width_oct` octaves wide. Returns (W_blended, crossovers).
    """
    W_list = [np.asarray(W) for W in W_list]
    blended = W_list[0]
    log2f = np.log2(np.asarray(freqs, dtype=float))
    crossovers = []
    for W in W_list[1:]:
        P_large = np.mean(np.abs(blended) ** 2, axis=1)
        P_small = np.mean(np.abs(W) ** 2, axis=1)
        fx = aperture_crossover_frequency(P_small, P_large, freqs, band=band)
        w = 1.0 / (1.0 + np.exp((log2f - np.log2(fx)) / width_oct))   # 1 below fx
        blended = w[:, None] * blended + (1.0 - w[:, None]) * W
        crossovers.append(fx)
    return blended, crossovers


# ---------------------------------------------------------------------------
# Spectral recolor: Krogstad phase, directionally-complete direct amplitude
# ---------------------------------------------------------------------------
def recolor_to_direct_spectrum(eta_krog, disc_slopes, fs, depth,
                               current_m_s=0.0, highpass_fmin=0.08,
                               highpass_width_oct=0.25, jinc_correct=True,
                               min_transfer=0.3, blend_band=None,
                               blend_width_oct=0.35):
    """Keep the Krogstad long-wave phase, impose the direct amplitude spectrum.

    The Krogstad projection drops off-axis slope variance (~25% low Hs). The
    direct amplitude A(f) = sqrt(|rfft(sx)|^2 + |rfft(sy)|^2)/k never projects
    (it equals |rfft(eta)| for any wave direction), so it is directionally
    complete. Per disc it is jinc-corrected (divided by the aperture transfer
    H(k), aperture_transfer_gain) and the discs are blended largest-first
    (blend_aperture_coeffs) for an aperture-unbiased estimate. The output keeps
    the rfft phase of eta_krog and this amplitude, high-passed above
    highpass_fmin to suppress the low-f 1/k slope-noise blow-up:

        eta = irfft( HP(f) * A_direct(f) * exp(i*angle(rfft(eta_krog))) ).

    Args:
        eta_krog    : (T,) Krogstad long-wave series (phase carrier).
        disc_slopes : sequence of (sx_mean, sy_mean, diameter_m) per aperture;
                      the disc-mean slopes that drove the inversion. diameter_m
                      is the circular-disc diameter (m) for the jinc correction,
                      or None to skip it. One element = single aperture, no blend.
        fs          : sampling frequency (Hz).
        depth       : water depth (m) for the dispersion relation.
        current_m_s : current projected on the look (m/s). Default 0.
        highpass_fmin, highpass_width_oct : high-pass corner (Hz) and width (oct).
        jinc_correct, min_transfer : enable 1/H; drop bands where |H|<min_transfer.
        blend_band, blend_width_oct : aperture-handoff search window and width.

    Returns:
        eta : (T,) real, zero-mean, direct amplitude + Krogstad phase.
    """
    eta_krog = np.asarray(eta_krog, dtype=float)
    T = eta_krog.size
    f = np.fft.rfftfreq(T, d=1.0 / fs)
    _, k = lindisp_with_current(2.0 * np.pi * f, depth, current_m_s)
    k = np.asarray(k, dtype=float)

    discs = sorted(disc_slopes,                       # largest disc first
                   key=lambda d: (np.inf if d[2] is None else float(d[2])),
                   reverse=True)
    A_list = []
    for sx_mean, sy_mean, diameter_m in discs:
        sx = np.asarray(sx_mean, dtype=float)
        sy = np.asarray(sy_mean, dtype=float)
        Sx = np.fft.rfft(sx - sx.mean())
        Sy = np.fft.rfft(sy - sy.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) / k
        if jinc_correct and diameter_m is not None:
            with warnings.catch_warnings():           # null bands expected here
                warnings.simplefilter("ignore", UserWarning)
                A = A * aperture_transfer_gain(f, k, float(diameter_m),
                                               shape="circular",
                                               min_transfer=min_transfer)
        A_list.append(np.where(np.isfinite(A), A, 0.0).reshape(-1, 1))

    if len(A_list) == 1:
        A_direct = A_list[0].ravel()
    else:
        if blend_band is None:                        # search below large null
            d0 = discs[0][2]
            if d0 is not None:
                H0 = aperture_transfer_function(k, float(d0), shape="circular")
                past = np.flatnonzero((f > highpass_fmin)
                                      & (np.abs(H0) < min_transfer))
                hi = float(f[past[0]]) if past.size else float(f[-1])
            else:
                hi = min(0.6, fs / 4.0)
            blend_band = (highpass_fmin, max(hi, 2.0 * highpass_fmin))
        with np.errstate(divide="ignore"):            # log2(f=0) in the blend
            A_blended, _ = blend_aperture_coeffs(A_list, f, band=blend_band,
                                                 width_oct=blend_width_oct)
        A_direct = np.real(A_blended).ravel()

    with np.errstate(divide="ignore"):
        lr = ((np.log2(np.maximum(f, 1e-12)) - np.log2(highpass_fmin))
              / highpass_width_oct)
    A_direct = A_direct * np.clip(1.0 / (1.0 + np.exp(-lr)), 0.0, 1.0)

    phase = np.angle(np.fft.rfft(eta_krog - eta_krog.mean()))
    eta = np.fft.irfft(A_direct * np.exp(1j * phase), n=T)
    return eta - eta.mean()


# ---------------------------------------------------------------------------
# Krogstad skirt-correction: compensate the finite-bandwidth 1/k(omega) bias
# ---------------------------------------------------------------------------
_SKIRT_CACHE = {}


def skirt_correction(freqs, fs, k_disp, T, mother=None,
                     per_scale=True, temporal_alpha=0.25):
    """Per-frequency correction for the krogstad 1/k(omega) skirt reshaping.

    krogstad_eta_coeffs divides the slope CWT coefficients by k(omega) per
    frequency.  Because each Morlet wavelet has finite bandwidth, a
    monochromatic surface component at f0 has CWT energy spread across
    neighboring scales; dividing that skirt by k(omega) (~f^2 in deep water)
    reshapes it asymmetrically and biases the reconstructed elevation
    amplitude.  This bias grows with frequency and is depth-dependent (via the
    dispersion relation), so it is upstream of, and invisible to, _inverse_cwt.

    This routine returns a per-frequency factor c(f) such that a unit
    monochromatic surface component reconstructs at unit amplitude through the
    full chain.  It is calibrated, at call time, by pushing unit reference
    surface tones through the exact CWT -> (i/k) -> inverse pipeline on the
    caller's own (freqs, fs, k_disp, mother) grid -- self-contained,
    parameter-free, depth-aware through k_disp.  The reshaping acts on the
    coefficient magnitude profile identically for any wave direction, so the
    theta=0 calibration used here generalizes across direction.

    Multiply the krogstad elevation coefficients by c(f)[:, None] before the
    inverse CWT (e.g. via krogstad_eta_coeffs(..., skirt_gain=c)), pairing
    `per_scale` here with the inverse mode used at reconstruction time.

    Args:
        freqs          : (nf,) CWT frequency grid (Hz)
        fs             : sampling frequency (Hz)
        k_disp         : (nf,) dispersion wavenumber on `freqs` (rad/m); same
                         array passed to krogstad_eta_coeffs (carries depth).
        T              : record length (samples) the correction is used on.
        mother         : EWDM mother wavelet; default Morlet(6.0).
        per_scale      : inverse-CWT mode to pair with (must match runtime).
        temporal_alpha : Tukey alpha matching the reconstruction window.

    Returns:
        c : (nf,) real correction; W_eta *= c[:, None] before _inverse_cwt.
    """
    freqs = np.asarray(freqs, dtype=float)
    k_disp = np.asarray(k_disp, dtype=float)
    key = (tuple(np.round(freqs, 10)), float(fs),
           tuple(np.round(k_disp, 8)), int(T), bool(per_scale),
           float(temporal_alpha), _mother_key(mother))
    cached = _SKIRT_CACHE.get(key)
    if cached is not None:
        return cached

    if mother is None:
        mother = Morlet(6.0)
    t = np.arange(T) / fs
    win = tukey(T, alpha=temporal_alpha)
    c = slice(int(0.2 * T), int(0.8 * T))
    k_col = k_disp[:, None]

    corr = np.empty(freqs.size, dtype=float)
    for i, f0 in enumerate(freqs):
        k0 = k_disp[i]
        eta_ref = np.cos(2.0 * np.pi * f0 * t)
        sx_ref = k0 * np.sin(2.0 * np.pi * f0 * t)        # theta=0 exact slope
        sx_w = (sx_ref - sx_ref.mean()) * win
        da = xr.DataArray(sx_w, coords={"time": t}, dims=["time"])
        Wsx = cwt(da, freqs=freqs, fs=fs, mother=mother).values
        with np.errstate(divide="ignore", invalid="ignore"):
            W_eta = +1j * Wsx / k_col                      # krogstad op at theta=0
        W_eta = np.where(np.isfinite(W_eta), W_eta, 0.0)
        rec = np.real(_inverse_cwt(W_eta, freqs, fs, mother, per_scale=per_scale))
        denom = np.std(((eta_ref - eta_ref.mean()) * win)[c])
        r = (np.std(rec[c]) / denom) if denom > 0 else np.nan
        corr[i] = (1.0 / r) if (np.isfinite(r) and r > 1e-6) else np.nan

    corr = _fill_nonfinite_linear(corr)
    corr = np.clip(corr, 0.2, 5.0)
    _SKIRT_CACHE[key] = corr
    return corr
