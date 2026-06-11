"""
Tests for the eta_field_recon sibling package.

We test the key invariants (orthogonal decomposition, shapes, return
types) on small synthetic inputs that run in seconds. The full multi-mode
demo in eta_field_recon/demo_eta_field.py exercises the pipeline at scale
and is meant for visual inspection, not unit testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from eta_field_recon import reconstruct_eta_field, lindisp_with_current


# ---------------------------------------------------------------------------
# Signed amplitude calibration: reconstruction must be upright (not inverted)
# for any mother wavelet. ewdm tabulates cdelta only for Morlet(6); the -1
# sentinel for other omega0 used to silently sign-flip eta(t) while Hs and the
# spectrum looked perfect.
# ---------------------------------------------------------------------------

def _mono_roundtrip(f0, theta_deg, w0, fs=4.0, T=2048, depth=100.0,
                    per_scale=True):
    """Round-trip a single tone: true cos(wt) elevation -> exact slopes at
    theta_deg -> CWT -> krogstad -> inverse. Returns (eta_true, rec) over the
    cone-of-influence-light central window."""
    from ewdm.wavelets import Morlet
    from eta_field_recon.wavelet_core import (
        _cwt, _inverse_cwt, krogstad_eta_coeffs, skirt_correction)

    mother = Morlet(float(w0))
    t = np.arange(T) / fs
    omega = 2 * np.pi * f0
    _, k = lindisp_with_current(np.array([omega]), depth, 0.0)
    k = k[0]
    th = np.deg2rad(theta_deg)
    eta_true = np.cos(omega * t)
    sx = k * np.cos(th) * np.sin(omega * t)      # d eta/dx at the origin
    sy = k * np.sin(th) * np.sin(omega * t)      # d eta/dy at the origin
    freqs = np.linspace(0.05, 2.0, 80)
    Wsx = _cwt(sx, freqs, fs, mother).values
    Wsy = _cwt(sy, freqs, fs, mother).values
    _, k_disp = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    skirt = skirt_correction(freqs, fs, k_disp, T, mother, per_scale=per_scale)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, k_disp, skirt_gain=skirt)
    rec = _inverse_cwt(W_eta, freqs, fs, mother, per_scale=per_scale)
    c = slice(int(0.2 * T), int(0.8 * T))
    return eta_true[c], rec[c]


@pytest.mark.parametrize("w0", [4, 6, 8, 10, 12])
@pytest.mark.parametrize("per_scale", [False, True])
def test_monochromatic_upright(w0, per_scale):
    """Reconstruction stays upright (+correlation) for every mother, not just
    the cdelta-tabulated Morlet(6)."""
    eta_true, rec = _mono_roundtrip(0.2, 45.0, w0, per_scale=per_scale)
    assert np.corrcoef(eta_true, rec)[0, 1] > 0.99


def test_cdelta_sentinel_warns_not_silent():
    """A non-Morlet(6) mother (ewdm cdelta == -1 sentinel) must warn, never
    pass through silently and invert the surface."""
    import warnings
    from ewdm.wavelets import Morlet
    from eta_field_recon import wavelet_core as wc

    wc._CDELTA_WARNED.discard(wc._mother_key(Morlet(8.0)))
    with pytest.warns(UserWarning, match="cdelta"):
        wc._mother_cdelta(Morlet(8.0))
    # Morlet(6) is tabulated -> no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert wc._mother_cdelta(Morlet(6.0)) > 0


# ---------------------------------------------------------------------------
# Cone-of-influence guard: uncalibratable bands return NaN + warn, never a
# silently clamped value that reads as a recovered band.
# ---------------------------------------------------------------------------

def test_coi_guard_warns_single_scale():
    """A single-frequency request cannot form the inverse scale spacing ->
    NaN, with a warning, rather than a crash or a clamped value."""
    from ewdm.wavelets import Morlet
    from eta_field_recon.wavelet_core import _per_scale_gain
    with pytest.warns(UserWarning):
        g = _per_scale_gain(np.array([0.06]), fs=4.0, T=4096, mother=Morlet(16.0))
    assert np.isnan(g[0])


def test_coi_guard_flags_oversize_wavelet():
    """On a short record the low-frequency bands' wavelets exceed the record;
    those bands must come back NaN, while the well-resolved bands stay finite."""
    from ewdm.wavelets import Morlet
    from eta_field_recon.wavelet_core import _per_scale_gain
    freqs = np.linspace(0.05, 2.0, 80)
    with pytest.warns(UserWarning, match="cone-of-influence"):
        g = _per_scale_gain(freqs, fs=4.0, T=256, mother=Morlet(16.0))
    assert np.isnan(g[0])            # 0.05 Hz wavelet too wide for a 64 s record
    assert np.isfinite(g).any()      # high-frequency bands still calibrate


def test_coi_nan_bands_do_not_poison_reconstruction():
    """NaN-gain bands are dropped by the inverse: a record with both flagged
    and calibrated bands reconstructs finite, non-trivial eta."""
    from ewdm.wavelets import Morlet
    from eta_field_recon import wavelet_core as wc
    freqs = np.linspace(0.05, 2.0, 80)
    fs, T = 4.0, 512
    wc._PER_SCALE_GAIN_CACHE.clear()
    g = wc._per_scale_gain(freqs, fs, T, Morlet(16.0))
    assert np.isnan(g).any() and np.isfinite(g).any()   # mix of both
    good = int(np.flatnonzero(np.isfinite(g))[0])
    W = np.zeros((freqs.size, T), dtype=complex)
    W[good] = 1.0 + 0.0j
    rec = wc._inverse_cwt(W, freqs, fs=fs, mother=Morlet(16.0), per_scale=True)
    assert np.isfinite(rec).all()
    assert np.any(rec != 0.0)


# ---------------------------------------------------------------------------
# Direction-sign robustness: recovered Hs must not dip on the along-look axis
# (90/270 deg), where the cross-look slope channel vanishes and the pointwise
# relative-phase sign would otherwise be set by noise.
# ---------------------------------------------------------------------------

def _unidir_sea(theta_deg, s=64, Hs=1.0, fp=0.12, fs=2.0, T=2048, depth=100.0,
                seed=1):
    """Gaussian sea: Bretschneider S(f) x narrow cos-2s spreading about
    theta_deg. Returns spatial-point (sx, sy, eta) series with exact slope-
    elevation cross-covariance (sx,sy are the quadrature of eta scaled by
    k*cos/sin(theta))."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    df = fs / T
    f = np.arange(1, T // 2) * df
    S = (5 / 16) * Hs ** 2 * fp ** 4 / f ** 5 * np.exp(-1.25 * (fp / f) ** 4)
    th = np.deg2rad(theta_deg) + rng.normal(0, 1.0 / np.sqrt(2 * s + 1), f.size)
    a = np.sqrt(2 * S * df)
    ph = rng.uniform(0, 2 * np.pi, f.size)
    om = 2 * np.pi * f
    _, k = lindisp_with_current(om, depth, 0.0)
    arg = np.outer(t, om) - ph[None, :]
    eta = (a[None, :] * np.cos(arg)).sum(1)
    sx = ((a * k * np.cos(th))[None, :] * np.sin(arg)).sum(1)
    sy = ((a * k * np.sin(th))[None, :] * np.sin(arg)).sum(1)
    return sx, sy, eta, dict(fs=fs, T=T, depth=depth)


def _hs_ratio(sea, w0=6.0):
    """Recovered Hs / input Hs through the long-wave krogstad inversion."""
    from ewdm.wavelets import Morlet
    from scipy.signal.windows import tukey
    from eta_field_recon.wavelet_core import _cwt, _inverse_cwt, krogstad_eta_coeffs
    sx, sy, eta, p = sea
    fs, T, depth = p["fs"], p["T"], p["depth"]
    mother = Morlet(w0)
    freqs = np.linspace(0.05, 2.0, 80)
    win = tukey(T, 0.25)
    Wsx = _cwt((sx - sx.mean()) * win, freqs, fs, mother).values
    Wsy = _cwt((sy - sy.mean()) * win, freqs, fs, mother).values
    _, kd = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, kd)
    rec = _inverse_cwt(W_eta, freqs, fs, mother, per_scale=False)
    c = slice(int(0.2 * T), int(0.8 * T))
    return (4 * np.std(rec[c])) / (4 * np.std((eta * win)[c]))


def test_direction_invariance_unidirectional():
    ratios = [_hs_ratio(_unidir_sea(theta_deg=d, s=64)) for d in range(0, 360, 5)]
    assert min(ratios) > 0.92                 # no cardinal-axis dip
    assert max(ratios) - min(ratios) < 0.06


def test_along_look_axis_no_hs_dip():
    """Direct 90 deg vs 45 deg check: the along-look axis must recover the same
    Hs as a diagonal sea (the bug collapsed 90 deg to ~0.77)."""
    r45 = _hs_ratio(_unidir_sea(theta_deg=45, s=64))
    r90 = _hs_ratio(_unidir_sea(theta_deg=90, s=64))
    assert r90 > 0.92
    assert abs(r90 - r45) < 0.05


# ---------------------------------------------------------------------------
# Directional-spreading Hs bias: the krogstad projection discards off-axis
# variance, so recovered Hs runs low for broad seas. The r2-based correction
# (directional_spread + spread_hs_factor) must restore it to within ~3%.
# ---------------------------------------------------------------------------

def _hs_ratio_calibrated(sea, w0=6.0):
    """Recovered/input Hs and measured r2 on the calibrated (per_scale+skirt)
    path, where the spreading bias is mother-agnostic."""
    from ewdm.wavelets import Morlet
    from scipy.signal.windows import tukey
    from eta_field_recon.wavelet_core import (
        _cwt, _inverse_cwt, krogstad_eta_coeffs, skirt_correction,
        directional_spread)
    sx, sy, eta, p = sea
    fs, T, depth = p["fs"], p["T"], p["depth"]
    mother = Morlet(w0)
    freqs = np.linspace(0.05, 2.0, 80)
    win = tukey(T, 0.25)
    Wsx = _cwt((sx - sx.mean()) * win, freqs, fs, mother).values
    Wsy = _cwt((sy - sy.mean()) * win, freqs, fs, mother).values
    _, kd = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    sg = skirt_correction(freqs, fs, kd, T, mother, per_scale=True)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, kd, skirt_gain=sg)
    rec = _inverse_cwt(W_eta, freqs, fs, mother, per_scale=True)
    c = slice(int(0.2 * T), int(0.8 * T))
    ratio = np.std(rec[c]) / np.std((eta * win)[c])
    r2 = directional_spread(Wsx[:, c], Wsy[:, c])["r2"]
    return ratio, r2


def test_directional_spread_monotonic_in_r2():
    """r2 must fall monotonically as the cos-2s spread broadens."""
    r2s = [_hs_ratio_calibrated(_unidir_sea(theta_deg=45, s=s))[1]
           for s in (1, 4, 16, 64)]
    assert all(a < b for a, b in zip(r2s, r2s[1:]))   # increasing with s


@pytest.mark.parametrize("s", [2, 8, 32])
def test_spread_correction_recovers_hs(s):
    """Corrected Hs within ~3% of input across cos-2s s in {2,8,32} at 45 deg."""
    from eta_field_recon.wavelet_core import spread_hs_factor
    corrected = []
    for seed in range(4):
        sea = _unidir_sea(theta_deg=45, s=s, seed=seed)
        ratio, r2 = _hs_ratio_calibrated(sea)
        corrected.append(ratio * spread_hs_factor(r2))
    assert abs(np.mean(corrected) - 1.0) < 0.03


def test_spread_factor_unity_for_unidirectional():
    from eta_field_recon.wavelet_core import spread_hs_factor
    assert abs(spread_hs_factor(1.0) - 1.0) < 0.02   # no boost when unidirectional
    assert spread_hs_factor(0.5) > spread_hs_factor(0.9)   # more boost when broad


# ---------------------------------------------------------------------------
# Aperture transfer function: averaging slope over a finite footprint low-passes
# the wave by H(k); the inversion can divide it back out below the null.
# ---------------------------------------------------------------------------

def test_aperture_transfer_unity_at_zero_wavenumber():
    from eta_field_recon import aperture_transfer_function
    H = aperture_transfer_function(np.array([0.0, 1e-12]), diameter_m=1.5)
    np.testing.assert_allclose(H, 1.0, atol=1e-6)


def test_aperture_transfer_monotone_decrease_to_null():
    """Circular jinc decreases from 1 and hits its first null at kR=3.832."""
    from eta_field_recon import aperture_transfer_function
    from eta_field_recon.wavelet_core import _J1_FIRST_NULL
    R = 0.75
    k = np.linspace(1e-3, _J1_FIRST_NULL / R, 200)
    H = aperture_transfer_function(k, diameter_m=2 * R)
    assert H[0] > 0.999
    assert np.all(np.diff(H) < 1e-9)            # monotone non-increasing to null
    assert abs(H[-1]) < 1e-3                     # ~0 at the first null


def test_aperture_transfer_gain_inverts_below_null_nan_above():
    """Gain is 1/H where H is trusted, NaN (with a warning) at/beyond the null."""
    from eta_field_recon import aperture_transfer_function, aperture_transfer_gain
    freqs = np.linspace(0.05, 1.2, 60)
    _, k = lindisp_with_current(2 * np.pi * freqs, h=15.0, current_m_s=0.0)
    R = 2.915 / 2
    with pytest.warns(UserWarning, match="aperture transfer null"):
        g = aperture_transfer_gain(freqs, k, diameter_m=2 * R, min_transfer=0.3)
    H = aperture_transfer_function(k, diameter_m=2 * R)
    good = np.abs(H) >= 0.3
    np.testing.assert_allclose(g[good], 1.0 / H[good], rtol=1e-10)
    assert np.all(np.isnan(g[~good]))
    assert np.all(g[good] >= 1.0)                # a correction never attenuates


def test_aperture_correction_boosts_high_frequency_eta():
    """End-to-end: a disc-aperture reconstruction with the transfer correction
    recovers more variance than without (the high-f bands are un-suppressed)."""
    from eta_field_recon import reconstruct_eta_field
    fs, T = 4.0, 512
    f0 = 0.4                                     # mid-band, meaningfully suppressed
    g = 9.806
    k0 = (2 * np.pi * f0) ** 2 / g
    t = np.arange(T) / fs
    Ny = Nx = 24
    dx = 0.05                                    # ~1.2 m frame
    # uniform-in-time swell tilt along +y, spatially resolved so the disc mean
    # is suppressed by H(k0 R)
    x = (np.arange(Nx) - Nx / 2) * dx
    phase = k0 * x[None, None, :] - 2 * np.pi * f0 * t[:, None, None]
    sy = (k0 * 0.1 * np.cos(phase)) * np.ones((T, Ny, Nx))
    sx = np.zeros_like(sy)
    common = dict(dx=dx, fs=fs, water_depth_m=100.0, downsample=2,
                  long_wave=True, short_wave=False,
                  aperture_diameter_m=0.8, verbose=False)
    _, el_raw, _, _, _ = reconstruct_eta_field(sx, sy, **common)
    _, el_cor, _, _, diag = reconstruct_eta_field(
        sx, sy, aperture_transfer_correct=True, **common)
    assert diag["aperture_gain"] is not None
    assert el_cor.std() > el_raw.std()           # correction restores amplitude


# ---------------------------------------------------------------------------
# Multi-aperture long-wave blend: large disc (low noise) below the handoff,
# small disc (less aperture suppression) above, joined at the maximum-overlap
# frequency.
# ---------------------------------------------------------------------------

def test_crossover_finds_minimum_ratio():
    from eta_field_recon import aperture_crossover_frequency
    freqs = np.linspace(0.05, 2.0, 80)
    # ratio elevated at both ends (small-disc noise low-f, suppression high-f),
    # minimum at 0.3 Hz
    ratio = 1.0 + (np.log2(freqs / 0.3)) ** 2
    P_small = ratio
    P_large = np.ones_like(freqs)
    fx = aperture_crossover_frequency(P_small, P_large, freqs, band=(0.1, 1.0))
    assert abs(fx - 0.3) < 0.05


def test_blend_takes_large_below_small_above():
    """Blended coeffs follow the large disc below the handoff and the small disc
    above it."""
    from eta_field_recon import blend_aperture_coeffs
    freqs = np.linspace(0.05, 2.0, 80)
    T = 32
    center = 0.35
    # small disc inflated below center (1/k^2-amplified noise), large disc
    # suppressed above center (aperture transfer); they agree at center, so the
    # power ratio (small/large) has its minimum there -> handoff at center.
    A_small = 1.0 + 2.0 * np.clip(np.log2(center / freqs), 0, None)
    A_large = 1.0 / (1.0 + np.clip(np.log2(freqs / center), 0, None))
    W_small = (A_small[:, None] * np.ones((freqs.size, T))).astype(complex)
    W_large = (A_large[:, None] * np.ones((freqs.size, T))).astype(complex)
    W_bl, xovers = blend_aperture_coeffs([W_large, W_small], freqs, band=(0.1, 1.0))
    assert len(xovers) == 1
    assert abs(xovers[0] - center) < 0.1
    lowf = freqs < xovers[0] / 2.5      # well below the logistic transition
    highf = freqs > xovers[0] * 2.5     # well above it
    # below handoff blended ~ large (clean); above ~ small (less suppressed)
    assert np.allclose(np.abs(W_bl[lowf]), np.abs(W_large[lowf]), atol=0.1)
    assert np.allclose(np.abs(W_bl[highf]), np.abs(W_small[highf]), atol=0.1)


def test_reconstruct_multi_aperture_runs_and_reports_crossovers():
    """reconstruct_eta_field with aperture_diameters_m blends discs and reports
    the handoff frequencies in diag."""
    from eta_field_recon import reconstruct_eta_field
    fs, T, Ny, Nx, dx = 10.0, 256, 24, 24, 0.05
    g = 9.806
    t = np.arange(T) / fs
    rng = np.random.RandomState(0)
    # a swell (uniform tilt) + a shorter wave resolved across the frame
    x = (np.arange(Nx) - Nx / 2) * dx
    k1 = (2 * np.pi * 0.5) ** 2 / g
    sy = (0.15 * np.sin(2 * np.pi * 0.12 * t))[:, None, None] * np.ones((T, Ny, Nx))
    sy = sy + 0.05 * np.cos(k1 * x[None, None, :] - 2 * np.pi * 0.5 * t[:, None, None])
    sx = np.zeros((T, Ny, Nx))
    _, el, _, _, diag = reconstruct_eta_field(
        sx, sy, dx=dx, fs=fs, water_depth_m=100.0, downsample=2,
        long_wave=True, short_wave=False,
        aperture_diameters_m=[1.0, 0.4], verbose=False)
    assert diag["aperture_crossovers"] is not None
    assert len(diag["aperture_crossovers"]) == 1
    assert np.isfinite(el).all() and el.std() > 0


# ---------------------------------------------------------------------------
# Spectral recolor: keep the Krogstad phase, impose the directionally-complete
# direct amplitude. Closes the projection-loss Hs shortfall for broad seas.
# ---------------------------------------------------------------------------

def _bandstd(x, fs, lo=0.06, hi=1.5):
    """Std of x band-limited to [lo, hi] Hz (FFT brick-wall)."""
    n = len(x)
    X = np.fft.rfft(x - np.mean(x))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    X[(fr < lo) | (fr > hi)] = 0.0
    return np.std(np.fft.irfft(X, n))


def _krog_long(sea, freqs=None):
    """Plain (uncalibrated) Krogstad long-wave series for a sea."""
    from eta_field_recon.wavelet_core import _cwt, _inverse_cwt, krogstad_eta_coeffs
    from scipy.signal.windows import tukey
    sx, sy, eta, p = sea
    fs, T, depth = p["fs"], p["T"], p["depth"]
    if freqs is None:
        freqs = np.linspace(0.05, 2.0, 80)
    win = tukey(T, 0.25)
    Wsx = _cwt((sx - sx.mean()) * win, freqs, fs).values
    Wsy = _cwt((sy - sy.mean()) * win, freqs, fs).values
    _, kd = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, kd)
    return _inverse_cwt(W_eta, freqs, fs, per_scale=False)


@pytest.mark.parametrize("s", [32, 8, 2])
def test_recolor_recovers_directional_variance(s):
    """Krogstad under-reads broader seas; recoloring to the direct amplitude
    restores ~unit Hs regardless of spread."""
    from eta_field_recon import recolor_to_direct_spectrum
    krog, rec = [], []
    for seed in range(4):
        sea = _unidir_sea(theta_deg=45, s=s, seed=seed)
        sx, sy, eta, p = sea
        el_krog = _krog_long(sea)
        el_rec = recolor_to_direct_spectrum(
            el_krog, [(sx - sx.mean(), sy - sy.mean(), None)],
            p["fs"], p["depth"], highpass_fmin=0.04)
        krog.append(_bandstd(el_krog, p["fs"]) / _bandstd(eta, p["fs"]))
        rec.append(_bandstd(el_rec, p["fs"]) / _bandstd(eta, p["fs"]))
    krog, rec = np.mean(krog), np.mean(rec)
    assert rec > krog                         # recovers the projection loss
    assert abs(rec - 1.0) < 0.05              # lands on unit Hs


def test_recolor_amplitude_independent_of_carrier_scale():
    """Output depends only on the carrier PHASE: scaling eta_krog leaves it
    unchanged (the magnitude comes entirely from the direct slope spectrum)."""
    from eta_field_recon import recolor_to_direct_spectrum
    sea = _unidir_sea(theta_deg=30, s=8, seed=3)
    sx, sy, _, p = sea
    el = _krog_long(sea)
    e1 = recolor_to_direct_spectrum(el, [(sx, sy, None)], p["fs"], p["depth"])
    e2 = recolor_to_direct_spectrum(7.0 * el, [(sx, sy, None)], p["fs"], p["depth"])
    np.testing.assert_allclose(e1, e2, atol=1e-9)


def test_recolor_zero_slopes_give_zero():
    """Amplitude comes from the slopes: zero slope variance -> zero eta,
    whatever the carrier."""
    from eta_field_recon import recolor_to_direct_spectrum
    sea = _unidir_sea(theta_deg=30, s=8, seed=0)
    _, _, _, p = sea
    z = np.zeros(p["T"])
    out = recolor_to_direct_spectrum(_krog_long(sea), [(z, z, None)],
                                     p["fs"], p["depth"])
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_recolor_blend_path_jinc_corrects_two_discs():
    """Multi-disc recolor (jinc + aperture blend) runs and returns finite,
    non-trivial eta."""
    from eta_field_recon import recolor_to_direct_spectrum
    sea = _unidir_sea(theta_deg=45, s=8, seed=1)
    sx, sy, _, p = sea
    sx, sy = sx - sx.mean(), sy - sy.mean()
    out = recolor_to_direct_spectrum(
        _krog_long(sea), [(sx, sy, 2.915), (sx, sy, 0.729)],
        p["fs"], p["depth"], highpass_fmin=0.05, blend_band=(0.06, 0.6))
    assert np.isfinite(out).all() and out.std() > 0


def test_reconstruct_recolor_direct_changes_eta_long():
    """recolor_direct=True recolors the multi-aperture long wave: eta_long stays
    finite, differs from the plain Krogstad path, and diag carries the carrier."""
    from eta_field_recon import reconstruct_eta_field
    fs, T, Ny, Nx, dx = 10.0, 256, 24, 24, 0.05
    g = 9.806
    t = np.arange(T) / fs
    x = (np.arange(Nx) - Nx / 2) * dx
    k1 = (2 * np.pi * 0.5) ** 2 / g
    sy = (0.15 * np.sin(2 * np.pi * 0.12 * t))[:, None, None] * np.ones((T, Ny, Nx))
    sy = sy + 0.05 * np.cos(k1 * x[None, None, :] - 2 * np.pi * 0.5 * t[:, None, None])
    sx = np.zeros((T, Ny, Nx))
    common = dict(dx=dx, fs=fs, water_depth_m=100.0, downsample=2,
                  long_wave=True, short_wave=False,
                  aperture_diameters_m=[1.0, 0.4], verbose=False)
    _, el_plain, _, _, d0 = reconstruct_eta_field(sx, sy, **common)
    _, el_rec, _, _, d1 = reconstruct_eta_field(sx, sy, recolor_direct=True, **common)
    assert d0["recolor_direct"] is False and d0["eta_long_krog"] is None
    assert d1["recolor_direct"] is True and d1["eta_long_krog"] is not None
    assert np.isfinite(el_rec).all() and el_rec.std() > 0
    assert not np.allclose(el_rec, el_plain, atol=1e-6)


# ---------------------------------------------------------------------------
# Dispersion-relation sanity
# ---------------------------------------------------------------------------

def test_dispersion_returns_finite_for_typical_frequencies():
    omega = 2 * np.pi * np.array([0.1, 0.5, 1.0, 2.0])  # Hz -> rad/s
    c, k = lindisp_with_current(omega, h=100.0, current_m_s=0.0)
    assert np.isfinite(c).all()
    assert np.isfinite(k).all()
    assert (k > 0).all()


def test_dispersion_deep_water_limit():
    """For omega^2 >> g/h (deep-water limit), k ~ omega^2 / g.

    At f = 1 Hz, omega = 6.28 rad/s, omega^2/g = 4.02 rad/m. For h=100 m,
    we're well in the deep-water regime."""
    omega = 2 * np.pi * np.array([1.0])
    _, k = lindisp_with_current(omega, h=100.0, current_m_s=0.0)
    k_deep = omega**2 / 9.806
    np.testing.assert_allclose(k, k_deep, rtol=0.02)


def test_dispersion_current_shifts_k():
    """Current along wave direction lowers k for a given omega."""
    omega = 2 * np.pi * np.array([0.5])
    _, k_still   = lindisp_with_current(omega, h=100.0, current_m_s=0.0)
    _, k_current = lindisp_with_current(omega, h=100.0, current_m_s=1.0)
    assert k_current < k_still


def test_dispersion_opposing_current_blocks_high_omega():
    """A strong opposing current makes omega(k) non-monotonic (wave blocking):
    the inversion must warn and return NaN above the blocking frequency rather
    than interpolate a non-monotonic abscissa."""
    # U=-1 m/s blocks waves above ~0.39 Hz; 0.1 Hz resolves, 1.0 Hz does not.
    omega = 2 * np.pi * np.array([0.1, 1.0])
    with pytest.warns(UserWarning, match="non-monotonic"):
        _, k = lindisp_with_current(omega, h=100.0, current_m_s=-1.0)
    assert np.isfinite(k[0])       # low omega still resolved
    assert np.isnan(k[-1])         # blocked high omega -> NaN, not garbage


# ---------------------------------------------------------------------------
# Reconstruction smoke test on a synthetic single-mode field
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def single_mode_slope_stack():
    """A single plane wave at 0.5 Hz, ~6.25 m wavelength, traveling +x.

    For deep water at f=0.5 Hz, k = (2pi)^2 * 0.5^2 / 9.806 = 1.006 rad/m,
    so lambda = 6.25 m. With a 3 m frame, that's lambda/L ~ 2 -- right in
    the band where both paths contribute meaningfully.
    """
    rng = np.random.default_rng(0)
    Nx = Ny = 32          # small for speed
    dx = 0.1               # 0.1 m -> 3.2 m frame
    fs = 4.0               # 4 Hz
    T = 128                # 32 s
    f0 = 0.5
    Hs = 0.2
    g = 9.806
    omega = 2 * np.pi * f0
    k = omega**2 / g
    a = Hs / 2.0           # amplitude

    x = (np.arange(Nx) - Nx/2) * dx
    y = (np.arange(Ny) - Ny/2) * dx
    t = np.arange(T) / fs

    X, Y = np.meshgrid(x, y, indexing="xy")
    # Propagate along +x
    phase = k * X[None, :, :] - omega * t[:, None, None]
    eta_true = a * np.cos(phase).astype(np.float64)
    # Slope = d eta/dx, d eta/dy
    slope_x = -a * k * np.sin(phase)
    slope_y = np.zeros_like(slope_x)
    return slope_x, slope_y, eta_true, dict(dx=dx, fs=fs, T=T, k=k, omega=omega, f0=f0)


def test_reconstruct_returns_expected_shapes(single_mode_slope_stack):
    slope_x, slope_y, eta_true, p = single_mode_slope_stack
    eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    T = p["T"]
    Ny_d, Nx_d = slope_x.shape[1] // 2, slope_x.shape[2] // 2
    assert eta_xyt.shape   == (T, Ny_d, Nx_d)
    assert eta_long.shape  == (T,)
    assert eta_short.shape == (T, Ny_d, Nx_d)
    assert conf.shape      == (T, Ny_d, Nx_d)
    assert isinstance(diag, dict)


def test_eta_short_has_zero_spatial_mean_per_frame(single_mode_slope_stack):
    """Short path is constructed to be zero-mean per frame."""
    slope_x, slope_y, _, p = single_mode_slope_stack
    _, _, eta_short, _, _ = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    means = eta_short.mean(axis=(1, 2))
    np.testing.assert_allclose(means, 0.0, atol=1e-10)


def test_eta_long_matches_spatial_mean_of_eta_xyt(single_mode_slope_stack):
    """By construction, eta_long is the spatial mean of eta_xyt (because
    eta_short is zero-mean per frame)."""
    slope_x, slope_y, _, p = single_mode_slope_stack
    eta_xyt, eta_long, _, _, _ = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    spatial_mean = eta_xyt.mean(axis=(1, 2))
    np.testing.assert_allclose(spatial_mean, eta_long, atol=1e-10)


def test_reconstruct_correlates_with_truth(single_mode_slope_stack):
    """For a clean single-mode field, center-pixel correlation with truth
    should be high after the wavelet edge-of-record taper settles."""
    slope_x, slope_y, eta_true, p = single_mode_slope_stack
    eta_xyt, _, _, conf, _ = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    T = p["T"]
    # Center pixel, in both truth (full-res) and recon (half-res)
    Ny_t, Nx_t = eta_true.shape[1], eta_true.shape[2]
    truth_center = eta_true[:, Ny_t // 2, Nx_t // 2]
    recon_center = eta_xyt[:, eta_xyt.shape[1] // 2, eta_xyt.shape[2] // 2]
    # Confidence weighting addresses the temporal taper.
    w = conf[:, eta_xyt.shape[1] // 2, eta_xyt.shape[2] // 2]
    # Weighted correlation
    tw = truth_center * w
    rw = recon_center * w
    num = np.sum(tw * rw)
    den = np.sqrt(np.sum(tw**2) * np.sum(rw**2))
    corr_weighted = num / den
    assert corr_weighted > 0.85, f"weighted correlation {corr_weighted} too low"


def test_zero_input_gives_zero_output():
    """A genuine zero slope field should produce zero eta."""
    T, Ny, Nx = 64, 16, 16
    zeros = np.zeros((T, Ny, Nx))
    eta_xyt, eta_long, eta_short, _, _ = reconstruct_eta_field(
        zeros, zeros, dx=0.1, fs=4.0, water_depth_m=100.0,
        downsample=2, verbose=False,
    )
    np.testing.assert_allclose(eta_xyt,   0.0, atol=1e-12)
    np.testing.assert_allclose(eta_long,  0.0, atol=1e-12)
    np.testing.assert_allclose(eta_short, 0.0, atol=1e-12)


def test_confidence_in_unit_interval(single_mode_slope_stack):
    """Confidence mask must be on [0, 1]."""
    slope_x, slope_y, _, p = single_mode_slope_stack
    _, _, _, conf, _ = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    assert conf.min() >= 0.0
    assert conf.max() <= 1.0


def test_diag_contains_expected_keys(single_mode_slope_stack):
    slope_x, slope_y, _, p = single_mode_slope_stack
    _, _, _, _, diag = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    expected = {"sx_mean", "sy_mean", "Wsx", "Wsy", "W_eta",
                "cos_th", "sin_th", "k_disp", "x_ds", "y_ds",
                "spatial_W", "temporal_W"}
    missing = expected - set(diag.keys())
    assert not missing, f"diag is missing keys: {missing}"


# ---------------------------------------------------------------------------
# long_wave switch (used by the length gate in eta_pipeline)
# ---------------------------------------------------------------------------

def _synthetic_swell_stack(T=120, Ny=16, Nx=16, fs=10.0, f0=0.2, A=0.3):
    """A single deep-water swell traveling +x, as a (T,Ny,Nx) slope stack."""
    g = 9.806
    k0 = (2 * np.pi * f0) ** 2 / g
    t = np.arange(T) / fs
    phase = 2 * np.pi * f0 * t
    sx = (-k0 * A * np.sin(phase))[:, None, None] * np.ones((T, Ny, Nx))
    sy = np.zeros((T, Ny, Nx))
    return sx, sy, fs


def test_long_wave_false_zeroes_eta_long_and_skips_cwt():
    sx, sy, fs = _synthetic_swell_stack()
    eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
        sx, sy, dx=0.05, fs=fs, water_depth_m=100.0,
        downsample=2, long_wave=False, verbose=False)
    assert np.allclose(eta_long, 0.0)
    assert np.allclose(eta_xyt, eta_short)          # field reduces to short path
    assert diag["long_wave"] is False
    assert diag["Wsx"] is None and diag["W_eta"] is None  # CWT not run


def test_long_wave_true_recovers_swell():
    sx, sy, fs = _synthetic_swell_stack(T=150, A=0.3)
    _, eta_long, _, _, diag = reconstruct_eta_field(
        sx, sy, dx=0.05, fs=fs, water_depth_m=100.0,
        downsample=2, long_wave=True, verbose=False)
    assert diag["long_wave"] is True
    assert diag["Wsx"] is not None
    # Single-mode swell, std ~ A/sqrt(2) ~ 0.21 m. The CWT round-trip and
    # cone-of-influence edges make the exact amplitude record-length-dependent,
    # so assert only order-of-magnitude recovery, not a tight match.
    assert 0.1 < eta_long.std() < 0.35


# ---------------------------------------------------------------------------
# Field-data driver: physics-based length gate
# ---------------------------------------------------------------------------

def test_pipeline_gate_skips_long_wave_on_single_frame_record():
    """The bundled single-frame record is far too short -> long-wave skipped."""
    pytest.importorskip("netCDF4")
    from eta_field_recon import reconstruct_eta_from_record
    import _data

    try:
        frame = _data.frame_path(allow_download=False)
        median = _data.median_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("bundled example NetCDF not available")

    res = reconstruct_eta_from_record(
        frame, ground_dx_m=0.006, gain_reference_path=median,
        downsample=8, verbose=False)

    assert res.long_wave_ran is False
    assert res.n_frames == 1
    assert res.record_duration_s < res.gate_threshold_s
    assert np.allclose(res.eta_long, 0.0)
    assert res.eta_xyt is None      # short_wave defaults off at entry points
    assert res.eta_short is None


def test_pipeline_force_long_wave_overrides_gate():
    """force_long_wave=False is honored even when not otherwise needed."""
    pytest.importorskip("netCDF4")
    from eta_field_recon import reconstruct_eta_from_record
    import _data

    try:
        frame = _data.frame_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("bundled example NetCDF not available")

    res = reconstruct_eta_from_record(
        frame, ground_dx_m=0.006, downsample=8,
        gain_mode="none", force_long_wave=False, verbose=False)
    assert res.long_wave_ran is False
    assert "force_long_wave=False" in res.gate_reason


# ---------------------------------------------------------------------------
# Static orthorectification
# ---------------------------------------------------------------------------

def test_orthorectify_flat_surface_stays_flat():
    from eta_field_recon import orthorectify_static
    Ny = Nx = 80
    sx = np.zeros((Ny, Nx))
    sy = np.zeros((Ny, Nx))
    r = orthorectify_static(
        sx, sy, freeboard_m=23.0, theta_i_mean_deg=30.0,
        focal_length_m=0.075, pixel_pitch_m=6.9e-6)
    assert np.nanmax(np.abs(r.slope_x)) < 1e-9
    assert np.nanmax(np.abs(r.slope_y)) < 1e-9
    assert r.dx_m > 0


def test_orthorectify_uniform_slope_preserved():
    from eta_field_recon import orthorectify_static
    Ny = Nx = 80
    sx = np.full((Ny, Nx), 0.05)
    sy = np.full((Ny, Nx), -0.02)
    r = orthorectify_static(
        sx, sy, freeboard_m=23.0, theta_i_mean_deg=30.0,
        focal_length_m=0.075, pixel_pitch_m=6.9e-6)
    v = r.valid
    assert abs(np.nanmedian(r.slope_x[v]) - 0.05) < 1e-3
    assert abs(np.nanmedian(r.slope_y[v]) - (-0.02)) < 1e-3


def test_orthorectify_stack_shapes_and_dx():
    from eta_field_recon import orthorectify_static
    T, Ny, Nx = 4, 60, 60
    sx = np.zeros((T, Ny, Nx))
    sy = np.zeros((T, Ny, Nx))
    r = orthorectify_static(
        sx, sy, freeboard_m=23.0, theta_i_mean_deg=30.0,
        focal_length_m=0.075, pixel_pitch_m=6.9e-6)
    assert r.slope_x.ndim == 3 and r.slope_x.shape[0] == T
    assert r.slope_x.shape[1:] == r.valid.shape
    # trapezoid ratio >= 1 (far pixels sample more ground than near)
    assert r.diag["trapezoid_ratio"] >= 1.0


def test_pipeline_orthorectify_derives_dx():
    """orthorectify=True derives dx from optics; ground_dx_m may be omitted."""
    pytest.importorskip("netCDF4")
    from eta_field_recon import reconstruct_eta_from_record
    import _data

    try:
        frame = _data.frame_path(allow_download=False)
        median = _data.median_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("bundled example NetCDF not available")

    res = reconstruct_eta_from_record(
        frame, orthorectify=True, gain_reference_path=median,
        downsample=8, verbose=False)
    assert res.orthorectified is True
    assert res.ortho is not None
    assert res.ortho.dx_m > 0
    assert res.ortho.diag["trapezoid_ratio"] >= 1.0


def test_pipeline_requires_dx_when_not_orthorectifying():
    pytest.importorskip("netCDF4")
    from eta_field_recon import reconstruct_eta_from_record
    import _data

    try:
        frame = _data.frame_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("bundled example NetCDF not available")

    with pytest.raises(ValueError, match="ground_dx_m is required"):
        reconstruct_eta_from_record(frame, orthorectify=False, verbose=False)


# ---------------------------------------------------------------------------
# Top-level run_epss entry point
# ---------------------------------------------------------------------------

def _bundled_raw_frame():
    import pytest
    pytest.importorskip("netCDF4")
    from pss import read_netcdf_frame, apply_layout_from_meta
    import _data
    try:
        path = _data.frame_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("example frame not present locally; skipping offline")
    frame, meta = read_netcdf_frame(path)
    apply_layout_from_meta(meta)
    return frame


def test_run_epss_frames_only_returns_slopes():
    """Frames only -> slope stack, no eta stage."""
    frame = _bundled_raw_frame()
    from epss import run_epss
    r = run_epss(frame, verbose=False)
    assert r.eta_ran is False
    assert r.slope_x.ndim == 3 and r.slope_x.shape[0] == 1
    assert r.slope_x.shape == r.slope_y.shape
    assert r.eta_xyt is None
    assert r.slope_results == []          # retention is opt-in (memory)


def test_run_epss_all_params_runs_eta():
    """All acquisition params -> ortho + eta. Default short_wave=False ships
    eta_long but not the resolved field; opting in produces eta_xyt."""
    frame = _bundled_raw_frame()
    from epss import run_epss
    r = run_epss(
        frame, fs=30.0, theta_i_mean_deg=30.0, freeboard_m=23.0,
        pixel_pitch_m=3.45e-6, focal_length_m=0.075,
        downsample=8, verbose=False)
    assert r.eta_ran is True
    # default: no resolved short-wave field
    assert r.eta_xyt is None
    assert r.ortho is not None and r.dx_m > 0
    # single frame -> long-wave gated off
    assert r.long_wave_ran is False
    assert np.allclose(r.eta_long, 0.0)
    # opting into short_wave produces the field
    r2 = run_epss(
        frame, fs=30.0, theta_i_mean_deg=30.0, freeboard_m=23.0,
        pixel_pitch_m=3.45e-6, focal_length_m=0.075,
        downsample=8, short_wave=True, verbose=False)
    assert r2.eta_xyt is not None


def test_run_epss_partial_params_raises():
    """Partial eta-stage GEOMETRY -> clear error.

    Under the decoupled gate, fs and theta_i_mean_deg may be supplied alone
    (they enable the empirical-gain path), so it is specifically a partial
    set of the three geometry params (freeboard/pitch/focal) that must raise.
    Uses a synthetic raw frame so it runs offline.
    """
    from epss import run_epss
    rng = np.random.RandomState(0)
    frame = (1000 + 200 * rng.rand(16, 16)).astype(float)
    # one geometry param without the other two -> all-or-nothing violation
    with pytest.raises(ValueError, match="all-or-nothing"):
        run_epss(frame, fs=30.0, theta_i_mean_deg=30.0, freeboard_m=23.0,
                 verbose=False)


def test_run_epss_fs_and_theta_alone_is_allowed():
    """fs + theta_i without the full geometry is now valid (gain path only).

    This is the contract change from decoupling fs/theta_i from the
    eta-geometry all-or-nothing set: the call must NOT raise, must not run the
    eta stage, and must return slope fields. (A short synthetic record means
    no gain is auto-enabled, which is fine -- we are testing the gate, not the
    gain.)
    """
    from epss import run_epss
    rng = np.random.RandomState(1)
    frame = (1000 + 200 * rng.rand(16, 16)).astype(float)
    r = run_epss(frame, fs=30.0, theta_i_mean_deg=30.0, verbose=False)
    assert r.eta_ran is False
    assert r.eta_xyt is None
    assert r.slope_x.shape[0] == 1


def test_run_epss_empirical_gain_falls_back_without_theta_i():
    """A reference frame but no theta_i must not crash; gain falls back."""
    import pytest
    pytest.importorskip("netCDF4")
    from pss import read_netcdf_frame, apply_layout_from_meta
    import _data
    try:
        frame_p = _data.frame_path(allow_download=False)
        median_p = _data.median_path(allow_download=False)
    except FileNotFoundError:
        pytest.skip("example data not present locally; skipping offline")
    frame, meta = read_netcdf_frame(frame_p)
    apply_layout_from_meta(meta)
    median, _ = read_netcdf_frame(median_p)

    from epss import run_epss
    r = run_epss(frame, gain_reference_frame=median, verbose=False)
    assert r.eta_ran is False
    # reduction still produced slopes (gain just not applied)
    assert r.slope_x.shape[0] == 1


def test_run_epss_single_frame_promoted_to_stack():
    """A 2-D frame is treated as a length-1 record."""
    frame = _bundled_raw_frame()
    from epss import run_epss
    assert frame.ndim == 2
    r = run_epss(frame, verbose=False)
    assert r.slope_x.shape[0] == 1


# ---------------------------------------------------------------------------
# Regression guard: a uniform swell tilt must survive to eta_long.
# This is the test that would have caught the per-frame de-mean bug, where
# pss.compute_slope_field subtracted each frame's spatial-mean slope and so
# destroyed the swell-induced footprint tilt before the long-wave inversion.
# ---------------------------------------------------------------------------

def test_uniform_swell_tilt_survives_to_eta_long():
    """A spatially-uniform slope that oscillates in time at a swell frequency
    represents a long wave tilting the whole footprint. Its amplitude must be
    recovered in eta_long; if the spatial mean were stripped per frame, this
    would collapse to ~zero.
    """
    fs = 10.0
    T = 200                      # 20 s record -> clears the long-wave gate
    Ny = Nx = 8
    f0 = 0.15                    # swell band
    g = 9.806
    k0 = (2 * np.pi * f0) ** 2 / g       # deep-water k
    A = 0.4                              # target elevation amplitude (m)
    t = np.arange(T) / fs
    # along-look slope of a wave traveling +y: spatially uniform per frame,
    # oscillating in time with amplitude A*k0.
    slope_t = A * k0 * np.sin(2 * np.pi * f0 * t)
    sx = np.zeros((T, Ny, Nx))
    sy = slope_t[:, None, None] * np.ones((T, Ny, Nx))

    _, eta_long, _, _, diag = reconstruct_eta_field(
        sx, sy, dx=0.05, fs=fs, water_depth_m=100.0,
        downsample=2, long_wave=True, verbose=False)

    assert diag["long_wave"] is True
    # eta_long must carry real swell energy, not be collapsed to ~zero.
    recovered_amp = eta_long.std() * np.sqrt(2)
    assert recovered_amp > 0.5 * A, (
        f"eta_long amplitude {recovered_amp:.3f} m collapsed vs target {A} m "
        f"-- the spatial-mean (swell) tilt was lost.")


def test_short_wave_false_skips_field_returns_none():
    """short_wave=False skips the g2s loop: eta_short and eta_xyt are None,
    eta_long is still computed, and confidence/diag are intact."""
    import numpy as np
    from eta_field_recon import reconstruct_eta_field
    rng = np.random.RandomState(0)
    T, Ny, Nx, fs, dx = 128, 48, 48, 10.0, 0.05
    t = np.arange(T) / fs
    tilt = 0.02 * np.sin(2 * np.pi * 0.12 * t)
    sx = np.broadcast_to(tilt[:, None, None], (T, Ny, Nx)).copy()
    sx += 0.001 * rng.randn(T, Ny, Nx)
    sy = 0.5 * sx

    # short_wave on (default) vs off, same inputs
    e_on, el_on, es_on, _, _ = reconstruct_eta_field(
        sx, sy, dx=dx, fs=fs, downsample=2, long_wave=True,
        short_wave=True, verbose=False)
    e_off, el_off, es_off, conf, diag = reconstruct_eta_field(
        sx, sy, dx=dx, fs=fs, downsample=2, long_wave=True,
        short_wave=False, verbose=False)

    # off: field outputs are None
    assert e_off is None
    assert es_off is None
    # eta_long is identical whether or not the short-wave path ran
    assert np.array_equal(el_on, el_off)
    # confidence and diag still produced
    assert conf is not None
    assert diag["dx_ds"] == dx * 2


# ---------------------------------------------------------------------------
# Weak-point regressions: per-band direction sign, crossover edge bias,
# transfer-function NaN handling, empty aperture, skirt NaN policy,
# single-band inverse, recolor band ceiling.
# ---------------------------------------------------------------------------

def _krogstad_series(sx, sy, fs, T, freqs=None, depth=100.0):
    """slope series -> eta_long via the full calibrated chain."""
    import warnings as _w
    from eta_field_recon.wavelet_core import (
        _cwt, _inverse_cwt, krogstad_eta_coeffs, skirt_correction)
    from eta_field_recon.recon import _make_temporal_window

    if freqs is None:
        freqs = np.linspace(0.05, 2.0, 80)
    win = _make_temporal_window(T, "tukey", 0.25)
    Wsx = _cwt(sx * win, freqs, fs, None).values
    Wsy = _cwt(sy * win, freqs, fs, None).values
    _, k = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        sg = skirt_correction(freqs, fs, k, T, None, per_scale=True)
        W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, k, skirt_gain=sg)
        return np.real(_inverse_cwt(W_eta, freqs, fs, None, per_scale=True))


def test_opposing_systems_keep_their_own_signs():
    """Two near-axis systems traveling in OPPOSITE along-look directions at
    different frequencies must both reconstruct upright. A global band sign
    inverted the weaker system (per-band r = -1) while preserving Hs."""
    fs, T, g = 10.0, 2048, 9.806
    t = np.arange(T) / fs
    systems = [(0.10, 0.4, np.deg2rad(80.0)), (0.30, 0.2, np.deg2rad(280.0))]
    sx = np.zeros(T)
    sy = np.zeros(T)
    for f0, A, th in systems:
        k0 = (2 * np.pi * f0) ** 2 / g
        sx += A * k0 * np.cos(th) * np.sin(2 * np.pi * f0 * t)
        sy += A * k0 * np.sin(th) * np.sin(2 * np.pi * f0 * t)
    rec = _krogstad_series(sx, sy, fs, T)
    m = slice(T // 4, 3 * T // 4)
    F = np.fft.rfft(rec[m])
    fr = np.fft.rfftfreq(rec[m].size, 1 / fs)

    def band(lo, hi):
        G = F.copy()
        G[(fr < lo) | (fr > hi)] = 0
        return np.fft.irfft(G, n=rec[m].size)

    r1 = np.corrcoef(band(0.05, 0.2), 0.4 * np.cos(2 * np.pi * 0.10 * t[m]))[0, 1]
    r2 = np.corrcoef(band(0.2, 0.45), 0.2 * np.cos(2 * np.pi * 0.30 * t[m]))[0, 1]
    assert r1 > 0.95, f"low-frequency system inverted/degraded: r={r1:.3f}"
    assert r2 > 0.95, f"high-frequency system inverted/degraded: r={r2:.3f}"


def test_crossover_ignores_band_edges():
    """Edge samples must not win the crossover search: zero-padded smoothing
    used to bias the first/last ratio samples low, so a flat-plateau ratio
    with a genuine interior minimum returned the grid edge."""
    from eta_field_recon.wavelet_core import aperture_crossover_frequency

    freqs = np.linspace(0.05, 2.0, 80)
    ratio = np.full(80, 1.7)
    i0 = int(np.argmin(np.abs(freqs - 0.5)))
    ratio[i0] = 1.2
    fx = aperture_crossover_frequency(ratio, np.ones(80), freqs, band=None)
    assert abs(fx - 0.5) < 0.1, f"crossover steered to {fx} Hz, not ~0.5"


def test_crossover_nan_and_empty_band():
    from eta_field_recon.wavelet_core import aperture_crossover_frequency

    freqs = np.linspace(0.05, 2.0, 80)
    ratio = np.full(80, 1.7)
    ratio[40] = 1.2
    ratio[10] = np.nan                       # must not win via NaN argmin
    fx = aperture_crossover_frequency(ratio, np.ones(80), freqs, band=None)
    assert np.isfinite(fx) and abs(fx - freqs[40]) < 0.1
    with pytest.raises(ValueError):
        aperture_crossover_frequency(ratio, np.ones(80), freqs, band=(5.0, 6.0))


def test_aperture_transfer_nan_k_passthrough():
    """NaN wavenumber (blocked frequency) must yield NaN transfer and NaN
    gain, not a silently plausible H = 1."""
    import warnings as _w
    from eta_field_recon.wavelet_core import (aperture_transfer_function,
                                              aperture_transfer_gain)

    H = aperture_transfer_function(np.array([0.1, np.nan, 5.0]), 1.0)
    assert np.isnan(H[1]) and np.isfinite(H[0]) and np.isfinite(H[2])
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        gain = aperture_transfer_gain(np.array([0.1, 0.2, 0.3]),
                                      np.array([0.1, np.nan, 5.0]), 1.0)
    assert np.isnan(gain[1])


def test_empty_aperture_mask_raises():
    """An aperture smaller than the cell spacing selects no cells; that must
    raise, not return an all-NaN spatial mean."""
    from eta_field_recon.recon import _circular_aperture_mask

    with pytest.raises(ValueError):
        _circular_aperture_mask(24, 24, dx=0.1, diameter_m=0.05)


def test_skirt_correction_nan_not_clamped():
    """Uncalibratable skirt bands must come back NaN (with a warning), not
    silently interpolated and clamped; NaN bands are dropped at application."""
    import warnings as _w
    from eta_field_recon.wavelet_core import skirt_correction, krogstad_eta_coeffs

    fs, T = 4.0, 128                          # short record: low bands fail COI
    freqs = np.linspace(0.05, 1.5, 40)
    _, k = lindisp_with_current(2 * np.pi * freqs, 100.0, 0.0)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        sg = skirt_correction(freqs, fs, k, T, None, per_scale=True)
    assert np.isnan(sg).any(), "expected NaN for COI-blocked low bands"
    assert any("uncalibratable" in str(w.message) for w in rec)
    # NaN gain drops the band instead of poisoning the coefficients
    Wsx = np.ones((freqs.size, T), dtype=complex)
    Wsy = np.zeros_like(Wsx)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, k, skirt_gain=sg)
    assert np.isfinite(W_eta).all()
    assert (W_eta[np.isnan(sg)] == 0).all()


def test_inverse_cwt_single_band_returns_zeros():
    import warnings as _w
    from eta_field_recon.wavelet_core import _inverse_cwt

    W = np.ones((1, 128), dtype=complex)
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        out = _inverse_cwt(W, np.array([0.5]), 4.0, None, per_scale=False)
        out2 = _inverse_cwt(W, np.array([0.5]), 4.0, None, per_scale=True)
    assert out.shape == (128,) and np.all(out == 0)
    assert out2.shape == (128,) and np.all(out2 == 0)


def test_recolor_band_ceiling_and_window():
    """Recolor must not deposit amplitude above fmax (where the carrier phase
    is undefined), and the windowed amplitude estimate stays calibrated."""
    from eta_field_recon.wavelet_core import recolor_to_direct_spectrum

    fs, T, g = 10.0, 2048, 9.806
    t = np.arange(T) / fs
    f0, A = 0.2, 0.4
    k0 = (2 * np.pi * f0) ** 2 / g
    eta_krog = A * np.cos(2 * np.pi * f0 * t)
    sx = A * k0 * np.sin(2 * np.pi * f0 * t)
    # add a strong out-of-band tone in the slopes
    f_hi = 3.0
    k_hi = (2 * np.pi * f_hi) ** 2 / g
    sx_hi = sx + 0.4 * k_hi * np.sin(2 * np.pi * f_hi * t)
    eta = recolor_to_direct_spectrum(
        eta_krog, [(sx_hi, np.zeros(T), None)], fs, 100.0, fmax=2.0)
    F = np.abs(np.fft.rfft(eta - eta.mean()))
    fr = np.fft.rfftfreq(T, 1 / fs)
    in_band = F[(fr > 0.1) & (fr < 0.4)].sum()
    out_band = F[fr > 2.0].sum()
    assert out_band < 0.01 * in_band, "energy deposited above the fmax ceiling"
    # amplitude calibration survives the windowing (compare to truth std)
    m = slice(T // 4, 3 * T // 4)
    ratio = eta[m].std() / eta_krog[m].std()
    assert 0.8 < ratio < 1.25
