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
    """For a clean single-mode field, centre-pixel correlation with truth
    should be high after the wavelet edge-of-record taper settles."""
    slope_x, slope_y, eta_true, p = single_mode_slope_stack
    eta_xyt, _, _, conf, _ = reconstruct_eta_field(
        slope_x, slope_y, dx=p["dx"], fs=p["fs"],
        water_depth_m=100.0, downsample=2, verbose=False,
    )
    T = p["T"]
    # Centre pixel, in both truth (full-res) and recon (half-res)
    Ny_t, Nx_t = eta_true.shape[1], eta_true.shape[2]
    truth_centre = eta_true[:, Ny_t // 2, Nx_t // 2]
    recon_centre = eta_xyt[:, eta_xyt.shape[1] // 2, eta_xyt.shape[2] // 2]
    # Confidence weighting addresses the temporal taper.
    w = conf[:, eta_xyt.shape[1] // 2, eta_xyt.shape[2] // 2]
    # Weighted correlation
    tw = truth_centre * w
    rw = recon_centre * w
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
    """A single deep-water swell travelling +x, as a (T,Ny,Nx) slope stack."""
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
    from examples import _data

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
    assert np.allclose(res.eta_xyt, res.eta_short)


def test_pipeline_force_long_wave_overrides_gate():
    """force_long_wave=False is honored even when not otherwise needed."""
    pytest.importorskip("netCDF4")
    from eta_field_recon import reconstruct_eta_from_record
    from examples import _data

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
    from examples import _data

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
    from examples import _data

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
    from examples import _data
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
    assert len(r.slope_results) == 1


def test_run_epss_all_params_runs_eta():
    """All five acquisition params -> ortho + eta."""
    frame = _bundled_raw_frame()
    from epss import run_epss
    r = run_epss(
        frame, fs=30.0, theta_i_mean_deg=30.0, freeboard_m=23.0,
        pixel_pitch_m=3.45e-6, focal_length_m=0.075,
        downsample=8, verbose=False)
    assert r.eta_ran is True
    assert r.eta_xyt is not None
    assert r.ortho is not None and r.dx_m > 0
    # single frame -> long-wave gated off
    assert r.long_wave_ran is False
    assert np.allclose(r.eta_long, 0.0)


def test_run_epss_partial_params_raises():
    """Some-but-not-all acquisition params -> clear error."""
    frame = _bundled_raw_frame()
    from epss import run_epss
    with pytest.raises(ValueError, match="all-or-nothing"):
        run_epss(frame, fs=30.0, theta_i_mean_deg=30.0, verbose=False)


def test_run_epss_empirical_gain_falls_back_without_theta_i():
    """A reference frame but no theta_i must not crash; gain falls back."""
    import pytest
    pytest.importorskip("netCDF4")
    from pss import read_netcdf_frame, apply_layout_from_meta
    from examples import _data
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
    # along-look slope of a wave travelling +y: spatially uniform per frame,
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
