"""
Tests for the step-3 entry-point work:

  A. The empirical-gain decision in `run_epss`, including the >=30 s
     auto-temporal-median trigger and the decoupling of fs / theta_i from the
     eta-stage geometry gate.
  B. The `run_epss_from_slopes` entry point for already-orthorectified slope
     fields.

All synthetic and fully offline (no Zenodo data). Raw "DoFP" frames here are
just positive arrays with even dimensions -- enough to exercise the reduction
and the gate logic without needing a real polarized scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from epss import run_epss, run_epss_from_slopes, DEFAULT_MIN_GAIN_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_stack(T, H=32, W=32, seed=0):
    """A synthetic raw DoFP stack: positive intensities, even dims."""
    rng = np.random.RandomState(seed)
    return (1000.0 + 200.0 * rng.rand(T, H, W))


def _ortho_slope_stack(T=256, Ny=48, Nx=48, fs=10.0, dx=0.05, seed=0):
    """A synthetic orthorectified slope stack with a coherent long wave."""
    rng = np.random.RandomState(seed)
    t = np.arange(T) / fs
    tilt = 0.02 * np.sin(2 * np.pi * 0.12 * t)
    sx = np.broadcast_to(tilt[:, None, None], (T, Ny, Nx)).copy()
    sx += 0.001 * rng.randn(T, Ny, Nx)
    sy = 0.5 * sx
    return sx, sy, dx, fs


# ---------------------------------------------------------------------------
# A. run_epss empirical-gain decision
# ---------------------------------------------------------------------------

def test_default_min_gain_seconds_is_thirty():
    assert DEFAULT_MIN_GAIN_SECONDS == 30.0


def test_short_record_no_reference_gives_no_gain():
    # 5 frames @ 10 Hz = 0.5 s, well under 30 s, no reference frame.
    r = run_epss(_raw_stack(5), theta_i_mean_deg=30.0, fs=10.0, verbose=False)
    assert r.gain_mode == "none"
    assert r.gain_auto_median is False


def test_long_record_auto_median_enables_empirical():
    # 300 frames @ 10 Hz = 30 s, exactly the default threshold, no ref frame.
    r = run_epss(_raw_stack(300), theta_i_mean_deg=30.0, fs=10.0, verbose=False)
    assert r.gain_mode == "empirical"
    assert r.gain_auto_median is True


def test_explicit_reference_enables_empirical_at_any_length():
    stack = _raw_stack(5)
    ref = np.nanmedian(stack, axis=0)
    r = run_epss(stack, theta_i_mean_deg=30.0, fs=10.0,
                 gain_reference_frame=ref, verbose=False)
    assert r.gain_mode == "empirical"
    # explicit reference is not the auto-median path
    assert r.gain_auto_median is False


def test_explicit_reference_enables_empirical_without_fs():
    # A supplied mean/median frame needs no record duration, so fs is not
    # required for the empirical gain in this case.
    stack = _raw_stack(5)
    ref = np.nanmedian(stack, axis=0)
    r = run_epss(stack, theta_i_mean_deg=30.0,
                 gain_reference_frame=ref, verbose=False)   # no fs
    assert r.gain_mode == "empirical"
    assert r.gain_auto_median is False
    assert r.eta_ran is False


def test_no_theta_i_means_no_gain_even_for_long_record():
    r = run_epss(_raw_stack(300), fs=10.0, verbose=False)
    assert r.gain_mode == "none"
    assert r.gain_auto_median is False


def test_long_record_without_fs_cannot_auto_trigger():
    # No fs -> record duration unknown -> auto-median trigger unavailable.
    r = run_epss(_raw_stack(300), theta_i_mean_deg=30.0, verbose=False)
    assert r.gain_mode == "none"
    assert r.gain_auto_median is False


def test_min_gain_seconds_override():
    # 20 frames @ 10 Hz = 2 s. Default 30 s -> no gain; lowered to 1 s -> gain.
    r_default = run_epss(_raw_stack(20), theta_i_mean_deg=30.0, fs=10.0,
                         verbose=False)
    assert r_default.gain_mode == "none"
    r_low = run_epss(_raw_stack(20), theta_i_mean_deg=30.0, fs=10.0,
                     min_gain_seconds=1.0, verbose=False)
    assert r_low.gain_mode == "empirical"
    assert r_low.gain_auto_median is True


def test_explicit_gain_mode_is_respected():
    # If the caller pins gain_mode, the auto logic does not override it.
    r = run_epss(_raw_stack(300), theta_i_mean_deg=30.0, fs=10.0,
                 gain_mode="none", verbose=False)
    assert r.gain_mode == "none"
    assert r.gain_auto_median is False


# ---------------------------------------------------------------------------
# A. eta-stage gate decoupling
# ---------------------------------------------------------------------------

def test_fs_and_theta_alone_does_not_run_eta_or_raise():
    r = run_epss(_raw_stack(5), fs=10.0, theta_i_mean_deg=30.0, verbose=False)
    assert r.eta_ran is False
    assert r.eta_xyt is None
    assert r.slope_x.shape[0] == 5  # the 5-frame stack came through


def test_bare_stack_returns_slopes_only():
    r = run_epss(_raw_stack(4), verbose=False)
    assert r.eta_ran is False
    assert r.gain_mode == "none"
    assert r.slope_x.ndim == 3 and r.slope_x.shape[0] == 4
    assert r.slope_results == []          # retention is opt-in (memory)
    r2 = run_epss(_raw_stack(4), keep_slope_results=True, verbose=False)
    assert len(r2.slope_results) == 4


def test_partial_geometry_raises():
    # freeboard without pitch/focal -> all-or-nothing violation.
    with pytest.raises(ValueError, match="all-or-nothing"):
        run_epss(_raw_stack(5), fs=10.0, theta_i_mean_deg=30.0,
                 freeboard_m=20.0, verbose=False)


def test_complete_geometry_without_fs_raises():
    # All three geometry params + theta but no fs -> eta cannot run.
    with pytest.raises(ValueError, match="fs"):
        run_epss(_raw_stack(5), theta_i_mean_deg=30.0, freeboard_m=20.0,
                 pixel_pitch_m=3.45e-6, focal_length_m=0.075, verbose=False)


# ---------------------------------------------------------------------------
# B. run_epss_from_slopes
# ---------------------------------------------------------------------------

def test_from_slopes_long_record_runs_long_wave():
    sx, sy, dx, fs = _ortho_slope_stack()      # 256 frames @ 10 Hz = 25.6 s
    r = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                             verbose=False)
    assert r.eta_ran is True
    assert r.long_wave_ran is True
    # default short_wave=False: no resolved field, eta_long carries the signal
    assert r.eta_xyt is None
    assert r.eta_short is None
    assert r.eta_long.shape[0] == sx.shape[0]
    # gain is N/A on the slopes-in path
    assert r.gain_mode is None
    assert r.gain_auto_median is False
    # the caller orthorectified; no internal ortho, no per-frame reduction
    assert r.ortho is None
    assert r.slope_results == []
    # eta_long should carry real variance from the coherent tilt
    assert np.std(r.eta_long) > 0


def test_from_slopes_short_wave_opt_in_produces_field():
    # Opting into short_wave produces the resolved field and combined eta_xyt.
    sx, sy, dx, fs = _ortho_slope_stack()
    r = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                             short_wave=True, verbose=False)
    assert r.eta_xyt is not None
    assert r.eta_short is not None
    assert r.eta_xyt.shape[0] == sx.shape[0]


def test_from_slopes_single_frame_gates_off_long_wave():
    sx, sy, dx, fs = _ortho_slope_stack()
    r = run_epss_from_slopes(sx[0], sy[0], dx_m=dx, fs=fs, downsample=2,
                             verbose=False)
    assert r.long_wave_ran is False
    assert np.allclose(r.eta_long, 0.0)
    # single frame promoted to a length-1 stack (check via eta_long length)
    assert r.eta_long.shape[0] == 1


def test_from_slopes_aperture_threads_through():
    sx, sy, dx, fs = _ortho_slope_stack()
    r = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                             aperture_diameter_m=1.0, verbose=False)
    assert r.diag["aperture_diameter_m"] == 1.0
    assert r.diag["aperture_mask"].sum() < r.diag["aperture_mask"].size


def test_from_slopes_force_long_wave_override():
    sx, sy, dx, fs = _ortho_slope_stack(T=20)  # 2 s, normally too short
    r_off = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                                 verbose=False)
    assert r_off.long_wave_ran is False
    r_on = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                                force_long_wave=True, verbose=False)
    assert r_on.long_wave_ran is True


def test_from_slopes_shape_mismatch_raises():
    sx, sy, dx, fs = _ortho_slope_stack()
    with pytest.raises(ValueError, match="same shape"):
        run_epss_from_slopes(sx, sy[..., :10], dx_m=dx, fs=fs, verbose=False)


def test_from_slopes_bad_ndim_raises():
    bad = np.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError, match="Ny, Nx"):
        run_epss_from_slopes(bad, bad, dx_m=0.05, fs=10.0, verbose=False)


def test_from_slopes_nan_input_is_tolerated():
    # NaNs (e.g. ortho no-data border) must be zeroed, not propagate. Opt into
    # short_wave so there is a resolved field to check for finiteness.
    sx, sy, dx, fs = _ortho_slope_stack(T=64)
    sx = sx.copy()
    sx[:, :2, :2] = np.nan
    r = run_epss_from_slopes(sx, sy, dx_m=dx, fs=fs, downsample=2,
                             short_wave=True, verbose=False)
    assert np.isfinite(r.eta_xyt).all()


# ---------------------------------------------------------------------------
# C. Wide-FOV inversion modes + the Pistellato resolution.
# ---------------------------------------------------------------------------
from types import SimpleNamespace

from pss import build_lookup_table, compute_slope_field
from pss.widefov import WideFOVCalibration


def _passthrough_calibration():
    """A calibration whose empirical LUT IS the ideal Fresnel table, so the
    'empirical_wide' path must produce identical slopes to 'fresnel'."""
    lut = build_lookup_table(n_water=1.34)
    return WideFOVCalibration(
        theta_deg=np.array([0.0]), dolp_measured=np.array([0.0]),
        lut_fresnel=lut, lut_empirical=lut, lut_seapol=None, lut_hybrid=None,
        n_water=1.34)


def test_unknown_inversion_raises():
    with pytest.raises(ValueError, match="inversion must be"):
        run_epss(_raw_stack(3), inversion="bogus", verbose=False)


def test_empirical_wide_requires_calibration():
    with pytest.raises(ValueError, match="requires wide_calibration"):
        run_epss(_raw_stack(3), inversion="empirical_wide", verbose=False)


def test_hybrid_missing_lut_raises():
    # Calibration present, but its hybrid LUT is None (seapol absent at build).
    cal = _passthrough_calibration()
    with pytest.raises(ValueError, match="lut_hybrid"):
        run_epss(_raw_stack(3), inversion="hybrid", wide_calibration=cal,
                 verbose=False)


def test_empirical_wide_matches_fresnel_when_lut_is_identical():
    frames = _raw_stack(4)
    cal = _passthrough_calibration()
    r_fre = run_epss(frames, inversion="fresnel", verbose=False)
    r_emp = run_epss(frames, inversion="empirical_wide", wide_calibration=cal,
                     verbose=False)
    assert r_emp.inversion == "empirical_wide"
    assert np.allclose(r_emp.slope_x, r_fre.slope_x, equal_nan=True)
    assert np.allclose(r_emp.slope_y, r_fre.slope_y, equal_nan=True)


def test_pistellato_resolution_requires_geometry():
    frame = _raw_stack(1)[0]
    with pytest.raises(ValueError, match="requires focal_length_m"):
        compute_slope_field(frame, resolution="pistellato")


def test_pistellato_resolution_returns_half_res_result():
    frame = _raw_stack(1, H=64, W=64)[0]
    r = compute_slope_field(frame, resolution="pistellato",
                            focal_length_m=0.005, pixel_pitch_m=3.45e-6)
    assert r.Sx.shape == (32, 32)
