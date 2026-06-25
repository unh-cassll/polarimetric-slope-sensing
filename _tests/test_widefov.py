"""Tests for the wide-FOV DoLP->AOI calibration and its empirical LUT builder.

Synthetic and offline. The seapol/hybrid LUTs are exercised only when seapol is
importable; otherwise they must degrade to None with a note (no error).
"""

from __future__ import annotations

import numpy as np
import pytest

from pss.fresnel import build_lookup_table, dolp_to_aoi, fresnel_dolp, lut_from_curve
from pss.widefov import calibrate_widefov, _HAS_SEAPOL


def test_lut_from_curve_format_and_monotonicity():
    aoi = np.linspace(2.0, 50.0, 200)              # rising branch (n=1.34)
    dolp = fresnel_dolp(aoi, n_water=1.34)
    DOLP_full, theta_full = lut_from_curve(aoi, dolp)
    assert DOLP_full.shape == (10_000,) and theta_full.shape == (10_000,)
    assert np.allclose(DOLP_full, np.linspace(0, 1, 10_000))
    assert theta_full[0] == 0.0
    # non-decreasing on the rising branch
    assert np.all(np.diff(theta_full) >= -1e-9)


def test_lut_from_curve_keeps_rising_branch_only():
    """A non-monotone DoLP(theta) (rises to Brewster, then falls) must be
    reduced to the rising branch and still invert correctly."""
    aoi = np.linspace(2.0, 80.0, 400)              # crosses Brewster (~53 deg)
    dolp = fresnel_dolp(aoi, n_water=1.34)
    assert dolp.argmax() < len(dolp) - 1           # genuinely non-monotone input
    D, T = lut_from_curve(aoi, dolp)
    # A point on the rising branch round-trips to its incidence angle.
    aoi_test = 35.0
    d_test = fresnel_dolp(aoi_test, n_water=1.34)
    rec = dolp_to_aoi(np.array([d_test]), D, T)[0]
    assert abs(rec - aoi_test) < 1.0


def _synthetic_wide_frame(H=400, W=200, dolp_peak_row_scale=1.0, seed=0):
    """A wide-FOV mean frame whose DoLP rises down the rows (toward grazing).

    Built from orthographic Malus with a DoLP that increases with row index, so
    the per-row profile is a clean rising curve for the LUT builder.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(H)
    # DoLP ramps 0.05 -> 0.6 down the frame; orientation fixed (phi=0).
    dolp_row = np.clip(0.05 + 0.55 * rows / (H - 1), 0, 1) * dolp_peak_row_scale
    DoLP = np.repeat(dolp_row[:, None], W, axis=1)
    S0 = np.full((H, W), 1000.0)
    s1 = DoLP            # phi = 0 -> all polarized power in s1
    s2 = np.zeros_like(DoLP)
    S1, S2 = s1 * S0, s2 * S0

    from pss.stokes import _OFFSETS
    angle = {"I0": 0.0, "I45": 45.0, "I90": 90.0, "I135": 135.0}
    frame = np.zeros((H, W))
    for name, (r, c) in _OFFSETS.items():
        a = np.deg2rad(angle[name])
        I = 0.5 * (S0 + S1 * np.cos(2 * a) + S2 * np.sin(2 * a))
        frame[r::2, c::2] = I[r::2, c::2]
    frame += rng.normal(0, 0.2, frame.shape)
    return frame


def test_calibrate_widefov_builds_fresnel_and_empirical():
    frame = _synthetic_wide_frame()
    cal = calibrate_widefov(
        frame, focal_length_m=0.005, pixel_pitch_m=3.45e-6,
        incidence_mean_deg=40.0, n_water=1.34, row_sign=1.0, verbose=False)
    # Always-available tables are finite and in the canonical format.
    for lut in (cal.lut_fresnel, cal.lut_empirical):
        D, T = lut
        assert D.shape == (10_000,) and T.shape == (10_000,)
        assert np.all(np.isfinite(D))
    assert np.isfinite(cal.dolp_measured).any()


def test_calibrate_widefov_seapol_optional():
    """Without seapol (or without sun geometry) the seapol/hybrid LUTs are
    None and a note explains why; never an error."""
    frame = _synthetic_wide_frame()
    cal = calibrate_widefov(
        frame, focal_length_m=0.005, pixel_pitch_m=3.45e-6,
        incidence_mean_deg=40.0, row_sign=1.0, verbose=False)   # no sun geometry passed
    assert cal.lut_seapol is None and cal.lut_hybrid is None
    assert cal.notes


@pytest.mark.skipif(not _HAS_SEAPOL, reason="seapol not installed")
def test_calibrate_widefov_with_seapol_builds_all_four():
    frame = _synthetic_wide_frame()
    cal = calibrate_widefov(
        frame, focal_length_m=0.005, pixel_pitch_m=3.45e-6,
        incidence_mean_deg=40.0, sun_zenith_deg=35.0, sun_azimuth_deg=150.0,
        heading_deg=350.0, row_sign=1.0, verbose=False)
    assert cal.lut_seapol is not None and cal.lut_hybrid is not None
    assert cal.hybrid_params is not None
