"""Tests for the DoLP <-> theta_i lookup table."""

from __future__ import annotations

import numpy as np
import pytest

from pss.fresnel import build_lookup_table, dolp_to_aoi, fresnel_dolp


def test_fresnel_dolp_endpoints():
    """At theta=0, DoLP is exactly 0 (specular reflection of unpolarized light
    at normal incidence is unpolarized). At theta=90 deg the numerator goes
    to zero too."""
    assert fresnel_dolp(0.0) == pytest.approx(0.0)
    assert fresnel_dolp(89.99) == pytest.approx(0.0, abs=1e-3)


def test_fresnel_dolp_peaks_near_brewster_for_water():
    """For water (n=1.34), Brewster's angle is arctan(1.34) ~ 53.27 deg, and
    DoLP reaches 1 there."""
    brewster = np.degrees(np.arctan(1.34))
    assert fresnel_dolp(brewster, n_water=1.34) == pytest.approx(1.0, rel=1e-3)


def test_fresnel_dolp_is_increasing_below_brewster():
    """Rising branch is monotonic."""
    angles = np.linspace(1.0, 50.0, 100)
    dolp = fresnel_dolp(angles, n_water=1.34)
    assert np.all(np.diff(dolp) > 0)


def test_lookup_table_shape_and_endpoints():
    DOLP, theta = build_lookup_table(n_points=1000)
    assert DOLP.shape == (1000,)
    assert theta.shape == (1000,)
    # DoLP grid uniform on [0, 1]
    np.testing.assert_allclose(DOLP[0],  0.0, atol=1e-12)
    np.testing.assert_allclose(DOLP[-1], 1.0, atol=1e-12)
    # theta at DoLP=0 is 0; at DoLP=1 is at Brewster
    assert theta[0] == pytest.approx(0.0, abs=1e-3)
    brewster = np.degrees(np.arctan(1.34))
    assert theta[-1] == pytest.approx(brewster, abs=0.5)


def test_lookup_table_roundtrip():
    """For each theta on the rising branch, lookup(fresnel_dolp(theta)) should
    recover theta to within the table's PCHIP resolution."""
    DOLP, theta_full = build_lookup_table(n_points=10000)
    test_angles = np.linspace(5.0, 50.0, 50)
    test_dolp = fresnel_dolp(test_angles, n_water=1.34)
    recovered = dolp_to_aoi(test_dolp, DOLP, theta_full)
    np.testing.assert_allclose(recovered, test_angles, atol=0.1)  # 0.1 deg slack


def test_dolp_to_aoi_clips_out_of_range():
    """DoLP values outside [0, 1] should map without crashing."""
    DOLP, theta_full = build_lookup_table(n_points=1000)
    # Just check no IndexError; the values themselves are clamped.
    out = dolp_to_aoi(np.array([-0.5, 0.0, 0.5, 1.0, 1.5]), DOLP, theta_full)
    assert np.isfinite(out).all()
    assert out.shape == (5,)


def test_load_lookup_table_vec_fallback_clamps_low_dolp(tmp_path):
    """A DOLP_vec/theta_vec .mat whose measured curve starts above DoLP=0 must
    map low DoLP to the SMALL-angle end, not to the peak (regression: a single
    nan fill sent near-flat water to Brewster's angle), and must tolerate
    repeated DoLP samples before the peak."""
    from scipy.io import savemat

    from pss.fresnel import load_lookup_table

    theta_vec = np.linspace(20.0, 70.0, 40)
    dolp_vec = fresnel_dolp(theta_vec, n_water=1.34)
    dolp_vec[5] = dolp_vec[4]          # duplicated sample: must not raise
    path = tmp_path / "lut.mat"
    savemat(path, {"DOLP_vec": dolp_vec, "theta_vec": theta_vec})

    DOLP_full, theta_full = load_lookup_table(path)
    aoi = dolp_to_aoi(np.array([0.001, 0.05, dolp_vec.min()]),
                      DOLP_full, theta_full)
    # Everything at/below the measured floor clamps to the smallest measured
    # angle -- far from the ~53 deg Brewster peak.
    assert (aoi <= theta_vec[0] + 1.0).all()
