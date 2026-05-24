"""Tests for the DoLP gain modes (none / lab / empirical)."""

from __future__ import annotations

import numpy as np
import pytest

from pss import apply_gain, DEFAULT_LAB_GAIN
from pss.fresnel import fresnel_dolp


@pytest.fixture
def stokes_pair() -> tuple[np.ndarray, np.ndarray]:
    """Simple s1, s2 fields with a known median DoLP."""
    rng = np.random.default_rng(0)
    s1 = rng.uniform(0.1, 0.3, size=(64, 64))
    s2 = rng.uniform(0.1, 0.3, size=(64, 64))
    return s1, s2


def test_none_mode_returns_input_unchanged(stokes_pair):
    s1, s2 = stokes_pair
    r = apply_gain(s1, s2, mode="none")
    np.testing.assert_array_equal(r.s1, s1)
    np.testing.assert_array_equal(r.s2, s2)
    assert r.g1 == 1.0 and r.g2 == 1.0
    assert r.mode == "none"


def test_lab_mode_applies_default_gains(stokes_pair):
    s1, s2 = stokes_pair
    r = apply_gain(s1, s2, mode="lab")
    g1_expected, g2_expected = DEFAULT_LAB_GAIN
    np.testing.assert_array_equal(r.s1, s1 * g1_expected)
    np.testing.assert_array_equal(r.s2, s2 * g2_expected)
    assert r.g1 == pytest.approx(g1_expected)
    assert r.g2 == pytest.approx(g2_expected)


def test_lab_mode_accepts_custom_gains(stokes_pair):
    s1, s2 = stokes_pair
    r = apply_gain(s1, s2, mode="lab", lab_gain=(2.0, 3.0))
    np.testing.assert_array_equal(r.s1, s1 * 2.0)
    np.testing.assert_array_equal(r.s2, s2 * 3.0)


def test_empirical_mode_aligns_median_dolp_to_fresnel_ideal(stokes_pair):
    """The empirical gain is defined as DoLP_ideal / median(DoLP_obs); after
    applying it, the new median DoLP should equal the Fresnel ideal. (We pass
    the same field as the reference to validate the formula directly.)"""
    s1, s2 = stokes_pair
    theta = 30.0
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=theta,
                   s1_ref=s1, s2_ref=s2)
    new_median = np.median(np.sqrt(r.s1**2 + r.s2**2))
    ideal = fresnel_dolp(theta, n_water=1.34)
    assert new_median == pytest.approx(ideal, rel=1e-10)


def test_empirical_mode_requires_theta(stokes_pair):
    s1, s2 = stokes_pair
    with pytest.raises(ValueError, match="theta_i_mean_deg"):
        apply_gain(s1, s2, mode="empirical")


def test_empirical_gain_clipped_when_extreme(stokes_pair):
    """A degenerate input (all-zero DoLP) would produce infinite gain;
    instead, the empirical-mode should refuse it."""
    s1 = np.full((16, 16), 1e-9)
    s2 = np.full((16, 16), 1e-9)
    # Default clip is (0.5, 3.0); a near-zero observed DoLP yields enormous g
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=30.0,
                   s1_ref=s1, s2_ref=s2)
    assert r.g1 == 3.0   # hit upper clip
    assert "CLIPPED" in r.notes


def test_empirical_zero_input_raises(stokes_pair):
    """Exactly zero observed DoLP can't be corrected."""
    s1 = np.zeros((16, 16))
    s2 = np.zeros((16, 16))
    with pytest.raises(ValueError, match="non-positive"):
        apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=30.0,
                   s1_ref=s1, s2_ref=s2)


def test_unknown_mode_rejected(stokes_pair):
    s1, s2 = stokes_pair
    with pytest.raises(ValueError, match="unknown gain mode"):
        apply_gain(s1, s2, mode="rubber_chicken")


def test_dolp_invariance_when_gain_modes_only_change_scale(stokes_pair):
    """The lab mode applies different scalars to s1 and s2, but in the limit
    g1==g2 it should be equivalent to scaling DoLP by that scalar."""
    s1, s2 = stokes_pair
    g = 1.5
    r = apply_gain(s1, s2, mode="lab", lab_gain=(g, g))
    dolp_old = np.sqrt(s1**2 + s2**2)
    dolp_new = np.sqrt(r.s1**2 + r.s2**2)
    np.testing.assert_allclose(dolp_new, dolp_old * g, rtol=1e-12)


# ----------------------------------------------------------------------------
# Empirical gain with an external reference (E-PSS median-frame workflow).
# ----------------------------------------------------------------------------

def test_empirical_dolp_obs_median_used_when_no_frame(stokes_pair):
    """A pre-computed dolp_obs_median drives the gain when no reference frame
    is supplied (the single-frame fallback path)."""
    s1, s2 = stokes_pair
    theta = 30.0
    dolp_ideal = fresnel_dolp(np.array([theta]))[0]
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=theta,
                   dolp_obs_median=0.25)
    assert r.g1 == pytest.approx(dolp_ideal / 0.25, rel=1e-9)
    assert "ref=precomputed" in r.notes


def test_empirical_reference_frame_beats_precomputed(stokes_pair):
    """When BOTH a reference frame and a precomputed scalar are given, the
    temporal-median reference frame wins."""
    s1, s2 = stokes_pair
    s1_ref = np.full_like(s1, 0.20)
    s2_ref = np.zeros_like(s2)          # DoLP_ref == 0.20
    theta = 30.0
    dolp_ideal = fresnel_dolp(np.array([theta]))[0]
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=theta,
                   s1_ref=s1_ref, s2_ref=s2_ref, dolp_obs_median=0.99)
    assert r.g1 == pytest.approx(dolp_ideal / 0.20, rel=1e-9)  # used 0.20, not 0.99
    assert "ref=temporal-median frame" in r.notes


def test_empirical_external_reference_frame_sets_gain(stokes_pair):
    """s1_ref/s2_ref should supply the observed-DoLP median, not s1/s2."""
    s1, s2 = stokes_pair
    # Reference with a deliberately different DoLP level than s1/s2.
    s1_ref = np.full_like(s1, 0.20)
    s2_ref = np.zeros_like(s2)            # DoLP_ref == 0.20 everywhere
    theta = 30.0
    dolp_ideal = fresnel_dolp(np.array([theta]))[0]
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=theta,
                   s1_ref=s1_ref, s2_ref=s2_ref)
    assert r.g1 == pytest.approx(dolp_ideal / 0.20, rel=1e-9)
    assert "ref=temporal-median frame" in r.notes
    # And the gain is actually applied to s1/s2 (not the reference).
    np.testing.assert_allclose(r.s1, s1 * r.g1, rtol=1e-12)


def test_empirical_no_reference_falls_back_to_no_gain(stokes_pair):
    """With NO reference supplied, the empirical gain must NOT self-reference
    (the single-frame flat-surface assumption is invalid). It falls back to
    no gain, leaving s1/s2 untouched, and says so."""
    s1, s2 = stokes_pair
    r = apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=30.0)
    assert r.g1 == 1.0 and r.g2 == 1.0
    assert r.mode == "none"
    np.testing.assert_array_equal(r.s1, s1)
    np.testing.assert_array_equal(r.s2, s2)
    assert "no reference" in r.notes.lower()


def test_empirical_partial_reference_raises(stokes_pair):
    """Supplying only one of s1_ref/s2_ref is an error."""
    s1, s2 = stokes_pair
    with pytest.raises(ValueError, match="together"):
        apply_gain(s1, s2, mode="empirical", theta_i_mean_deg=30.0,
                   s1_ref=s1)
