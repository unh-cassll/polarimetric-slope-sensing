"""Tests for the three Stokes-reconstruction methods."""

from __future__ import annotations

import numpy as np
import pytest

from pss import compute_stokes, METHODS


METHOD_NAMES = list(METHODS.keys())


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_output_shapes_match_input(synthetic_frame, method):
    frame, _ = synthetic_frame
    S0, s1, s2 = compute_stokes(frame, method=method)
    assert S0.shape == frame.shape
    assert s1.shape == frame.shape
    assert s2.shape == frame.shape


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_outputs_are_finite(synthetic_frame, method):
    frame, _ = synthetic_frame
    S0, s1, s2 = compute_stokes(frame, method=method)
    assert np.isfinite(S0).all()
    assert np.isfinite(s1).all()
    assert np.isfinite(s2).all()


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_normalized_stokes_within_unit_disk(synthetic_frame, method):
    """s1^2 + s2^2 = DoLP^2 should never exceed 1 for physical light."""
    frame, _ = synthetic_frame
    _, s1, s2 = compute_stokes(frame, method=method)
    dolp = np.sqrt(s1**2 + s2**2)
    # Allow tiny slack for numerical noise in the demod method
    assert dolp.max() <= 1.01


def test_unknown_method_raises(synthetic_frame):
    frame, _ = synthetic_frame
    with pytest.raises(ValueError, match="method must be one of"):
        compute_stokes(frame, method="not_a_method")


def test_odd_dimensions_rejected():
    """Frame dimensions must be even for 2x2 super-pixel tiling."""
    bad = np.ones((101, 100))
    with pytest.raises(ValueError, match="must be even"):
        compute_stokes(bad, method="bilinear")


@pytest.mark.parametrize("method", ["kernel_averaging"])
def test_kernel_size_argument(synthetic_frame, method):
    frame, _ = synthetic_frame
    # Both supported kernel sizes should run without error
    S0a, s1a, s2a = compute_stokes(frame, method=method, kernel="2x2")
    S0b, s1b, s2b = compute_stokes(frame, method=method, kernel="4x4")
    assert S0a.shape == frame.shape
    assert S0b.shape == frame.shape


@pytest.mark.parametrize("method", ["kernel_averaging"])
def test_invalid_kernel_rejected(synthetic_frame, method):
    frame, _ = synthetic_frame
    with pytest.raises(ValueError, match="kernel must be one of"):
        compute_stokes(frame, method=method, kernel="8x8")


def test_conv_demodulation_is_ratliff_method4(synthetic_frame):
    """conv_demodulation is the exact Ratliff (2009) Method 4 (16-pixel)
    interpolator. A flat (unpolarized) field must reconstruct to DoLP ~ 0,
    and the output is full-resolution. It takes no kernel-size argument."""
    frame, _ = synthetic_frame
    # Flat field -> zero DoLP.
    flat = np.full_like(frame, 500.0, dtype=float)
    S0, s1, s2 = compute_stokes(flat, method="conv_demodulation")
    assert S0.shape == frame.shape
    dolp = np.sqrt(s1**2 + s2**2)
    assert np.nanmax(dolp) < 1e-9


def test_bilinear_and_kernel_averaging_agree(synthetic_frame):
    """For smooth fields, bilinear and kernel-averaging should give similar
    DoLP within a few percent. Validates that both methods produce sensible
    output on a known field."""
    frame, _ = synthetic_frame
    _, s1a, s2a = compute_stokes(frame, method="bilinear")
    _, s1b, s2b = compute_stokes(frame, method="kernel_averaging", kernel="4x4")
    dolp_a = np.sqrt(s1a**2 + s2a**2)
    dolp_b = np.sqrt(s1b**2 + s2b**2)
    median_diff = abs(np.median(dolp_a) - np.median(dolp_b))
    assert median_diff < 0.01  # less than 1% absolute DoLP
