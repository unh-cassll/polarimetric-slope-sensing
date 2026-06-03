"""
Tests for the circular-aperture spatial-mean knob on the long-wave inversion,
and for the shared Krogstad-projection helper extracted into wavelet_core.

All synthetic and fully offline (no Zenodo data): they exercise the new code
paths added when the long-wave inversion gained an `aperture_diameter_m`
option, plus the dedup of the Krogstad signed projection.
"""

from __future__ import annotations

import numpy as np
import pytest

from eta_field_recon.recon import (
    _circular_aperture_mask,
    _aperture_spatial_mean,
    reconstruct_eta_field,
)
from eta_field_recon.wavelet_core import krogstad_eta_coeffs


# ---------------------------------------------------------------------------
# _circular_aperture_mask
# ---------------------------------------------------------------------------

def test_none_diameter_is_full_frame():
    mask = _circular_aperture_mask(32, 48, dx=0.05, diameter_m=None)
    assert mask.shape == (32, 48)
    assert mask.all()
    assert mask.dtype == bool


def test_disc_area_matches_geometry():
    # A disc of diameter D on a grid of spacing dx covers ~ pi*(D/2/dx)^2 cells.
    Ny = Nx = 128
    dx = 0.05
    D = 2.0  # meters
    mask = _circular_aperture_mask(Ny, Nx, dx, D)
    expected = np.pi * (D / 2.0 / dx) ** 2
    got = int(mask.sum())
    # within 2% of the continuous area (boundary discretization)
    assert abs(got - expected) / expected < 0.02


def test_disc_is_centered():
    # The mask's center of mass should sit at the grid center.
    Ny = Nx = 64
    dx = 0.1
    mask = _circular_aperture_mask(Ny, Nx, dx, 3.0)
    rows, cols = np.nonzero(mask)
    assert abs(rows.mean() - (Ny - 1) / 2.0) < 1e-9
    assert abs(cols.mean() - (Nx - 1) / 2.0) < 1e-9


def test_disc_larger_than_frame_clips_to_edges():
    # A diameter larger than the frame WIDTH but smaller than the corner
    # diagonal is allowed and genuinely clips: the mask covers the central
    # cross but not the corners. Frame is 1.6 m wide, diagonal ~2.26 m, so a
    # 2.0 m diameter (radius 1.0 m) includes edge midpoints but excludes the
    # corners (~1.13 m from center). Never raises.
    Ny = Nx = 16
    dx = 0.1  # 1.6 m frame width; corner distance ~= 1.13 m from center
    mask = _circular_aperture_mask(Ny, Nx, dx, 2.0)
    assert not mask[0, 0]                 # corner excluded
    assert mask[Ny // 2, Nx // 2]         # center included
    assert mask[Ny // 2, 0]              # edge midpoint (0.75 m) included


def test_nonpositive_diameter_raises():
    with pytest.raises(ValueError):
        _circular_aperture_mask(16, 16, 0.1, 0.0)
    with pytest.raises(ValueError):
        _circular_aperture_mask(16, 16, 0.1, -1.0)


# ---------------------------------------------------------------------------
# _aperture_spatial_mean
# ---------------------------------------------------------------------------

def test_full_mask_equals_plain_mean_exactly():
    rng = np.random.RandomState(1)
    stack = rng.randn(7, 20, 24)
    mask = _circular_aperture_mask(20, 24, 0.05, None)
    got = _aperture_spatial_mean(stack, mask)
    expected = stack.mean(axis=(1, 2))
    # exact, not approximate: the all-True path delegates to .mean directly
    assert np.array_equal(got, expected)


def test_aperture_ignores_out_of_disc_values():
    # Put a huge spike outside a central disc; the disc mean must ignore it
    # while the full-frame mean is dominated by it.
    Ny = Nx = 64
    dx = 0.05
    stack = np.zeros((3, Ny, Nx))
    stack[:, :4, :4] = 1e6  # corner spike, outside a central 1 m disc
    full = _circular_aperture_mask(Ny, Nx, dx, None)
    disc = _circular_aperture_mask(Ny, Nx, dx, 1.0)
    assert _aperture_spatial_mean(stack, full).mean() > 1.0  # corrupted
    assert _aperture_spatial_mean(stack, disc).mean() == 0.0  # clean


def test_aperture_mean_of_constant_field_is_that_constant():
    stack = np.full((4, 30, 30), 2.5)
    disc = _circular_aperture_mask(30, 30, 0.1, 1.5)
    got = _aperture_spatial_mean(stack, disc)
    assert np.allclose(got, 2.5)


# ---------------------------------------------------------------------------
# reconstruct_eta_field: aperture default is unchanged; finite aperture differs
# ---------------------------------------------------------------------------

def _synthetic_slope_stack(T=64, Ny=32, Nx=32, fs=10.0, dx=0.05, seed=0):
    """A small slope stack with a coherent long-wave tilt plus corner noise.

    The spatial-mean slope carries a slow oscillation (the long wave); a large
    static spike in one corner lets the aperture demonstrably change the mean.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(T) / fs
    # slow coherent tilt, same across the frame (a long wave)
    tilt = 0.02 * np.sin(2 * np.pi * 0.15 * t)
    sx = np.broadcast_to(tilt[:, None, None], (T, Ny, Nx)).copy()
    sy = 0.5 * sx
    # small incoherent short-wave roughness
    sx += 0.001 * rng.randn(T, Ny, Nx)
    sy += 0.001 * rng.randn(T, Ny, Nx)
    # a persistent corner artifact outside any central disc
    sx[:, :4, :4] += 0.5
    return sx, sy, dx, fs


def test_reconstruct_default_aperture_matches_explicit_none():
    sx, sy, dx, fs = _synthetic_slope_stack()
    out_default = reconstruct_eta_field(sx, sy, dx, fs, downsample=2,
                                        verbose=False)
    out_none = reconstruct_eta_field(sx, sy, dx, fs, downsample=2,
                                     aperture_diameter_m=None, verbose=False)
    # eta_long identical between the implicit and explicit full-frame cases
    assert np.array_equal(out_default[1], out_none[1])


def test_reconstruct_finite_aperture_changes_eta_long():
    sx, sy, dx, fs = _synthetic_slope_stack()
    _, eta_long_full, _, _, _ = reconstruct_eta_field(
        sx, sy, dx, fs, downsample=2, verbose=False)
    _, eta_long_disc, _, _, diag = reconstruct_eta_field(
        sx, sy, dx, fs, downsample=2, aperture_diameter_m=0.8, verbose=False)
    # the corner artifact biases the full-frame mean but not the disc mean,
    # so the two long-wave series must differ
    assert not np.allclose(eta_long_full, eta_long_disc)
    # diag exposes the aperture for inspection
    assert diag["aperture_diameter_m"] == 0.8
    assert diag["aperture_mask"].sum() < diag["aperture_mask"].size


def test_reconstruct_aperture_does_not_touch_eta_short():
    # eta_short is per-frame spatial integration and must be independent of
    # the aperture (which only governs the long-wave spatial-mean slope).
    sx, sy, dx, fs = _synthetic_slope_stack()
    _, _, eta_short_full, _, _ = reconstruct_eta_field(
        sx, sy, dx, fs, downsample=2, verbose=False)
    _, _, eta_short_disc, _, _ = reconstruct_eta_field(
        sx, sy, dx, fs, downsample=2, aperture_diameter_m=0.8, verbose=False)
    assert np.array_equal(eta_short_full, eta_short_disc)


# ---------------------------------------------------------------------------
# krogstad_eta_coeffs (the extracted shared helper)
# ---------------------------------------------------------------------------

def test_krogstad_shapes_and_finiteness():
    nf, T = 12, 50
    rng = np.random.RandomState(2)
    Wsx = rng.randn(nf, T) + 1j * rng.randn(nf, T)
    Wsy = rng.randn(nf, T) + 1j * rng.randn(nf, T)
    k = np.linspace(0.1, 5.0, nf)
    W_eta, cos_th, sin_th = krogstad_eta_coeffs(Wsx, Wsy, k)
    assert W_eta.shape == (nf, T)
    assert cos_th.shape == (nf, T)
    assert sin_th.shape == (nf, T)
    assert np.isfinite(W_eta).all()
    # direction cosines satisfy cos^2 + sin^2 == 1 (up to the eps guard)
    assert np.allclose(cos_th ** 2 + sin_th ** 2, 1.0, atol=1e-6)


def test_krogstad_nonfinite_k_zeroed():
    # k = NaN (e.g. omega = 0) must yield W_eta = 0 at that frequency, not NaN.
    nf, T = 4, 10
    Wsx = np.ones((nf, T), dtype=complex)
    Wsy = np.ones((nf, T), dtype=complex)
    k = np.array([np.nan, 1.0, 2.0, np.inf])
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, k)
    assert np.isfinite(W_eta).all()
    assert np.all(W_eta[0] == 0.0)   # NaN k -> zeroed
    assert np.all(W_eta[3] == 0.0)   # inf k -> 1/inf = 0


def test_krogstad_sign_guard_keeps_along_look_wave():
    # A wave traveling exactly along-look has Wsx == 0, so the relative-phase
    # sign is indeterminate. The guard must keep the along-look magnitude
    # (default sign +1) rather than zeroing it.
    nf, T = 3, 8
    Wsx = np.zeros((nf, T), dtype=complex)
    Wsy = (1.0 + 0.0j) * np.ones((nf, T))
    k = np.full(nf, 2.0)
    W_eta, cos_th, sin_th = krogstad_eta_coeffs(Wsx, Wsy, k)
    # sin_th should be +1 (kept), not 0 (destroyed)
    assert np.allclose(sin_th, 1.0)
    assert np.allclose(cos_th, 0.0)
    assert not np.allclose(W_eta, 0.0)
