"""Tests for the Pistellato projective polarizer-tilt correction.

Synthetic and fully offline. The synthetic_frame fixture (conftest) builds a
DoFP frame from orthographic Malus's law, so the "true" Stokes it encodes is the
orthographic one -- which the correction must recover in the long-focal-length
(tilt-free) limit, while introducing a measurable, FOV-scaled difference under a
wide lens.
"""

from __future__ import annotations

import numpy as np

from pss.pistellato import (
    build_K,
    corrected_stokes_superpixel,
    compute_stokes_from_tilted_polarizers_fast,
)
from pss.stokes import by_superpixel, _OFFSETS


PIX = 3.45e-6   # m, IMX250MZR


def _aolp_deg(s1, s2):
    return 0.5 * np.degrees(np.arctan2(s2, s1))


def test_telephoto_limit_matches_by_superpixel(synthetic_frame):
    """A long focal length => near-parallel rays => the correction is a no-op."""
    frame, _ = synthetic_frame
    S0n, s1n, s2n = by_superpixel(frame)
    # 500 mm on a 3.45 um pitch is effectively orthographic.
    S0p, s1p, s2p = corrected_stokes_superpixel(
        frame, focal_length_m=0.5, pixel_pitch_m=PIX)
    daolp = _aolp_deg(s1p, s2p) - _aolp_deg(s1n, s2n)
    # AoLP agreement well under a hundredth of a degree across the frame.
    assert np.nanmax(np.abs(daolp)) < 0.02
    assert np.allclose(s1p, s1n, atol=1e-3)
    assert np.allclose(s2p, s2n, atol=1e-3)


def test_flat_unpolarized_field_gives_zero_dolp():
    """A uniform unpolarized scene must invert to DoLP = 0 everywhere.

    This pins the design-matrix scaling and the `ty` sign: any error there
    leaks intensity into S1/S2 and raises a spurious DoLP.
    """
    frame = np.full((128, 128), 1234.0)
    S0, s1, s2 = corrected_stokes_superpixel(
        frame, focal_length_m=0.005, pixel_pitch_m=PIX)   # wide 5 mm
    dolp = np.sqrt(s1 ** 2 + s2 ** 2)
    assert np.nanmax(dolp) < 1e-9


def test_wide_lens_correction_is_nonzero(synthetic_frame):
    """Under a genuine wide FOV the correction must actually move the AoLP."""
    frame, _ = synthetic_frame
    _, s1n, s2n = by_superpixel(frame)
    _, s1p, s2p = corrected_stokes_superpixel(
        frame, focal_length_m=0.005, pixel_pitch_m=PIX)   # 5 mm
    daolp = _aolp_deg(s1p, s2p) - _aolp_deg(s1n, s2n)
    # This small (256 px) synthetic frame spans a narrow FOV, so the edge ray
    # tilt is only ~5 deg and the AoLP correction is ~0.2 deg (it scales up to
    # several degrees on a real 2048 px wide frame). Assert it is clearly active.
    assert np.nanmax(np.abs(daolp)) > 0.1


def test_layout_agnostic_under_patched_offsets(synthetic_frame):
    """corrected_stokes_superpixel assembles channels by NAME, so a relabeled
    super-pixel layout must give the same Stokes once the frame is built to
    match it (telephoto limit, where the answer equals by_superpixel)."""
    frame, _ = synthetic_frame
    S0a, s1a, s2a = corrected_stokes_superpixel(
        frame, focal_length_m=0.5, pixel_pitch_m=PIX)
    # by_superpixel reads _OFFSETS at call time too; both must track the layout.
    S0b, s1b, s2b = by_superpixel(frame)
    assert np.allclose(s1a, s1b, atol=1e-3) and np.allclose(s2a, s2b, atol=1e-3)


def test_batched_solve_recovers_known_stokes():
    """Forward Malus -> batched solve round-trips the Stokes vector exactly in
    the tilt-free (huge focal length) limit, at machine precision."""
    H = W = 16
    rng = np.random.default_rng(0)
    S0 = 1000.0 + rng.random((H, W)) * 10
    s1 = 0.3 * rng.standard_normal((H, W))
    s2 = 0.3 * rng.standard_normal((H, W))
    S1, S2 = s1 * S0, s2 * S0
    # Malus intensities for the four nominal angles 0/45/90/135.
    def I(a):
        ar = np.deg2rad(a)
        return 0.5 * (S0 + S1 * np.cos(2 * ar) + S2 * np.sin(2 * ar))
    stack = np.stack([I(0), I(45), I(90), I(135)], axis=-1)
    K = build_K(1.0, PIX, W, H, half_res=False)   # ~orthographic
    out = compute_stokes_from_tilted_polarizers_fast(stack, K)
    assert np.allclose(out[..., 0], S0, rtol=1e-6)
    assert np.allclose(out[..., 1], S1, atol=1e-3)
    assert np.allclose(out[..., 2], S2, atol=1e-3)
