"""
Regression tests for the E-PSS median-frame empirical-gain workflow.

Exercises two things added for the single-frame + median-gain task:

  1. read_netcdf_frame() transparently handling the stack schema
     (raw_frame dims (time, x, y); superpixel_layout dims
     (super_col, super_row)) -> a 2-D (y, x) frame with the correct L0 layout.

  2. compute_slope_field(..., gain_reference_frame=median) deriving the
     empirical DoLP gain from the temporal-median frame and applying it to an
     individual frame.

These pin the reduced numbers for asit_2019_raw_pol_frame0001.nc calibrated against
asit_2019_raw_pol_median.nc. If a change moves them, something broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from pss import compute_slope_field, read_netcdf_frame
from pss.stokes import _OFFSETS

# Pinned regression values (DEFAULT settings: native half-resolution
# reduction, empirical gain from median frame, theta_i = 30 deg,
# n_water = 1.34).
EXPECTED_GAIN = 1.5587
EXPECTED_MEDIAN_DOLP = 0.3309      # this frame, post-gain (NOT the ideal 0.4406)
EXPECTED_MEDIAN_AOI = 26.099       # deg
EXPECTED_MSS = 0.022729            # dimensionless: var(Sx) + var(Sy)
EXPECTED_DOLP_OBS_MEDIAN = 0.2827  # from the median reference frame (native)


def test_reader_handles_stack_schema(frame_stack_nc_path):
    """The (time, x, y) frame should come back as 2-D (y, x) with L0 layout."""
    frame, meta = read_netcdf_frame(frame_stack_nc_path)
    assert frame.ndim == 2
    assert frame.shape == (2056, 2464)          # (y, x), not the stored (x, y)
    # Layout parsed from the transposed (super_col, super_row) grid == L0.
    assert meta.layout == _OFFSETS
    assert meta.theta_i_mean_deg == pytest.approx(30.0)
    assert meta.framerate_hz == pytest.approx(30.0)


def test_reader_time_index_out_of_range(frame_stack_nc_path):
    """A bad time index is rejected rather than silently wrapping."""
    with pytest.raises(IndexError):
        read_netcdf_frame(frame_stack_nc_path, time_index=5)


def test_median_frame_dolp_obs(median_nc_path):
    """The median reference frame yields the expected observed-DoLP median,
    which sets the empirical gain."""
    median_frame, _ = read_netcdf_frame(median_nc_path)
    from pss import by_superpixel
    _, s1, s2 = by_superpixel(median_frame)  # native default reduction
    dolp_obs = float(np.nanmedian(np.sqrt(s1 * s1 + s2 * s2)))
    assert dolp_obs == pytest.approx(EXPECTED_DOLP_OBS_MEDIAN, abs=1e-4)


def test_frame0001_reduced_with_median_gain(frame_stack_nc_path, median_nc_path):
    """End-to-end: reduce frame0001 with the median-derived empirical gain."""
    frame, meta = read_netcdf_frame(frame_stack_nc_path)
    median_frame, _ = read_netcdf_frame(median_nc_path)

    result = compute_slope_field(
        frame,
        gain_mode="empirical",  # resolution/method left at package defaults
        theta_i_mean_deg=meta.theta_i_mean_deg,
        n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )

    assert result.gain_g1 == pytest.approx(EXPECTED_GAIN, abs=1e-4)
    assert result.gain_g1 == result.gain_g2          # single scalar, both comps
    assert "ref=temporal-median frame" in result.gain_notes
    assert float(np.nanmedian(result.dolp)) == pytest.approx(
        EXPECTED_MEDIAN_DOLP, abs=1e-4)
    assert float(np.nanmedian(result.aoi_deg)) == pytest.approx(
        EXPECTED_MEDIAN_AOI, abs=1e-3)
    assert result.mss == pytest.approx(EXPECTED_MSS, abs=1e-5)


def test_empirical_without_reference_falls_back_to_no_gain(frame_stack_nc_path, median_nc_path):
    """Self-referencing is gone: empirical gain with NO reference frame must
    fall back to no gain (never force this frame's median DoLP onto the
    Fresnel ideal). The median-referenced result must differ from it."""
    frame, meta = read_netcdf_frame(frame_stack_nc_path)
    median_frame, _ = read_netcdf_frame(median_nc_path)

    r_median = compute_slope_field(
        frame, gain_mode="empirical",
        theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )
    r_noref = compute_slope_field(
        frame, gain_mode="empirical",
        theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
        # no gain_reference_frame -> must fall back to no gain
    )
    r_none = compute_slope_field(frame, gain_mode="none")

    # No-reference empirical == no gain (g=1, mode downgraded, identical mss).
    assert r_noref.gain_g1 == pytest.approx(1.0)
    assert r_noref.gain_mode == "none"
    assert "no reference" in r_noref.gain_notes.lower()
    assert r_noref.mss == pytest.approx(r_none.mss, rel=1e-12)
    # The median-referenced gain genuinely applies a correction (g != 1) and
    # leaves the frame's DoLP below the Fresnel ideal (not forced onto it).
    assert r_median.gain_g1 > 1.0
    assert float(np.nanmedian(r_median.dolp)) < 0.4406 - 0.05
