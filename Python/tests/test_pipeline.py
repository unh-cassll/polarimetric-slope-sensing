"""End-to-end pipeline tests using both the synthetic frame and the bundled
ASIT2019 example NetCDF."""

from __future__ import annotations

import numpy as np
import pytest

from pss import (
    apply_layout_from_meta,
    compute_slope_field,
    read_netcdf_frame,
)
from pss.fresnel import fresnel_dolp


# ---------------------------------------------------------------------------
# Synthetic-frame tests (run everywhere; no external data required)
# ---------------------------------------------------------------------------

def test_pipeline_runs_on_synthetic_frame(synthetic_frame):
    frame, truth = synthetic_frame
    # Default is now native (half-resolution): one Stokes vector per super-pixel.
    result = compute_slope_field(frame, gain_mode="none")
    H, W = truth["shape"]
    assert result.s1.shape == (H // 2, W // 2)
    assert np.isfinite(result.dolp).all()
    assert np.isfinite(result.aoi_deg).all()
    assert np.isfinite(result.mss)
    # resolution="full" restores the interpolated full-resolution grid.
    result_full = compute_slope_field(
        frame, resolution="full", method="bilinear", gain_mode="none"
    )
    assert result_full.s1.shape == truth["shape"]


def test_empirical_gain_pins_median_dolp_to_fresnel_ideal(synthetic_frame):
    frame, truth = synthetic_frame
    theta = truth["theta_i_mean_deg"]
    # Validate the gain FORMULA by self-referencing explicitly (pass the frame
    # as its own reference). This is a math check, not the recommended workflow.
    result = compute_slope_field(
        frame, resolution="full", method="bilinear", gain_mode="empirical",
        theta_i_mean_deg=theta, n_water=truth["n_water"],
        gain_reference_frame=frame,
    )
    ideal = fresnel_dolp(theta, n_water=truth["n_water"])
    assert np.median(result.dolp) == pytest.approx(ideal, rel=1e-6)


def test_empirical_gain_pins_median_aoi_to_theta(synthetic_frame):
    """Corollary: if DoLP is pinned to the ideal at theta, the inferred AOI
    via Fresnel inversion should land at theta too."""
    frame, truth = synthetic_frame
    theta = truth["theta_i_mean_deg"]
    result = compute_slope_field(
        frame, resolution="full", method="bilinear", gain_mode="empirical",
        theta_i_mean_deg=theta, n_water=truth["n_water"],
        gain_reference_frame=frame,   # self-reference: formula check only
    )
    assert np.median(result.aoi_deg) == pytest.approx(theta, abs=0.05)


@pytest.mark.parametrize("method", ["bilinear", "kernel_averaging", "conv_demodulation"])
@pytest.mark.parametrize("gain", ["none", "lab", "empirical"])
def test_all_method_gain_combinations_complete(synthetic_frame, method, gain):
    frame, truth = synthetic_frame
    kw = dict(resolution="full", method=method, gain_mode=gain)
    if gain == "empirical":
        kw["theta_i_mean_deg"] = truth["theta_i_mean_deg"]
        kw["gain_reference_frame"] = frame   # reference required; self ok for smoke test
    result = compute_slope_field(frame, **kw)
    assert np.isfinite(result.mss)
    assert result.gain_g1 > 0


@pytest.mark.parametrize("gain", ["none", "lab", "empirical"])
def test_native_resolution_all_gains_complete(synthetic_frame, gain):
    """The default native (half-resolution) path runs for every gain mode and
    returns half-shape Stokes."""
    frame, truth = synthetic_frame
    kw = dict(gain_mode=gain)  # resolution defaults to "native"
    if gain == "empirical":
        kw["theta_i_mean_deg"] = truth["theta_i_mean_deg"]
        kw["gain_reference_frame"] = frame   # reference required; self ok for smoke test
    result = compute_slope_field(frame, **kw)
    H, W = truth["shape"]
    assert result.s1.shape == (H // 2, W // 2)
    assert np.isfinite(result.mss)
    assert result.gain_g1 > 0


def test_slope_fields_have_zero_mean(synthetic_frame):
    """Sx, Sy are de-meaned by the pipeline (matches MATLAB behavior)."""
    frame, truth = synthetic_frame
    result = compute_slope_field(
        frame, method="bilinear", gain_mode="none",
    )
    assert np.nanmean(result.Sx) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmean(result.Sy) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# NetCDF round-trip tests (skipped if example data is not present)
# ---------------------------------------------------------------------------

def test_netcdf_read_returns_metadata(example_nc_path):
    frame, meta = read_netcdf_frame(example_nc_path)
    assert frame.ndim == 2
    assert meta.method in ("bilinear", "kernel_averaging", "conv_demodulation")
    assert meta.gain_mode in ("none", "lab", "empirical")
    assert meta.theta_i_mean_deg is not None
    assert 1.0 < meta.n_water < 2.0
    assert "I0" in meta.layout and "I90" in meta.layout


def test_netcdf_carries_acquisition_and_wind_metadata(example_nc_path):
    """Frame rate, exposure, wind speed/direction, and timestamp should all
    be present in the example file."""
    frame, meta = read_netcdf_frame(example_nc_path)
    # Frame rate is surfaced as a first-class metadata field
    assert meta.framerate_hz == pytest.approx(30.0)
    # Exposure and wind live in raw_vars
    rv = meta.raw_vars
    assert float(rv["exposure_time"]["value"]) == pytest.approx(2000.0)
    assert float(rv["wind_speed_10m"]["value"]) == pytest.approx(10.0)
    assert float(rv["wind_from_direction"]["value"]) == pytest.approx(181.9)
    # Timestamp in the global attributes
    assert meta.raw_attrs.get("time_coverage_start") == "2019-10-31T16:00:00Z"


def test_netcdf_example_native_default(example_nc_path, median_nc_path):
    """The bundled single-frame example reduced with the DEFAULT settings:
    native (half-resolution) reduction, median-referenced empirical gain
    (the only supported empirical workflow). This is what a user gets out of
    the box for the E-PSS pipeline."""
    frame, meta = read_netcdf_frame(example_nc_path)
    median_frame, _ = read_netcdf_frame(median_nc_path)
    apply_layout_from_meta(meta)
    result = compute_slope_field(
        frame, gain_mode="empirical",  # resolution defaults to "native"
        theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )
    # Output is half-resolution: one Stokes vector per super-pixel.
    assert result.s1.shape == (frame.shape[0] // 2, frame.shape[1] // 2)
    # frame0001, native, median-referenced gain, theta_i=30, n=1.34:
    assert result.gain_g1 == pytest.approx(1.5587, abs=0.001)
    assert np.median(result.dolp) == pytest.approx(0.3309, abs=0.001)
    assert np.median(result.aoi_deg) == pytest.approx(26.099, abs=0.01)
    assert result.mss == pytest.approx(0.022729, abs=1e-5)


def test_netcdf_example_full_bilinear(example_nc_path, median_nc_path):
    """Full-resolution interpolated path with bilinear, median-referenced
    empirical gain."""
    frame, meta = read_netcdf_frame(example_nc_path)
    median_frame, _ = read_netcdf_frame(median_nc_path)
    apply_layout_from_meta(meta)
    result = compute_slope_field(
        frame, resolution="full", method="bilinear", gain_mode="empirical",
        theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )
    assert result.s1.shape == frame.shape
    # frame0001, full/bilinear, median-referenced gain, theta_i=30, n=1.34:
    assert result.gain_g1 == pytest.approx(1.5593, abs=0.001)
    assert np.median(result.dolp) == pytest.approx(0.3289, abs=0.001)
    assert np.median(result.aoi_deg) == pytest.approx(26.023, abs=0.01)
    # mss is the dimensionless slope variance var(Sx) + var(Sy):
    assert result.mss == pytest.approx(0.021375, abs=1e-5)


def test_netcdf_example_full_method4(example_nc_path, median_nc_path):
    """Full-resolution path with the exact Ratliff Method 4 interpolator
    (conv_demodulation), median-referenced empirical gain."""
    frame, meta = read_netcdf_frame(example_nc_path)
    median_frame, _ = read_netcdf_frame(median_nc_path)
    apply_layout_from_meta(meta)
    result = compute_slope_field(
        frame, resolution="full", method="conv_demodulation",
        gain_mode="empirical",
        theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )
    assert result.s1.shape == frame.shape
    # frame0001, full/Method 4, median-referenced gain, theta_i=30, n=1.34:
    assert result.gain_g1 == pytest.approx(1.5599, abs=0.001)
    assert np.median(result.dolp) == pytest.approx(0.3278, abs=0.001)
    assert np.median(result.aoi_deg) == pytest.approx(25.977, abs=0.01)
    assert result.mss == pytest.approx(0.021078, abs=1e-5)


def test_netcdf_carries_expected_global_attributes(example_nc_path):
    _, meta = read_netcdf_frame(example_nc_path)
    a = meta.raw_attrs
    assert a.get("Conventions", "").startswith("CF-")
    assert "ASIT" in a.get("platform", "")
    assert "Laxague" in a.get("creator_name", "")
    assert a.get("project") == "ASIT2019"


def test_netcdf_layout_matches_package_default(example_nc_path):
    """The example file uses the L0 layout, which is also the package default."""
    from pss.stokes import _OFFSETS
    _, meta = read_netcdf_frame(example_nc_path)
    assert meta.layout == _OFFSETS
    assert meta.layout_id == "L0"
