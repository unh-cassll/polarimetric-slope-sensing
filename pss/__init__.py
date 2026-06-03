"""
pss — Polarimetric Slope Sensing, with E-PSS empirical-gain support.

Python port of the demo MATLAB driver `sample_slope_field_calculations.m`
from https://github.com/unh-cassll/polarimetric-slope-sensing, extended to
expose three DoLP-gain modes per Laxague et al. (2026, IEEE TGRS):

    "none"       - no gain (raw polarimeter measurement)
    "lab"        - fixed lab-calibrated gain (replaces the original
                   hard-coded 1.2185 / 1.2197 from the MATLAB)
    "empirical"  - frame-by-frame gain that scales the median DoLP to match
                   the Fresnel ideal at the camera's known angle of incidence

Quick start:
    from pss import compute_slope_field
    result = compute_slope_field(frame, gain_mode="empirical",
                                 theta_i_mean_deg=30.0)
"""

from .gain import DEFAULT_LAB_GAIN, GainResult, apply_gain
from .fresnel import (
    build_lookup_table,
    dolp_to_aoi,
    fresnel_dolp,
    load_lookup_table,
)
from .io import FrameMetadata, apply_layout_from_meta, read_netcdf_frame
from .slope import SlopeResult, compute_slope_field
from .stokes import (
    METHODS,
    by_bilinear_interpolation,
    by_superpixel,
    by_conv_demodulation,
    by_kernel_averaging,
    compute_stokes,
)

__all__ = [
    "apply_gain", "GainResult", "DEFAULT_LAB_GAIN",
    "build_lookup_table", "dolp_to_aoi", "fresnel_dolp", "load_lookup_table",
    "FrameMetadata", "read_netcdf_frame", "apply_layout_from_meta",
    "compute_slope_field", "SlopeResult",
    "compute_stokes", "METHODS",
    "by_bilinear_interpolation", "by_kernel_averaging", "by_conv_demodulation",
    "by_superpixel",
]
