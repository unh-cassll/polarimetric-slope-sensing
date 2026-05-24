"""
eta_field_recon -- reconstruct water surface elevation eta(x, y, t) from
a time series of 2-D slope-image frames.

Sibling package to `pss` (polarimetric slope sensing). `pss` produces the
slope fields one frame at a time; this package consumes a stack of those
fields and reconstructs the elevation field eta(x, y, t) over the entire
imager footprint, combining:

  - a per-frame Harker-O'Leary spatial integration of slope -> wave SHAPE
    (zero-mean-per-frame eta_short(x, y, t))

  - a continuous-wavelet-based temporal inversion of the spatial-mean
    slope -> long-wave time series (eta_long(t)), which fills in the
    "integration constant" that the spatial integration loses

These two paths are orthogonal: short has zero spatial mean per frame,
long has no spatial structure inside the frame. eta(x, y, t) = eta_short
+ eta_long.

Quick start:

    from eta_field_recon import reconstruct_eta_field

    # slope_x_field, slope_y_field: (T, Ny, Nx) float arrays, units rad
    eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
        slope_x_field, slope_y_field,
        dx=0.006, fs=10.0,
        water_depth_m=15.0,
        downsample=8,
    )

See README.md for the method, HANDOFF.md for the derivation, gotchas, and
ideas for extensions.
"""

from .recon import reconstruct_eta_field
from .wavelet_core import lindisp_with_current
from .orthorectify import orthorectify_static, OrthoResult, OrthoPlan, build_ortho_plan
from .eta_pipeline import PipelineResult, reconstruct_eta_from_record

__all__ = [
    "reconstruct_eta_field",
    "reconstruct_eta_from_record",
    "PipelineResult",
    "orthorectify_static",
    "build_ortho_plan",
    "OrthoResult",
    "OrthoPlan",
    "lindisp_with_current",
]
