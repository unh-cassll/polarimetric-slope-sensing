"""
NetCDF I/O for raw DoFP frames.

This module reads frames written to the pss/E-PSS NetCDF schema (CF-1.10 +
ACDD-1.3) and returns the raw array together with the metadata needed to
drive `compute_slope_field`. Example files ship with the package and longer
time series are distributed via a Zenodo archive (see examples/_data.py).

A typical workflow:

    from pss import read_netcdf_frame, compute_slope_field
    frame, meta = read_netcdf_frame("asit_example.nc")
    result = compute_slope_field(
        frame,
        method=meta["method"],
        gain_mode=meta["gain_mode"],
        theta_i_mean_deg=meta["theta_i_mean_deg"],
        n_water=meta["n_water"],
    )

The reader is intentionally permissive about which attributes are present so
it works on files that have been edited or extended by users; missing fields
default to package-level defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameMetadata:
    """Metadata extracted from a NetCDF DoFP frame file."""
    method: str = "bilinear"
    gain_mode: str = "empirical"
    lab_gain: tuple[float, float] = (1.2185, 1.2197)
    theta_i_mean_deg: float | None = None
    n_water: float = 1.34
    framerate_hz: float | None = None
    layout: dict[str, tuple[int, int]] = field(default_factory=dict)
    layout_id: str = "L0"
    # passthrough — every attribute we found, for downstream inspection
    raw_attrs: dict[str, Any] = field(default_factory=dict)
    raw_vars: dict[str, Any] = field(default_factory=dict)


def _layout_from_grid(grid: np.ndarray, dims: tuple[str, ...] | None = None) -> dict[str, tuple[int, int]]:
    """Convert a 2x2 grid of polarizer-angle integers to the dict format
    used by `pss.stokes._OFFSETS`.

    The package convention is grid[row, col] -> angle. If the on-disk variable
    stores the grid with dimensions ordered (super_col, super_row) — as the
    NetCDF stack writer does — the grid is transposed relative to the package
    convention, so we transpose it back using the declared dimension names.
    """
    grid = np.asarray(grid)
    if dims is not None and len(dims) == 2:
        # Put the row-like dimension first. Anything containing "row" is the
        # row axis; if it's currently axis 1, transpose.
        d0, d1 = dims[0].lower(), dims[1].lower()
        if "row" in d1 and "row" not in d0:
            grid = grid.T
    mapping: dict[str, tuple[int, int]] = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            angle = int(grid[r, c])
            if angle == -1:
                continue
            key = f"I{angle}"
            mapping[key] = (r, c)
    return mapping


def _orient_raw_frame(
    raw: np.ndarray,
    dims: tuple[str, ...],
    time_index: int = 0,
) -> np.ndarray:
    """Reduce an on-disk raw_frame array to a 2-D (y, x) frame.

    Handles two schemas transparently using the declared dimension names:
      * old 2-D export with dims (y, x)              -> returned unchanged
      * new stack export with dims (time, x, y)      -> select time_index,
                                                        then transpose to (y, x)
    Any leading length-1 'time' axis is selected by `time_index`; the
    remaining two axes are ordered so that the 'y' dimension is first.
    """
    dims = tuple(d.lower() for d in dims)
    arr = np.asarray(raw, dtype=np.float64)

    # Select along a time axis if present.
    if "time" in dims:
        t_ax = dims.index("time")
        n_t = arr.shape[t_ax]
        if not (-n_t <= time_index < n_t):
            raise IndexError(
                f"time_index {time_index} out of range for time dimension "
                f"of length {n_t}"
            )
        arr = np.take(arr, time_index, axis=t_ax)
        dims = dims[:t_ax] + dims[t_ax + 1:]

    if arr.ndim != 2:
        # Fall back to a plain squeeze for any other singleton axes.
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"raw_frame has unsupported shape {raw.shape} with dims "
                f"{dims!r}; expected a 2-D (y, x) frame after time selection"
            )
        return arr

    # Order the two spatial axes so y is first.
    if len(dims) == 2 and dims[0] == "x" and dims[1] == "y":
        arr = arr.T
    return arr


def read_netcdf_frame(
    path: str | Path, *, time_index: int = 0
) -> tuple[np.ndarray, FrameMetadata]:
    """Read a raw DoFP frame and its metadata from a NetCDF-4 file.

    Handles both the legacy 2-D ``(y, x)`` export and the newer stack export
    with dims ``(time, x, y)`` (the time axis is selected by ``time_index``
    and the spatial axes are reordered to ``(y, x)``).

    Parameters
    ----------
    path : str or Path
        NetCDF-4 file to read.
    time_index : int
        Which frame to select along the ``time`` dimension, if present.
        Defaults to 0 (the first/only frame). Ignored for 2-D files.

    Returns
    -------
    frame : ndarray, 2-D, float64
        Raw intensities in ``(y, x)`` order, promoted from on-disk dtype.
    meta : FrameMetadata
        Parsed metadata. Pass straight into `compute_slope_field`:

            r = compute_slope_field(frame, method=meta.method,
                                    gain_mode=meta.gain_mode,
                                    theta_i_mean_deg=meta.theta_i_mean_deg,
                                    n_water=meta.n_water)
    """
    try:
        import netCDF4 as nc
    except ImportError as e:
        raise ImportError(
            "reading NetCDF requires the netCDF4 package "
            "(`pip install netCDF4`)"
        ) from e

    path = Path(path)
    with nc.Dataset(str(path), "r") as ds:
        if "raw_frame" not in ds.variables:
            raise KeyError(
                f"{path}: missing required variable 'raw_frame'; "
                f"file does not appear to be a pss/E-PSS frame export"
            )
        rf = ds.variables["raw_frame"]
        raw = _orient_raw_frame(
            np.asarray(rf[:], dtype=np.float64),
            tuple(rf.dimensions),
            time_index=time_index,
        )

        meta = FrameMetadata()
        # Scalar geometry vars
        if "theta_i_mean" in ds.variables:
            v = ds.variables["theta_i_mean"]
            val = float(np.asarray(v[...]))
            status = getattr(v, "status", None)
            meta.theta_i_mean_deg = None if status == "unknown" else val
        if "n_water" in ds.variables:
            meta.n_water = float(np.asarray(ds.variables["n_water"][...]))
        if "framerate" in ds.variables:
            v = ds.variables["framerate"]
            val = float(np.asarray(v[...]))
            status = getattr(v, "status", None)
            meta.framerate_hz = None if status == "unknown" else val

        # Super-pixel layout
        if "superpixel_layout" in ds.variables:
            lv = ds.variables["superpixel_layout"]
            grid = np.asarray(lv[...])
            meta.layout = _layout_from_grid(grid, tuple(lv.dimensions))
        # Recommended pipeline settings
        meta.method = getattr(ds, "pss_processing_method", "bilinear")
        meta.gain_mode = getattr(ds, "pss_processing_gain_mode", "empirical")
        if hasattr(ds, "pss_processing_lab_gain"):
            lg = np.asarray(ds.pss_processing_lab_gain).flatten()
            if lg.size >= 2:
                meta.lab_gain = (float(lg[0]), float(lg[1]))
        meta.layout_id = getattr(ds, "pss_processing_layout_id", "L0")

        # Passthrough of remaining attributes for inspection
        meta.raw_attrs = {k: getattr(ds, k) for k in ds.ncattrs()}
        meta.raw_vars = {
            k: dict(value=np.asarray(v[...]),
                    attrs={a: v.getncattr(a) for a in v.ncattrs()})
            for k, v in ds.variables.items()
            if k != "raw_frame"  # huge; skip
        }

    return raw, meta


def apply_layout_from_meta(meta: FrameMetadata) -> None:
    """Mutate `pss.stokes._OFFSETS` in place to match a frame's layout.

    Use this when working with a frame whose layout differs from the package
    default. Wraps the patch in a context-manager-style helper if you need
    safe rollback:

        from pss.io import apply_layout_from_meta
        from pss.stokes import _OFFSETS as default_layout
        saved = dict(default_layout)
        apply_layout_from_meta(meta)
        try:
            ... # run pipeline
        finally:
            from pss import stokes
            stokes._OFFSETS = saved
    """
    if not meta.layout:
        return
    from . import stokes as _stokes
    _stokes._OFFSETS = dict(meta.layout)
