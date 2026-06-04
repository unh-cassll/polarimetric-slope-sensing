#!/usr/bin/env python3
"""
precompute_mean_wave.py -- one-off generator for the committed mean-slope file.

WHAT THIS DOES
--------------
Reads the full 60-second ASIT-2019 raw polarimetric stack, reduces every frame
to a wave-slope field with `pss`, orthorectifies each frame onto a uniform
ground grid (static-platform geometry from the file), and takes the SPATIAL
MEAN of the orthorectified slopes per frame. The result is two short 1-D time
series:

    sx_mean(t), sy_mean(t)    -- spatial-mean slope, post-orthorectification

plus the pre-orthorectification means (sx_mean_raw, sy_mean_raw) for
comparison. These are written to a small, well-documented NetCDF file
(a few KB) that IS committed to the repository:

    examples/asit2019_mean_slope_60s.nc

`examples/_data.py: mean_wave_timeseries()` then loads that file and runs the
long-wave inversion (CWT -> Krogstad -> dispersion -> inverse CWT) live, so the
mean-wave elevation eta_long(t) can be demonstrated on the real 60 s record
WITHOUT anyone needing to download the 10 GB stack.

WHY A ONE-OFF SCRIPT
--------------------
The per-frame reduction + ortho over 1800 frames takes tens of minutes and
needs the 10 GB stack. It is run ONCE, by a maintainer who has the stack, and
its tiny output is committed. End users never run it. Hence: a standalone
tools/ script (not a console entry point, not part of the test path).

USAGE
-----
    python tools/precompute_mean_wave.py --input /path/to/asit_2019_raw_pol_stack.nc

    # optional:
    #   --output examples/asit2019_mean_slope_60s.nc   (default shown)
    #   --median /path/to/asit_2019_raw_pol_median.nc   (empirical-gain ref)
    #   --max-frames N                                  (debug: stop early)

The output file's md5 will differ from run to run only if the inputs or code
change; commit it once and leave it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pss import read_netcdf_frame, apply_layout_from_meta, compute_slope_field
from pss.io import _orient_raw_frame
from eta_field_recon.orthorectify import orthorectify_static, build_ortho_plan


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _n_frames(path: Path) -> int:
    with Dataset(str(path)) as ds:
        return len(ds.dimensions["time"]) if "time" in ds.dimensions else 1


def _raw_scalar(meta, name):
    v = meta.raw_vars.get(name)
    return None if v is None else float(np.asarray(v["value"]).flatten()[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="path to the full 60 s raw polarimetric stack (.nc)")
    ap.add_argument("--output", type=Path,
                    default=Path("_data/asit2019_mean_slope_60s.nc"),
                    help="output NetCDF path (committed artifact)")
    ap.add_argument("--median", type=Path, default=None,
                    help="temporal-median frame for empirical gain (optional)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="debug: process only the first N frames")
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"input stack not found: {args.input}")

    n_total = _n_frames(args.input)
    n = n_total if args.max_frames is None else min(args.max_frames, n_total)
    print(f"input    : {args.input}  ({n_total} frames, processing {n})")

    # Frame 0 -> metadata + geometry (constant across the static record).
    frame0, meta = read_netcdf_frame(args.input, time_index=0)
    apply_layout_from_meta(meta)

    fs = meta.framerate_hz
    theta_i = meta.theta_i_mean_deg
    n_water = meta.n_water
    freeboard = _raw_scalar(meta, "freeboard")
    focal_mm = _raw_scalar(meta, "lens_focal_length")
    pitch_um = _raw_scalar(meta, "pixel_pitch")
    azimuth = _raw_scalar(meta, "camera_azimuth")
    water_depth = _raw_scalar(meta, "water_depth")
    for nm, val in [("framerate", fs), ("theta_i_mean", theta_i),
                    ("freeboard", freeboard), ("lens_focal_length", focal_mm),
                    ("pixel_pitch", pitch_um), ("water_depth", water_depth)]:
        if val is None:
            raise SystemExit(f"input is missing required metadata: {nm}")

    median_frame = None
    gain_mode = "none"
    if args.median is not None:
        median_frame, _ = read_netcdf_frame(args.median, time_index=0)
        gain_mode = "empirical"
    print(f"geometry : fs={fs} Hz, theta_i={theta_i} deg, freeboard={freeboard} m, "
          f"focal={focal_mm} mm, pitch={pitch_um} um, depth={water_depth} m")
    print(f"gain     : {gain_mode}"
          + (" (median-referenced)" if median_frame is not None else ""))

    # native-resolution slope field samples one Stokes vector per 2x2 superpixel
    pitch_field_m = (pitch_um / 1e6) * 2.0

    sx_mean = np.empty(n)
    sy_mean = np.empty(n)
    sx_mean_raw = np.empty(n)
    sy_mean_raw = np.empty(n)

    def reduce_one(frame):
        return compute_slope_field(
            frame, resolution="native", method="conv_demodulation",
            gain_mode=gain_mode, theta_i_mean_deg=theta_i, n_water=n_water,
            gain_reference_frame=median_frame)

    # Open the stack ONCE and stream frames from the open handle, slicing the
    # time axis lazily (rf[ti] reads only that frame's bytes). This avoids
    # reopening the (multi-GB) file 1800 times and keeps peak RAM at one frame.
    import time as _time
    _t0 = _time.time()
    with Dataset(str(args.input), "r") as ds:
        rf = ds.variables["raw_frame"]
        rf_dims = tuple(d.lower() for d in rf.dimensions)
        if "time" not in rf_dims:
            raise SystemExit("input has no 'time' dimension; not a stack.")
        t_ax = rf_dims.index("time")
        remaining_dims = rf_dims[:t_ax] + rf_dims[t_ax + 1:]

        def read_frame(ti):
            sl = [slice(None)] * len(rf_dims)
            sl[t_ax] = ti
            arr = np.asarray(rf[tuple(sl)], dtype=np.float64)
            return _orient_raw_frame(arr, remaining_dims, time_index=0)

        # Per-phase timers to localize stalls (typically a blocking read on a
        # high-latency or synced filesystem).
        t_read = t_reduce = t_ortho = 0.0
        _plan = None

        for ti in range(n):
            _a = _time.time()
            frame = read_frame(ti)
            _b = _time.time()
            res = reduce_one(frame)
            _c = _time.time()

            # pre-ortho spatial mean
            sx_mean_raw[ti] = np.nanmean(res.Sx)
            sy_mean_raw[ti] = np.nanmean(res.Sy)

            # orthorectify this frame, then spatial-mean the uniform-grid slopes.
            # Build the (expensive) triangulation plan ONCE on the first frame
            # and reuse it for all subsequent frames -- the geometry is static.
            if _plan is None:
                _plan = build_ortho_plan(
                    res.Sx.shape[0], res.Sx.shape[1],
                    freeboard_m=freeboard, theta_i_mean_deg=theta_i,
                    focal_length_m=focal_mm / 1000.0,
                    pixel_pitch_m=pitch_field_m,
                    camera_azimuth_deg=azimuth, verbose=True)
            o = orthorectify_static(
                res.Sx, res.Sy, freeboard_m=freeboard, theta_i_mean_deg=theta_i,
                focal_length_m=focal_mm / 1000.0, pixel_pitch_m=pitch_field_m,
                camera_azimuth_deg=azimuth, plan=_plan, verbose=False)
            v = o.valid
            sx_mean[ti] = np.nanmean(o.slope_x[v])
            sy_mean[ti] = np.nanmean(o.slope_y[v])
            _d = _time.time()

            t_read += _b - _a
            t_reduce += _c - _b
            t_ortho += _d - _c

            # Live progress: every 10 frames (and first/last), with rate, ETA,
            # and the read/reduce/ortho time split so the bottleneck is visible.
            done = ti + 1
            if done == 1 or done % 10 == 0 or done == n:
                el = _time.time() - _t0
                rate = done / el if el > 0 else 0.0
                eta = (n - done) / rate if rate > 0 else float("nan")
                print(f"  frame {done}/{n}  "
                      f"({rate:.2f} frame/s, {el:.0f}s elapsed, ~{eta:.0f}s left)  "
                      f"| read {t_read:.1f}s  reduce {t_reduce:.1f}s  "
                      f"ortho {t_ortho:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Write the documented NetCDF artifact.
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(args.output), "w", format="NETCDF4") as ds:
        ds.title = "ASIT-2019 spatial-mean wave slope time series (60 s)"
        ds.summary = (
            "Spatial mean of the orthorectified wave-slope field, per frame, "
            "for the full 60 s ASIT-2019 record. Produced once by "
            "tools/precompute_mean_wave.py; consumed by "
            "examples/_data.py:mean_wave_timeseries() to reconstruct the "
            "mean-wave elevation eta_long(t) live without the 10 GB stack.")
        ds.source_file = args.input.name
        ds.source_md5 = _md5_of(args.input)
        ds.source_zenodo_doi = "10.5281/zenodo.20361229"
        ds.processing_gain_mode = gain_mode
        ds.processing_resolution = "native"
        ds.processing_stokes_method = "conv_demodulation"
        ds.orthorectified = "yes (static geometry); sx_mean/sy_mean are post-ortho"
        ds.created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
        ds.n_frames = n

        ds.createDimension("time", n)
        for name, data, units, desc in [
            ("sx_mean", sx_mean, "radians",
             "spatial-mean cross-look slope (orthorectified)"),
            ("sy_mean", sy_mean, "radians",
             "spatial-mean along-look slope (orthorectified)"),
            ("sx_mean_raw", sx_mean_raw, "radians",
             "spatial-mean cross-look slope (pre-orthorectification)"),
            ("sy_mean_raw", sy_mean_raw, "radians",
             "spatial-mean along-look slope (pre-orthorectification)"),
        ]:
            var = ds.createVariable(name, "f8", ("time",))
            var[:] = data
            var.units = units
            var.description = desc

        for name, val, units in [
            ("fs", fs, "Hz"), ("theta_i_mean", theta_i, "degrees"),
            ("freeboard", freeboard, "m"), ("water_depth", water_depth, "m"),
            ("lens_focal_length", focal_mm, "mm"),
            ("pixel_pitch", pitch_um, "micrometers"),
            ("camera_azimuth", azimuth if azimuth is not None else np.nan, "degrees"),
        ]:
            sv = ds.createVariable(name, "f8")
            sv[...] = val
            sv.units = units

    print(f"\nwrote {args.output}  (md5 {_md5_of(args.output)})")
    print("Commit this file. mean_wave_timeseries() will load it.")


if __name__ == "__main__":
    main()
