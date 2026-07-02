#!/usr/bin/env python3
"""Developer one-shot: Piermont 2025 dual-camera mean frames -> epss NetCDF.

Reads the v7.3 (HDF5) mean-frame structs and the run log, and writes the minimal
artifacts the wide-FOV / dual-camera demo consumes, in the schema `pss.io`
already reads (vars raw_frame / theta_i_mean / n_water / framerate /
superpixel_layout; attrs pss_processing_*).

The afternoon session (runs 14-22 on 2025-09-29) ran two cameras at once:
  * LDEO 5 mm WIDE  @ ~43 deg  -> spans ~11-75 deg incidence across its FOV;
    this single frame is the DoLP-AOI calibration input.
  * UNH 75 mm NARROW scanned 23-51 deg -> the high-resolution imager whose DoLP
    we invert through the wide calibration.

Minimal-artifact policy
-----------------------
calibrate_widefov uses only the CENTRAL COLUMN (per-row incidence) and a narrow
central strip (the DoLP profile), and the narrow imager is reduced to one
(theta, DoLP) point per run. So we crop to a central column band, symmetric about
the optical center -- which preserves the central column and the per-row AOI
mapping exactly -- and store uint16 raw counts. That turns ~20 MB/frame into
well under 1 MB while remaining a faithful raw DoFP frame.

This is NOT part of the installed package; it requires h5py / pandas / openpyxl
and access to the MATLAB data directory. Run it once to (re)generate the
committed `_data/piermont2025_*.nc` files.

Usage:
    python _examples/convert_piermont_dualcam.py \
        --datadir /path/to/MATLAB/polarimetry/Piermont2025 \
        --outdir  _data
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

import h5py
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Site (Piermont, NY) and sensor constants.
SITE_LAT = 41.04313537091353
SITE_LON = -73.89601867276131
PIX_PITCH_M = 3.45e-6
N_WATER = 1.34
UTC_OFFSET_HOURS = -4           # EDT for the Sep/Oct 2025 field days
# epss/IMX250MZR super-pixel layout: grid[row, col] -> polarizer angle (deg).
LAYOUT_GRID = np.array([[90, 45], [135, 0]], dtype=np.int32)

CAMERAS = {
    "LDEO": dict(mat="Piermont2025_LDEO_mean_frames_struct.mat",
                 group="mean_frames_struct", shape=(2464, 2056),
                 fl_col="FL LDEO [mm]", inc_col="ROT_Y LDEO [from nadir]"),
    "UNH": dict(mat="Piermont2025_UNH_mean_frames_struct.mat",
                group="UNH_mean_frames_all_runs", shape=(2448, 2048),
                fl_col="FL UNH [mm]", inc_col="ROT_Y UNH  [from nadir]"),
}

# The afternoon dual-camera session.
WIDE_RUN = 16                                   # LDEO 5 mm @ 43 deg (14:40)
NARROW_RUNS = [14, 15, 16, 17, 18, 20, 21, 22]  # UNH 75 mm, 23-51 deg


def _read_frame(f, ds, run):
    """Dereference and orient the mean frame for a 1-based run number."""
    ref = ds[run - 1, 0]
    arr = np.array(f[ref]).T.astype(np.float64)   # h5py stores transposed
    return arr


def _central_band(frame, band_cols):
    """Symmetric central column band (even-aligned, to keep 2x2 tiling)."""
    W = frame.shape[1]
    half = (band_cols // 2) & ~1
    c0 = (W // 2 - half) & ~1
    return frame[:, c0:c0 + 2 * half]


def _to_uint16(frame):
    """Clip negatives, round to uint16 raw counts (read promotes to float64)."""
    return np.clip(np.rint(frame), 0, 65535).astype(np.uint16)


def _local_time(df, run):
    tl = pd.to_datetime(df[df["Run #"] == run]["Time Local"].values[0])
    return dt.datetime(tl.year, tl.month, tl.day, tl.hour, tl.minute, tl.second)


def _write_common_meta(ncf, *, focal_length_m, theta_i_mean_deg, camera,
                       acq_local, comment):
    n = ncf.createVariable("n_water", "f8"); n[...] = N_WATER
    fl = ncf.createVariable("lens_focal_length", "f8")
    fl[...] = focal_length_m; fl.units = "m"
    pp = ncf.createVariable("pixel_pitch", "f8")
    pp[...] = PIX_PITCH_M; pp.units = "m"
    ti = ncf.createVariable("theta_i_mean", "f8")
    ti[...] = theta_i_mean_deg; ti.units = "degree"
    ncf.createDimension("super_row", 2); ncf.createDimension("super_col", 2)
    lay = ncf.createVariable("superpixel_layout", "i4",
                             ("super_row", "super_col"))
    lay[...] = LAYOUT_GRID
    ncf.pss_processing_method = "bilinear"
    ncf.pss_processing_gain_mode = "none"
    ncf.pss_processing_layout_id = "L0"
    ncf.camera = camera
    ncf.site_lat = SITE_LAT
    ncf.site_lon = SITE_LON
    ncf.acquisition_time_local = acq_local.isoformat()
    ncf.acquisition_time_utc = (
        acq_local - dt.timedelta(hours=UTC_OFFSET_HOURS)).isoformat()
    ncf.utc_offset_hours = UTC_OFFSET_HOURS
    ncf.row_sign = -1.0     # Piermont mounts: incidence increases down the rows
    ncf.title = comment
    ncf.Conventions = "CF-1.10, ACDD-1.3"


def write_wide(datadir, outdir, band_cols):
    cam = CAMERAS["LDEO"]
    df = pd.read_excel(os.path.join(datadir, "Piermont2025_LDEO_UNH_DataLog.xlsx"),
                       sheet_name="Sheet1")
    inc = pd.to_numeric(df[cam["inc_col"]], errors="coerce").to_numpy()
    with h5py.File(os.path.join(datadir, cam["mat"]), "r") as f:
        ds = f[cam["group"]]["mean_raw_frame"]
        frame = _read_frame(f, ds, WIDE_RUN)
    frame = _to_uint16(_central_band(frame, band_cols))
    H, W = frame.shape
    out = os.path.join(outdir, "piermont2025_ldeo_wide_5mm_mean.nc")
    with Dataset(out, "w") as nc:
        nc.createDimension("y", H); nc.createDimension("x", W)
        rf = nc.createVariable("raw_frame", "u2", ("y", "x"), zlib=True)
        rf[...] = frame
        rf.long_name = "mean DoFP raw counts (central column band)"
        fr = nc.createVariable("framerate", "f8"); fr[...] = 0.0
        fr.status = "unknown"   # a time-averaged mean frame has no framerate
        _write_common_meta(
            nc, focal_length_m=0.005, theta_i_mean_deg=float(inc[WIDE_RUN - 1]),
            camera="LDEO 5mm wide", acq_local=_local_time(df, WIDE_RUN),
            comment=("Piermont 2025 LDEO 5 mm wide mean frame (run "
                     f"{WIDE_RUN}); central {W}-col band; DoLP-AOI calibration "
                     "input for the dual-camera demo."))
    print(f"wrote {out}  ({H}x{W} uint16, "
          f"{os.path.getsize(out)/1e6:.2f} MB)")


def write_narrow(datadir, outdir, band_cols):
    cam = CAMERAS["UNH"]
    df = pd.read_excel(os.path.join(datadir, "Piermont2025_LDEO_UNH_DataLog.xlsx"),
                       sheet_name="Sheet1")
    inc = pd.to_numeric(df[cam["inc_col"]], errors="coerce").to_numpy()
    frames, thetas = [], []
    with h5py.File(os.path.join(datadir, cam["mat"]), "r") as f:
        ds = f[cam["group"]]["mean_raw_frame"]
        for r in NARROW_RUNS:
            if np.isnan(inc[r - 1]) or f[ds[r - 1, 0]].shape != cam["shape"]:
                print(f"  skipping run {r} (missing/invalid)")
                continue
            frames.append(_to_uint16(_central_band(_read_frame(f, ds, r),
                                                    band_cols)))
            thetas.append(float(inc[r - 1]))
    stack = np.stack(frames)                       # (time, y, x)
    T, H, W = stack.shape
    out = os.path.join(outdir, "piermont2025_unh_narrow_75mm_stack.nc")
    with Dataset(out, "w") as nc:
        nc.createDimension("time", T)
        nc.createDimension("x", W); nc.createDimension("y", H)
        # stack schema: raw_frame dims (time, x, y) -> store frame.T per time
        rf = nc.createVariable("raw_frame", "u2", ("time", "x", "y"), zlib=True)
        rf[...] = np.transpose(stack, (0, 2, 1))
        rf.long_name = "mean DoFP raw counts per run (central column band)"
        tpf = nc.createVariable("theta_i_per_frame", "f8", ("time",))
        tpf[...] = np.array(thetas); tpf.units = "degree"
        tpf.long_name = "per-run incidence from nadir (the narrow scan)"
        fr = nc.createVariable("framerate", "f8"); fr[...] = 0.0
        fr.status = "unknown"
        _write_common_meta(
            nc, focal_length_m=0.075, theta_i_mean_deg=float(np.mean(thetas)),
            camera="UNH 75mm narrow", acq_local=_local_time(df, NARROW_RUNS[0]),
            comment=("Piermont 2025 UNH 75 mm narrow mean frames (runs "
                     f"{NARROW_RUNS}); central {W}-col band; multi-angle scan "
                     "for the dual-camera recovery demo."))
    print(f"wrote {out}  ({T}x{H}x{W} uint16, "
          f"{os.path.getsize(out)/1e6:.2f} MB; theta {min(thetas):.0f}-"
          f"{max(thetas):.0f} deg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", required=True,
                    help="directory holding the Piermont 2025 v7.3 .mat "
                         "mean-frame structs (developer one-shot; the "
                         "converted artifacts are already committed)")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_data"))
    ap.add_argument("--band-cols", type=int, default=128,
                    help="central column band width to keep (even)")
    args = ap.parse_args()
    write_wide(args.datadir, args.outdir, args.band_cols)
    write_narrow(args.datadir, args.outdir, args.band_cols)


if __name__ == "__main__":
    main()
