"""
Single-camera sky-aware inversion demo (mainline E-PSS option).

Runs the full chain on bundled ASIT raw DoFP frames:

    raw DoFP stack --(seapol sky-aware inversion + empirical slope anchor)-->
    slope fields --(ortho + eta)--> eta(x, y, t),  Hm0 vs lidar

The sky-aware path (inversion="skyaware") replaces the Fresnel DoLP->AoI
reduction with seapol's environmental forward-model inversion (true facet
slopes, look-independent anisotropy), then rescales the slope amplitude with a
single in-scene slope gain `f` so the long-wave (mean-tilt) amplitude matches
the empirical DoLP-gain pipeline -- recovering quantitative long-wave Hs while
preserving the short-wave / slope-PDF tail shape. The amplitude target is set by
gain_mode: "empirical" (in-scene), "lab" (a lab DoLP gain), or "none" (native).

Requires the optional `seapol` package:  pip install 'epss[skyaware]'.

Run from the repository root:

    python _examples/skyaware_demo.py                 # bundled 3 s stack (quick)
    python _examples/skyaware_demo.py --full          # full record + Hm0 vs lidar
    python _examples/skyaware_demo.py --gain-mode none
    python _examples/skyaware_demo.py --time 2019-10-31T16:05:00
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epss import run_epss                                   # noqa: E402
from pss.skyaware import require_seapol                     # noqa: E402
import _data                                                # noqa: E402

# ASIT (WHOI Air-Sea Interaction Tower) site coordinates.
ASIT_LAT, ASIT_LON = 41.325, -70.566
# A representative ASIT 2019 acquisition time (UTC) when the file carries none.
DEFAULT_TIME = dt.datetime(2019, 10, 31, 16, 5, 0)
F_BAND = (0.05, 1.0)


def _read_stack(path):
    """Load (T,H,W) raw DoFP frames + acquisition geometry from a bundled nc."""
    ds = nc.Dataset(str(path))
    raw = np.asarray(ds["raw_frame"][:], dtype=float)
    if raw.ndim == 2:
        raw = raw[None]
    geom = dict(
        theta_i_mean_deg=float(ds["theta_i_mean"][:]),
        n_water=float(ds["n_water"][:]),
        focal_length_m=float(ds["lens_focal_length"][:]),
        pixel_pitch_m=float(ds["pixel_pitch"][:]),
        camera_azimuth_deg=float(ds["camera_azimuth"][:]),
        freeboard_m=float(ds["freeboard"][:]),
        fs=float(ds["framerate"][:]),
    )
    ds.close()
    return raw, geom


def _hm0(eta_series, fs):
    from scipy.signal import welch
    f, S = welch(eta_series - np.nanmean(eta_series), fs=fs, nperseg=min(1024, eta_series.size))
    m = (f >= F_BAND[0]) & (f <= F_BAND[1])
    return 4.0 * np.sqrt(np.trapezoid(S[m], f[m])) if m.any() else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--full", action="store_true",
                    help="use the full record and reconstruct eta + Hm0 vs lidar")
    ap.add_argument("--gain-mode", default="empirical",
                    choices=["none", "lab", "empirical"],
                    help="slope-anchor amplitude target (default empirical)")
    ap.add_argument("--time", default=None,
                    help="acquisition UTC time, ISO 8601 (for the sun geometry)")
    args = ap.parse_args(argv)

    try:
        require_seapol()
    except ImportError as e:
        print(e)
        return 2

    path = _data.stack_full_path() if args.full else _data.stack_3s_path()
    frames, geom = _read_stack(path)
    t_utc = (dt.datetime.fromisoformat(args.time) if args.time else DEFAULT_TIME)
    print(f"sky-aware demo: {frames.shape[0]} frames from {Path(path).name}")
    print(f"  geometry: theta_i={geom['theta_i_mean_deg']:.1f} deg, "
          f"fs={geom['fs']:.1f} Hz, camera_az={geom['camera_azimuth_deg']:.1f} deg")

    common = dict(
        inversion="skyaware", gain_mode=args.gain_mode,
        acquisition_time_utc=t_utc, site_lat=ASIT_LAT, site_lon=ASIT_LON,
        camera_azimuth_compass_deg=geom["camera_azimuth_deg"],
        theta_v_deg=geom["theta_i_mean_deg"], water_case=2,
        n_water=geom["n_water"], verbose=True,
    )
    if args.full:
        res = run_epss(frames, fs=geom["fs"],
                       theta_i_mean_deg=geom["theta_i_mean_deg"],
                       freeboard_m=geom["freeboard_m"],
                       pixel_pitch_m=geom["pixel_pitch_m"],
                       focal_length_m=geom["focal_length_m"],
                       camera_azimuth_deg=geom["camera_azimuth_deg"],
                       water_depth_m=15.0, **common)
    else:
        # slope-only (a 3 s record is too short for the long-wave inversion).
        res = run_epss(frames, theta_i_mean_deg=geom["theta_i_mean_deg"], **common)

    env, anchor = res.skyaware_env, res.anchor
    print(f"  env  : sky_depolarization={env.sky_depolarization:.2f} upwelling_scale={env.upwelling_scale:.2f} "
          f"gain={env.gain:.3f} inferred={env.inferred}")
    print(f"  anchor: f={anchor.f:.3f} ({anchor.mode})")
    print(f"  slopes: {res.slope_x.shape}  mss=var(Sx)+var(Sy)="
          f"{np.nanvar(res.slope_x) + np.nanvar(res.slope_y):.4f}")

    if args.full and res.eta_xyt is not None:
        ny, nx = res.eta_xyt.shape[:2]
        cam = _hm0(res.eta_xyt[ny // 2, nx // 2], geom["fs"])
        t_l, elev = _data.lidar_elevation()
        fs_l = 1.0 / np.nanmedian(np.diff(t_l))
        lid = _hm0(np.asarray(elev), fs_l)
        print(f"  Hm0  : camera={cam:.2f} m  lidar={lid:.2f} m  "
              f"({100 * cam / lid:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
