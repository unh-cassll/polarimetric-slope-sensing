"""
E-PSS example: reduce a single raw DoFP frame, but calibrate the empirical
DoLP gain against a *separate* per-pixel temporal-median background frame.

This is the canonical E-PSS empirical-gain workflow (Laxague et al.):

    g = DoLP_ideal(theta_i) / median( DoLP_obs(median_frame) )

The gain is derived once from the stable temporal-median frame and then
applied to each individual frame. This differs from the single-frame demo
(`load_and_reduce.py`), which derives the gain from the frame it is correcting
and therefore forces every frame's median DoLP onto the Fresnel ideal --
erasing the real frame-to-frame DoLP variability that carries the wave signal.

It exercises:
    1. read_netcdf_frame()        -- load the frame to reduce  (frame .nc)
    2. read_netcdf_frame()        -- load the median background (median .nc)
    3. apply_layout_from_meta()   -- honor the file's super-pixel layout
    4. compute_slope_field(..., gain_reference_frame=median)
    5. plotting                   -- 2x2 layout matching the MATLAB driver

Run from the repository root:

    python examples/load_and_reduce_with_median_gain.py \
        examples/asit_2019_raw_pol_frame0001.nc \
        --median examples/asit_2019_raw_pol_median.nc

    # save the figure instead of showing it
    python examples/load_and_reduce_with_median_gain.py \
        examples/asit_2019_raw_pol_frame0001.nc \
        --median examples/asit_2019_raw_pol_median.nc \
        --save out.png --no-show

Both input files use the NetCDF stack schema with dims (time, x, y); the
reader transparently selects a frame along `time` and reorients to (y, x).
Use --time-index to pick a frame other than the first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from either the repo root or from within examples/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from pss import (
    apply_layout_from_meta,
    compute_slope_field,
    read_netcdf_frame,
)

# Local import of the example-data resolver.
import _data  # noqa: E402

# Downsample factor for display only; pipeline always runs on full resolution.
DISPLAY_STRIDE = 4


def _print_metadata_summary(meta, label: str) -> None:
    """One-screen summary of a NetCDF file's metadata."""
    attrs = meta.raw_attrs
    print("=" * 70)
    print(f"NetCDF metadata summary  [{label}]")
    print("=" * 70)
    print(f"  title           : {attrs.get('title', 'n/a')}")
    print(f"  id              : {attrs.get('id', 'n/a')}")
    print(f"  project         : {attrs.get('project', 'n/a')}")
    print(f"  instrument      : {attrs.get('instrument', 'n/a')}")
    print(f"  time            : {attrs.get('time_coverage_start', 'n/a')}")
    print(f"  theta_i_mean    : {meta.theta_i_mean_deg} deg")
    print(f"  n_water         : {meta.n_water}")
    print(f"  framerate       : {meta.framerate_hz} Hz")
    print(f"  layout_id       : {meta.layout_id}")
    print(f"  layout          : {meta.layout}")
    print()


def _plot_result(result, frame_shape, gain_source: str) -> plt.Figure:
    """2x2 panel mirroring the MATLAB driver: s1, s2, cross-look, along-look."""
    s = DISPLAY_STRIDE
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(
        f"E-PSS reduction  |  {frame_shape[0]}x{frame_shape[1]} px  |  "
        f"gain={result.gain_mode} (g={result.gain_g1:.4f}, from {gain_source})\n"
        f"median DoLP = {np.nanmedian(result.dolp):.4f}, "
        f"median AOI = {np.nanmedian(result.aoi_deg):.3f} deg, "
        f"mss = {result.mss:.4f} deg^2",
        fontsize=11,
    )
    panels = [
        (result.s1[::s, ::s],     "s1 (gain-corrected)",    None),
        (result.s2[::s, ::s],     "s2 (gain-corrected)",    None),
        (result.Ax_deg[::s, ::s], "cross-look slope [deg]", "RdBu_r"),
        (result.Ay_deg[::s, ::s], "along-look slope [deg]", "RdBu_r"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        if cmap == "RdBu_r":
            v = np.nanpercentile(np.abs(data), 99.5)
            im = ax.imshow(data, cmap=cmap, vmin=-v, vmax=v,
                           aspect="equal", interpolation="nearest")
        else:
            im = ax.imshow(data, cmap="gray",
                           aspect="equal", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("frame", type=Path, nargs="?", default=None,
                   help="raw frame to reduce (default: bundled "
                        "asit_2019_raw_pol_frame0001.nc)")
    p.add_argument("--median", type=Path, default=None,
                   help="temporal-median background frame for the empirical "
                        "gain (default: bundled asit_2019_raw_pol_median.nc)")
    p.add_argument("--time-index", type=int, default=0,
                   help="index along the time dimension to reduce (default 0)")
    p.add_argument("--method",
                   choices=["bilinear", "kernel_averaging", "conv_demodulation"],
                   default=None,
                   help="override the file's recommended Stokes method")
    p.add_argument("--theta", type=float, default=None,
                   help="override theta_i (degrees)")
    p.add_argument("--save", type=Path, default=None,
                   help="if given, write the figure here as PNG")
    p.add_argument("--no-show", action="store_true",
                   help="skip interactive plt.show()")
    args = p.parse_args(argv)

    frame_path = args.frame if args.frame is not None else _data.frame_path()
    median_path = args.median if args.median is not None else _data.median_path()

    # 1) Load the frame to reduce.
    print(f"Reading frame   : {frame_path}  (time_index={args.time_index})")
    frame, meta = read_netcdf_frame(frame_path, time_index=args.time_index)
    _print_metadata_summary(meta, "frame")

    # 2) Load the temporal-median background frame for the empirical gain.
    print(f"Reading median  : {median_path}")
    median_frame, med_meta = read_netcdf_frame(median_path)
    _print_metadata_summary(med_meta, "median (gain reference)")

    # Sanity: the two files should share geometry and super-pixel layout.
    if meta.layout and med_meta.layout and meta.layout != med_meta.layout:
        print("WARNING: frame and median super-pixel layouts differ; "
              "the empirical gain may be miscalibrated.\n")

    # 3) Apply the file's super-pixel layout so the package matches.
    apply_layout_from_meta(meta)

    # 4) Decide reduction parameters (CLI overrides win over metadata).
    method  = args.method or meta.method
    theta_i = args.theta if args.theta is not None else meta.theta_i_mean_deg
    if theta_i is None:
        print("error: empirical gain requires theta_i; pass --theta")
        return 2

    print(f"Reducing with method={method}, gain=empirical (from median), "
          f"theta_i={theta_i} deg, n_water={meta.n_water}...")
    result = compute_slope_field(
        frame,
        method=method,
        gain_mode="empirical",
        theta_i_mean_deg=theta_i,
        n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )

    print("\nResult:")
    print(f"  gain (g1, g2) : ({result.gain_g1:.4f}, {result.gain_g2:.4f})")
    print(f"  notes         : {result.gain_notes}")
    print(f"  median DoLP   : {np.nanmedian(result.dolp):.4f}  "
          f"(this frame, post-gain; NOT forced to the Fresnel ideal)")
    print(f"  median AOI    : {np.nanmedian(result.aoi_deg):.3f} deg")
    print(f"  mss           : {result.mss:.4f} deg^2")

    # 5) Plot.
    fig = _plot_result(result, frame.shape, gain_source="median frame")
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"\nsaved figure -> {args.save}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
