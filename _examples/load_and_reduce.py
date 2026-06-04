"""
End-to-end example: load a raw DoFP frame from NetCDF, reduce it with the
pss / E-PSS pipeline, and plot the result.

Canonical minimal example for the pss package, exercising:
    1. read_netcdf_frame()        -- load raw frame + metadata
    2. apply_layout_from_meta()   -- honor the file's super-pixel layout
    3. compute_slope_field()      -- full reduction with empirical DoLP gain
    4. plotting                   -- 2x2 layout matching the MATLAB driver

By default it reduces the bundled single-frame example
(`asit_2019_raw_pol_frame0001.nc`). If you also pass `--median`, the empirical DoLP
gain is calibrated against that temporal-median reference frame -- the E-PSS
workflow (see also load_and_reduce_with_median_gain.py, which is dedicated to
that path). Without `--median`, no empirical gain is applied: the gain is
never self-referenced from a single frame (that assumes a flat instantaneous
surface), so it falls back to no gain.

Run from the repository root:

    python examples/load_and_reduce.py
    python examples/load_and_reduce.py examples/asit_2019_raw_pol_frame0001.nc
    python examples/load_and_reduce.py --median examples/asit_2019_raw_pol_median.nc
    python examples/load_and_reduce.py --save out.png --no-show

Example data ships with the repository (examples/*.nc) and will eventually be
mirrored in a Zenodo archive; see examples/_data.py.
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

# Local import of the example-data resolver (works whether run in-place or
# via the installed console script, thanks to the sys.path insert above).
import _data  # noqa: E402

# Downsample factor for display only; pipeline always runs on full resolution.
DISPLAY_STRIDE = 4


def _print_metadata_summary(meta) -> None:
    """One-screen summary of the NetCDF metadata."""
    attrs = meta.raw_attrs
    print("=" * 70)
    print("NetCDF metadata summary")
    print("=" * 70)
    print(f"  title           : {attrs.get('title', 'n/a')}")
    print(f"  id              : {attrs.get('id', 'n/a')}")
    print(f"  project         : {attrs.get('project', 'n/a')}")
    print(f"  platform        : {attrs.get('platform', 'n/a')}")
    print(f"  instrument      : {attrs.get('instrument', 'n/a')}")
    print(f"  creator         : {attrs.get('creator_name', 'n/a')}")
    print(f"  time            : {attrs.get('time_coverage_start', 'n/a')}")
    print(f"  position        : "
          f"({attrs.get('geospatial_lat_min', 'n/a')}, "
          f"{attrs.get('geospatial_lon_min', 'n/a')})")
    print()
    print("  Pipeline settings (from file):")
    print(f"    method        : {meta.method}")
    print(f"    gain_mode     : {meta.gain_mode}")
    print(f"    theta_i_mean  : {meta.theta_i_mean_deg} deg")
    print(f"    n_water       : {meta.n_water}")
    print(f"    framerate     : {meta.framerate_hz} Hz")
    print(f"    layout_id     : {meta.layout_id}")
    print(f"    layout        : {meta.layout}")
    print()


def _plot_result(result, frame_shape, title_suffix=""):
    """2x2 panel mirroring the MATLAB driver: s1, s2, cross-look, along-look."""
    s = DISPLAY_STRIDE
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(
        f"pss / E-PSS reduction  |  {frame_shape[0]}x{frame_shape[1]} px  |  "
        f"gain={result.gain_mode} (g={result.gain_g1:.4f}) {title_suffix}\n"
        f"median DoLP = {np.nanmedian(result.dolp):.4f}, "
        f"median AOI = {np.nanmedian(result.aoi_deg):.3f} deg, "
        f"mss = {result.mss:.4f} deg^2",
        fontsize=11,
    )
    panels = [
        (result.s1[::s, ::s],     "s1 (gain-corrected)",     None),
        (result.s2[::s, ::s],     "s2 (gain-corrected)",     None),
        (result.Ax_deg[::s, ::s], "cross-look slope [deg]",  "RdBu_r"),
        (result.Ay_deg[::s, ::s], "along-look slope [deg]",  "RdBu_r"),
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
    p.add_argument("netcdf", type=Path, nargs="?", default=None,
                   help="raw frame NetCDF (default: the bundled "
                        "asit_2019_raw_pol_frame0001.nc)")
    p.add_argument("--median", type=Path, default=None,
                   help="optional temporal-median frame; if given, the "
                        "empirical gain is calibrated against it")
    p.add_argument("--time-index", type=int, default=0,
                   help="index along the time dimension to reduce (default 0)")
    p.add_argument("--method",
                   choices=["bilinear", "kernel_averaging", "conv_demodulation"],
                   default=None,
                   help="override the file's recommended Stokes method")
    p.add_argument("--gain",
                   choices=["none", "lab", "empirical"], default=None,
                   help="override the file's recommended gain mode")
    p.add_argument("--theta", type=float, default=None,
                   help="override theta_i (degrees)")
    p.add_argument("--save", type=Path, default=None,
                   help="if given, write the figure here as PNG")
    p.add_argument("--no-show", action="store_true",
                   help="skip interactive plt.show()")
    args = p.parse_args(argv)

    # 1) Resolve and load the frame + metadata.
    nc_path = args.netcdf if args.netcdf is not None else _data.frame_path()
    print(f"Reading {nc_path}  (time_index={args.time_index}) ...")
    frame, meta = read_netcdf_frame(nc_path, time_index=args.time_index)
    _print_metadata_summary(meta)

    # 2) Optional median reference frame for the empirical gain.
    median_frame = None
    if args.median is not None:
        print(f"Reading median (gain reference) {args.median} ...")
        median_frame, _ = read_netcdf_frame(args.median)

    # 3) Apply the file's super-pixel layout so the package matches.
    apply_layout_from_meta(meta)

    # 4) Decide reduction parameters (CLI overrides win over metadata).
    method  = args.method or meta.method
    gain    = args.gain   or meta.gain_mode
    theta_i = args.theta if args.theta is not None else meta.theta_i_mean_deg

    if gain == "empirical" and theta_i is None:
        print("error: gain_mode='empirical' requires theta_i; pass --theta")
        return 2

    src = "median frame" if median_frame is not None else "self"
    print(f"Reducing with method={method}, gain={gain} (ref={src}), "
          f"theta_i={theta_i}...")
    result = compute_slope_field(
        frame,
        method=method,
        gain_mode=gain,
        theta_i_mean_deg=theta_i,
        n_water=meta.n_water,
        gain_reference_frame=median_frame,
    )

    print("\nResult:")
    print(f"  gain (g1, g2) : ({result.gain_g1:.4f}, {result.gain_g2:.4f})")
    print(f"  notes         : {result.gain_notes}")
    print(f"  median DoLP   : {np.nanmedian(result.dolp):.4f}")
    print(f"  median AOI    : {np.nanmedian(result.aoi_deg):.3f} deg")
    print(f"  mss           : {result.mss:.4f} deg^2")

    # 5) Plot.
    fig = _plot_result(result, frame.shape, title_suffix=f"[ref={src}]")
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"\nsaved figure -> {args.save}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
