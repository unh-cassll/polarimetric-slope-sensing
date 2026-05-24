#!/usr/bin/env python3
"""
spectra_vs_aperture.py -- omnidirectional elevation spectra vs averaging aperture.

End-to-end demonstration of how the circular slope-averaging aperture affects
the recovered long-wave (mean-wave) elevation spectrum, compared against the
independent lidar ground truth.

Pipeline
--------
  1. Retrieve the full 60 s DoFP stack from Zenodo (10.1 GB; cached on first
     use, md5-verified -- see examples/_data.py). A flag allows the 505 MB
     3 s stack instead for plumbing tests, but note a 3 s record cannot
     resolve the low-frequency wave band and the long-wave spectrum will be
     meaningless there.
  2. Reduce every frame to a slope field with `pss` and orthorectify the
     stack onto a uniform ground grid -- ONCE -- via
     reconstruct_eta_from_record(..., return_slopes=True). Each frame is
     subsampled (reduce_downsample, default 4) as float32 BEFORE stacking, so
     the peak memory stays near ~2 GB rather than the tens of GB a full native
     stack would need (which OOMs a 32 GB machine during orthorectification).
  3. For three centered circular apertures -- full frame, 0.5x, and 0.25x of
     the inscribed-circle diameter (min(frame width, height)) -- re-run only
     the (cheap) CWT-based long-wave inversion via reconstruct_eta_field with
     the corresponding aperture_diameter_m, yielding three eta_long(t) series.
  4. Reconstruct the full elevation eta(x, y, t) = eta_short + eta_long and
     take the frame-center elevation series for each aperture (eta_short is
     aperture-independent; only the long-wave mean differs).
  5. Compute the omnidirectional frequency spectrum S(f) (Welch PSD, m^2/Hz)
     for each of the three series and for the full 10 min lidar elevation.
  6. Plot all four spectra on one log-log axis with labeled axes and legend.

"Full frame" aperture
---------------------
The 100% diameter is the inscribed circle -- min(frame width, height) in
meters on the orthorectified grid -- so the disc fits entirely inside the
frame and the 0.5x / 0.25x discs nest within it, all using real data only
(no edge clipping / no-data fill inside the aperture).

Lidar comparison
----------------
The lidar is a 10 min, 10 Hz record; the PSS stack is ~60 s. We use the FULL
lidar series for the best-resolved ground-truth spectrum (more record =
tighter low-frequency bins), so the lidar curve is better resolved at low f
than the PSS curves by construction. The two instruments view spatially
offset points and start together; this script compares SPECTRA (no time
alignment needed), unlike validate_eta_long_vs_lidar.py which measures the lag.

Run
---
    uv run python tools/spectra_vs_aperture.py
    # options:
    #   --stack {full,3s}   which Zenodo stack to reduce (default: full)
    #   --downsample N      output-grid subsample factor (default: 8)
    #   --seg-seconds S     Welch segment length in seconds (default: 30)
    #   --out PATH          figure path (default: spectra_vs_aperture.png)
    #   --no-download       fail rather than download from Zenodo

Requires the lidar artifact (committed): examples/asit2019_lidar_elevation_10min.nc
and network access to Zenodo for the stack (unless already cached).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

# make examples/_data and the packages importable regardless of invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples import _data  # noqa: E402
from eta_field_recon.eta_pipeline import reconstruct_eta_from_record  # noqa: E402
from eta_field_recon.recon import (  # noqa: E402
    reconstruct_eta_field,
    _aperture_spatial_mean,
    _circular_aperture_mask,
    _make_temporal_window,
)
from eta_field_recon.wavelet_core import (  # noqa: E402
    _cwt, _inverse_cwt, lindisp_with_current, krogstad_eta_coeffs,
)

# np.trapz was removed in NumPy 2.x in favor of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Aperture fractions of the inscribed-circle ("full frame") diameter.
APERTURE_FRACTIONS = (1.0, 0.5, 0.25)


def omnidirectional_spectrum(eta, fs, seg_seconds=30.0, detrend="linear"):
    """Omnidirectional frequency spectrum S(f) of an elevation series.

    Welch PSD of the (mean-removed) elevation, returning frequency (Hz) and
    spectral density (m^2/Hz). Non-finite samples are dropped. The integral of
    S over f equals the series variance (verified: Parseval).

    The Welch segment length is specified as a DURATION (seg_seconds) rather
    than a sample count, and converted to samples per series via fs. Keying to
    duration gives the same low-frequency resolution (1 / seg_seconds) across
    instruments with different sample rates -- e.g. the 30 Hz PSS series and
    the 10 Hz lidar both resolve down to ~1/30 s = 0.033 Hz at the default --
    which makes their spectra directly comparable at low frequency. Longer
    segments resolve lower frequencies but yield fewer averages (noisier).

    If the record is shorter than one segment, nperseg is clipped to the
    series length (the estimate degrades to a single-segment periodogram) and
    a warning is printed.

    Args:
        eta : 1-D elevation series (m).
        fs : sample rate (Hz).
        seg_seconds : Welch segment length in SECONDS (default 30). The sample
            count is round(seg_seconds * fs). Lowest resolved frequency is
            ~1 / seg_seconds.
        detrend : passed to scipy.signal.welch ('linear', 'constant', False).

    Returns:
        (f, S) : frequency (Hz) and PSD (m^2/Hz), both 1-D.
    """
    eta = np.asarray(eta, dtype=float)
    eta = eta[np.isfinite(eta)]
    if eta.size < 8:
        raise ValueError(f"elevation series too short for a spectrum: {eta.size} samples")
    nps_target = int(round(seg_seconds * fs))
    nps = int(min(nps_target, eta.size))
    if nps < nps_target:
        print(f"  WARNING: record ({eta.size/fs:.1f} s) shorter than the "
              f"requested {seg_seconds:.0f} s Welch segment; using a single "
              f"{nps}-sample segment (periodogram). Low-frequency resolution "
              f"is limited to ~{fs/nps:.3f} Hz.")
    f, S = welch(eta - eta.mean(), fs=fs, nperseg=nps, detrend=detrend)
    return f, S


def inscribed_diameter_m(diag) -> float:
    """Inscribed-circle diameter (m) of the orthorectified output grid.

    The 'full frame' aperture: the largest centered disc that fits inside the
    rectangular frame, i.e. min(frame width, frame height) in meters on the
    downsampled grid that reconstruct_eta_field actually operates on.
    """
    dx_ds = float(diag["dx_ds"])
    Ny, Nx = diag["aperture_mask"].shape
    return dx_ds * min(Ny, Nx)


def _eta_long_for_aperture(sx_ds, sy_ds, fs, aperture_mask, freqs_cwt,
                           water_depth_m, mother=None):
    """Long-wave elevation series eta_long(t) for one averaging aperture.

    This is exactly the eta_long block of reconstruct_eta_field, factored out
    so the (expensive, aperture-INDEPENDENT) per-frame g2s short-wave
    integration is not repeated for every aperture. Verified bit-identical to
    reconstruct_eta_field's eta_long for the same inputs.

    Args:
        sx_ds, sy_ds : (T, Ny, Nx) downsampled slope stack (the same grid
            reconstruct_eta_field operates on; here downsample=1 because the
            driver already downsampled).
        fs : frame rate (Hz).
        aperture_mask : (Ny, Nx) boolean circular aperture.
        freqs_cwt : CWT frequency grid (Hz).
        water_depth_m : depth for the dispersion relation.

    Returns:
        eta_long : (T,) spatial-mean elevation series (m).
    """
    T = sx_ds.shape[0]
    temporal_W = _make_temporal_window(T, "tukey", 0.25)
    sx_mean = _aperture_spatial_mean(sx_ds, aperture_mask)
    sy_mean = _aperture_spatial_mean(sy_ds, aperture_mask)
    sx_mean = sx_mean - sx_mean.mean()
    sy_mean = sy_mean - sy_mean.mean()
    Wsx = _cwt(sx_mean * temporal_W, freqs_cwt, fs, mother).values
    Wsy = _cwt(sy_mean * temporal_W, freqs_cwt, fs, mother).values
    _, k_disp = lindisp_with_current(2 * np.pi * freqs_cwt, water_depth_m, 0.0)
    W_eta, _, _ = krogstad_eta_coeffs(Wsx, Wsy, k_disp)
    return _inverse_cwt(W_eta, freqs_cwt, fs, mother)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", choices=("full", "3s"), default="full",
                   help="which Zenodo stack to reduce (default: full 60 s)")
    p.add_argument("--downsample", type=int, default=8,
                   help="output-grid subsample factor (default: 8)")
    p.add_argument("--reduce-downsample", type=int, default=4,
                   help="subsample each frame at reduce time, before stacking "
                        "(default: 4). This is the MEMORY lever: the full "
                        "native slope stack for the 60 s record is ~18-36 GB, "
                        "which OOMs a 32 GB machine during orthorectification. "
                        "4 keeps the peak near ~2 GB. Use 2 for finer spatial "
                        "resolution if you have >=32 GB free, 1 for full native.")
    p.add_argument("--seg-seconds", type=float, default=30.0,
                   help="Welch segment length in SECONDS (default: 30). "
                        "Converted to samples per series via each instrument's "
                        "sample rate, so PSS (30 Hz) and lidar (10 Hz) get the "
                        "same low-frequency resolution (~1/seg_seconds). Longer "
                        "= lower frequencies resolved but fewer averages.")
    p.add_argument("--out", default="spectra_vs_aperture.png",
                   help="output figure path")
    p.add_argument("--no-download", action="store_true",
                   help="fail rather than download the stack from Zenodo")
    args = p.parse_args(argv)

    allow_download = not args.no_download

    # ------------------------------------------------------------------
    # 1. Resolve the stack (download/cache from Zenodo on first use).
    # ------------------------------------------------------------------
    if args.stack == "full":
        stack_path = _data.stack_full_path(allow_download=allow_download)
    else:
        stack_path = _data.stack_3s_path(allow_download=allow_download)
        print("WARNING: the 3 s stack cannot resolve the low-frequency wave "
              "band; the long-wave spectrum will be unreliable. Use --stack "
              "full for a meaningful result.")
    print(f"stack: {stack_path}")

    # The empirical-gain reference (temporal median) is a separate small file.
    try:
        median_path = _data.median_path(allow_download=allow_download)
    except Exception as e:  # noqa: BLE001 -- proceed without gain if unavailable
        median_path = None
        print(f"(no median gain reference available: {e}; proceeding without "
              f"empirical gain)")

    # ------------------------------------------------------------------
    # 2. Reduce + orthorectify ONCE. return_slopes=True hands back the
    #    orthorectified slope stack so we can re-aperture cheaply below.
    #    The aperture here is irrelevant (full frame) -- we only want the
    #    slope stack and fs/dx; the eta it computes is discarded.
    # ------------------------------------------------------------------
    print("reducing + orthorectifying the stack (this is the expensive step) ...")
    base = reconstruct_eta_from_record(
        stack_path,
        orthorectify=True,
        gain_reference_path=median_path,
        downsample=args.downsample,
        reduce_downsample=args.reduce_downsample,
        return_slopes=True,
        verbose=True,
    )
    if base.slope_x is None:
        raise RuntimeError("return_slopes did not populate the slope stack")
    fs = base.fs_hz
    dx_m = base.slope_dx_m
    print(f"reduced stack: {base.slope_x.shape} @ fs={fs:.3f} Hz, "
          f"ground dx={dx_m*1000:.2f} mm; long_wave_ran={base.long_wave_ran}")

    if not base.long_wave_ran:
        print("WARNING: the record was too short to run the long-wave "
              "inversion; eta_long is zero and the spectra below will reflect "
              "only the short-wave shape. (Use the full 60 s stack.)")

    # ------------------------------------------------------------------
    # 3+4. eta_short (the per-frame g2s surface integration) is INDEPENDENT of
    #      the averaging aperture -- only the long-wave mean eta_long(t)
    #      depends on it. So we run the full reconstruct_eta_field ONCE (for
    #      eta_short and the full-frame eta_long), then for the other apertures
    #      recompute only the cheap 1-D long-wave inversion and add the SAME
    #      center eta_short. This is bit-identical to three full passes but
    #      skips two expensive g2s loops.
    # ------------------------------------------------------------------
    print("reconstructing eta (full pass: eta_short once + full-frame eta_long) ...")
    eta_xyt, eta_long_full, eta_short, conf, diag = reconstruct_eta_field(
        base.slope_x, base.slope_y, dx=dx_m, fs=fs, downsample=1,
        aperture_diameter_m=None, long_wave=base.long_wave_ran, verbose=True)

    Ny, Nx = eta_xyt.shape[1], eta_xyt.shape[2]
    cy, cx = Ny // 2, Nx // 2
    eta_short_center = eta_short[:, cy, cx]          # aperture-independent
    full_diam = inscribed_diameter_m(diag)

    # The downsampled slope grid and freqs the long-wave step operates on. The
    # driver already downsampled (downsample=1 here), so dx_ds == dx_m and the
    # grid is base.slope_x itself.
    sx_ds, sy_ds = base.slope_x, base.slope_y
    dx_ds = diag["dx_ds"]
    freqs_cwt = np.linspace(0.05, 2.0, 80)           # reconstruct_eta_field default
    water_depth_m = 100.0                            # recon default (deep water)

    series = {}   # label -> eta_center series
    for frac in APERTURE_FRACTIONS:
        if frac == 1.0:
            eta_long = eta_long_full
            label = f"PSS full frame (D={full_diam:.2f} m)"
        elif base.long_wave_ran:
            diam = frac * full_diam
            mask = _circular_aperture_mask(sx_ds.shape[1], sx_ds.shape[2],
                                           dx_ds, diam)
            eta_long = _eta_long_for_aperture(
                sx_ds, sy_ds, fs, mask, freqs_cwt, water_depth_m)
            label = f"PSS {frac:g}x (D={frac*full_diam:.2f} m)"
        else:
            eta_long = np.zeros(sx_ds.shape[0])
            label = f"PSS {frac:g}x (D={frac*full_diam:.2f} m, no long-wave)"

        eta_center = eta_short_center + eta_long
        series[label] = eta_center
        print(f"  aperture {frac:>4g}x: eta_center std = "
              f"{np.std(eta_center)*100:.2f} cm")

    # ------------------------------------------------------------------
    # 5. Lidar spectrum (full 10 min record).
    # ------------------------------------------------------------------
    t_lid, eta_lid = _data.lidar_elevation()
    fs_lid = 1.0 / np.median(np.diff(t_lid))
    print(f"lidar: {eta_lid.size} samples @ fs={fs_lid:.3f} Hz "
          f"({(t_lid[-1]-t_lid[0]):.1f} s)")

    # ------------------------------------------------------------------
    # 6. Compute spectra and plot.
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    pss_colors = ["#1f77b4", "#2ca02c", "#d62728"]   # full, 0.5x, 0.25x
    for (label, eta_c), color in zip(series.items(), pss_colors):
        f, S = omnidirectional_spectrum(eta_c, fs, seg_seconds=args.seg_seconds)
        ax.loglog(f, S, color=color, lw=1.6, label=label)

    f_lid, S_lid = omnidirectional_spectrum(eta_lid, fs_lid,
                                            seg_seconds=args.seg_seconds)
    ax.loglog(f_lid, S_lid, color="k", lw=2.0, ls="--",
              label=f"lidar (10 min, {fs_lid:.0f} Hz)")

    ax.set_xlabel("frequency  $f$  (Hz)")
    ax.set_ylabel(r"elevation spectral density  $S(f)$  (m$^2$/Hz)")
    ax.set_title("Omnidirectional elevation spectrum vs averaging aperture")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    # Focus on the gravity-wave band; trim the empty decades.
    ax.set_xlim(0.03, max(fs, fs_lid) / 2.0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
