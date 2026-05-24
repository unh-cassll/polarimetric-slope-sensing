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
     reconstruct_eta_from_record(..., return_slopes=True).
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
    #   --nperseg N         Welch segment length in samples (default: 256)
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
from eta_field_recon.recon import reconstruct_eta_field  # noqa: E402

# np.trapz was removed in NumPy 2.x in favor of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Aperture fractions of the inscribed-circle ("full frame") diameter.
APERTURE_FRACTIONS = (1.0, 0.5, 0.25)


def omnidirectional_spectrum(eta, fs, nperseg=256, detrend="linear"):
    """Omnidirectional frequency spectrum S(f) of an elevation series.

    Welch PSD of the (mean-removed) elevation, returning frequency (Hz) and
    spectral density (m^2/Hz). Non-finite samples are dropped. nperseg is
    clipped to the series length, so short series degrade to a periodogram
    rather than erroring. The integral of S over f equals the series variance
    (verified: Parseval).

    Args:
        eta : 1-D elevation series (m).
        fs : sample rate (Hz).
        nperseg : Welch segment length (samples). Larger -> finer frequency
            resolution but fewer averages (noisier); smaller -> smoother but
            coarser. Clipped to len(eta).
        detrend : passed to scipy.signal.welch ('linear', 'constant', False).

    Returns:
        (f, S) : frequency (Hz) and PSD (m^2/Hz), both 1-D.
    """
    eta = np.asarray(eta, dtype=float)
    eta = eta[np.isfinite(eta)]
    if eta.size < 8:
        raise ValueError(f"elevation series too short for a spectrum: {eta.size} samples")
    nps = int(min(nperseg, eta.size))
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", choices=("full", "3s"), default="full",
                   help="which Zenodo stack to reduce (default: full 60 s)")
    p.add_argument("--downsample", type=int, default=8,
                   help="output-grid subsample factor (default: 8)")
    p.add_argument("--nperseg", type=int, default=256,
                   help="Welch segment length in samples (default: 256)")
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
    # 3+4. For each aperture, re-run only the CWT-based eta step and take the
    #      frame-center elevation series. We reuse the inscribed diameter from
    #      the first (full-frame) reconstruction's diag.
    # ------------------------------------------------------------------
    recon_kwargs = dict(
        dx=dx_m, fs=fs, downsample=1,   # already downsampled in the driver
        long_wave=base.long_wave_ran, verbose=False,
    )
    if base.ortho is not None:
        # water depth was already resolved by the driver into base.diag use;
        # reconstruct_eta_field will fall back to its default if unset.
        pass

    full_diam = None
    series = {}   # label -> (eta_center_series,)
    for frac in APERTURE_FRACTIONS:
        if frac == 1.0:
            # First pass: full frame (aperture=None), and learn the inscribed
            # diameter from the diag for the fractional discs.
            eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
                base.slope_x, base.slope_y, aperture_diameter_m=None, **recon_kwargs)
            full_diam = inscribed_diameter_m(diag)
            label = f"PSS full frame (D={full_diam:.2f} m)"
        else:
            diam = frac * full_diam
            eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
                base.slope_x, base.slope_y, aperture_diameter_m=diam, **recon_kwargs)
            label = f"PSS {frac:g}x (D={diam:.2f} m)"

        Ny, Nx = eta_xyt.shape[1], eta_xyt.shape[2]
        eta_center = eta_xyt[:, Ny // 2, Nx // 2]
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
        f, S = omnidirectional_spectrum(eta_c, fs, nperseg=args.nperseg)
        ax.loglog(f, S, color=color, lw=1.6, label=label)

    f_lid, S_lid = omnidirectional_spectrum(eta_lid, fs_lid, nperseg=args.nperseg)
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
