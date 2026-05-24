#!/usr/bin/env python3
"""
validate_eta_long_vs_lidar.py -- cross-validate PSS eta_long against the lidar.

Compares the PSS long-wave reconstruction eta_long(t) (recovered live from the
committed spatial-mean slope artifact via _data.mean_wave_timeseries) against
the independent Riegl LD90-3 water-surface elevation series
(_data.lidar_elevation).

The two instruments START together but view spatially offset points, so the
same wave field reaches them at different times: a propagation LAG between the
series is expected and physical. This script:

  1. loads both series (they are at different sample rates and durations:
     PSS ~30 Hz over ~60 s; lidar 10 Hz over 600 s),
  2. band-limits and resamples both to a common rate,
  3. cross-correlates over the overlapping (shorter, PSS) window to MEASURE the
     lag and the peak correlation,
  4. reports amplitude agreement (std) on the matched window,
  5. writes a figure: both time series (lag-aligned) + the cross-correlation.

Run:
    uv run python tools/validate_eta_long_vs_lidar.py
    # options:
    #   --out  validation_eta_long_vs_lidar.png   (figure path)
    #   --max-lag  30   (s; search window for the lag, default 30)
    #   --band  0.05 0.5  (Hz; band-limit both series before correlating)

Requires both committed artifacts:
    examples/asit2019_mean_slope_60s.nc       (PSS spatial-mean slope)
    examples/asit2019_lidar_elevation_10min.nc (Riegl lidar elevation)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, correlate, correlation_lags

# make examples/_data importable regardless of invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples import _data  # noqa: E402


def _resample_uniform(t, y, fs_target):
    """Linear-resample (t, y) onto a uniform grid at fs_target, from t[0]."""
    t0, t1 = t[0], t[-1]
    n = int(np.floor((t1 - t0) * fs_target)) + 1
    tg = t0 + np.arange(n) / fs_target
    yg = np.interp(tg, t, y)
    return tg, yg


def _bandpass(y, fs, f_lo, f_hi):
    """Zero-phase band-pass; falls back to high/low-pass at the Nyquist edges."""
    nyq = fs / 2.0
    lo, hi = f_lo / nyq, min(f_hi / nyq, 0.99)
    if lo <= 0:
        b, a = butter(4, hi, btype="low")
    else:
        b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, y)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("validation_eta_long_vs_lidar.png"))
    ap.add_argument("--max-lag", type=float, default=30.0,
                    help="max |lag| to search, seconds (default 30)")
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 0.5),
                    metavar=("F_LO", "F_HI"),
                    help="band-pass (Hz) applied before correlating")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load both series.
    # ------------------------------------------------------------------
    t_eta, eta = _data.mean_wave_timeseries(verbose=False)
    t_lid, lid = _data.lidar_elevation()
    fs_eta = 1.0 / (t_eta[1] - t_eta[0])
    fs_lid = 1.0 / (t_lid[1] - t_lid[0])
    print(f"eta_long : {eta.size} samples @ {fs_eta:.3g} Hz "
          f"({t_eta[-1]:.1f} s), std={eta.std()*100:.2f} cm")
    print(f"lidar    : {lid.size} samples @ {fs_lid:.3g} Hz "
          f"({t_lid[-1]:.1f} s), std={lid.std()*100:.2f} cm")

    # ------------------------------------------------------------------
    # Common rate (the lower of the two), band-limit, resample.
    # ------------------------------------------------------------------
    fs = min(fs_eta, fs_lid)
    f_lo, f_hi = args.band
    f_hi = min(f_hi, 0.45 * fs)
    _, eta_r = _resample_uniform(t_eta, eta, fs)
    t_lid_r, lid_r = _resample_uniform(t_lid, lid, fs)
    eta_bp = _bandpass(eta_r, fs, f_lo, f_hi)
    lid_bp = _bandpass(lid_r, fs, f_lo, f_hi)

    # ------------------------------------------------------------------
    # Cross-correlate over the overlapping window (length of the shorter,
    # PSS, record). Slide the PSS window through the lidar within +/-max_lag.
    # ------------------------------------------------------------------
    n_eta = eta_bp.size
    # lidar segment long enough to allow the lag search on both sides
    pad = int(args.max_lag * fs)
    seg = lid_bp[:n_eta + pad] if lid_bp.size >= n_eta + pad else lid_bp
    # normalized cross-correlation
    a = (eta_bp - eta_bp.mean()) / (eta_bp.std() + 1e-30)
    b = (seg - seg.mean()) / (seg.std() + 1e-30)
    xc = correlate(b, a, mode="full") / min(a.size, b.size)
    lags = correlation_lags(b.size, a.size, mode="full") / fs
    keep = np.abs(lags) <= args.max_lag
    lags_k, xc_k = lags[keep], xc[keep]
    pk = int(np.argmax(xc_k))
    lag_peak = lags_k[pk]
    r_peak = xc_k[pk]
    print(f"\ncross-correlation (band {f_lo}-{f_hi:.2f} Hz):")
    print(f"  peak lag        = {lag_peak:+.2f} s "
          f"(lidar leads if negative)")
    print(f"  broadband corr  = {r_peak:.3f}  (over full slid window)")

    # ------------------------------------------------------------------
    # Align both series by the measured lag and keep ONLY the genuine overlap
    # window -- the span where both the PSS and the (shifted) lidar series
    # actually have data. The correlation and std are recomputed over just
    # this overlap, which is the honest comparison: outside it, one series is
    # absent and only dilutes the broadband number.
    # ------------------------------------------------------------------
    # eta_bp lives on te = [0, (n_eta-1)/fs]. The lidar sample j sits at PSS
    # time (j/fs - lag_peak) after alignment. Build a common index range.
    te = np.arange(n_eta) / fs
    tl_full = np.arange(lid_bp.size) / fs - lag_peak
    t_start = max(te[0], tl_full[0])
    t_stop = min(te[-1], tl_full[-1])
    # sample the overlap on the eta grid
    mask = (te >= t_start) & (te <= t_stop)
    t_ov = te[mask]
    eta_ov = eta_bp[mask]
    lid_ov = np.interp(t_ov, tl_full, lid_bp)   # lidar resampled onto eta grid

    # correlation + amplitude over the overlap only
    if eta_ov.size > 4:
        r_ov = float(np.corrcoef(eta_ov, lid_ov)[0, 1])
    else:
        r_ov = float("nan")
    std_e, std_l = eta_ov.std(), lid_ov.std()
    print(f"  overlap window  = {t_start:.1f}-{t_stop:.1f} s "
          f"({t_ov[-1]-t_ov[0]:.1f} s, where both series exist)")
    print(f"  overlap corr    = {r_ov:.3f}")
    print(f"  band-passed std : eta_long {std_e*100:.2f} cm vs "
          f"lidar {std_l*100:.2f} cm (ratio {std_e/std_l:.2f})")

    # ------------------------------------------------------------------
    # Figure: overlap time series (lag-aligned) + cross-correlation.
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    ax1.plot(t_ov, eta_ov, lw=1.5, color="#3b2f8f", label="PSS $\\eta_{long}$")
    ax1.plot(t_ov, lid_ov, lw=1.2, alpha=0.85, color="#d2691e",
             label=f"Riegl lidar (aligned, shifted {-lag_peak:+.2f} s)")
    ax1.set_xlim(t_ov[0], t_ov[-1])
    ax1.set_xlabel("time [s]  (PSS clock)")
    ax1.set_ylabel("elevation [m]")
    ax1.set_title(f"PSS $\\eta_{{long}}$ vs Riegl lidar, overlap window "
                  f"(band {f_lo}-{f_hi:.2f} Hz, r = {r_ov:.2f})")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(lags_k, xc_k, lw=1.3, color="#3b2f8f")
    ax2.axvline(lag_peak, color="k", ls="--", lw=1,
                label=f"peak lag {lag_peak:+.2f} s, r={r_peak:.2f}")
    ax2.set_xlabel("lag [s]  (PSS relative to lidar)")
    ax2.set_ylabel("normalized cross-correlation")
    ax2.set_title("Cross-correlation (lag finder)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote figure -> {args.out}")


if __name__ == "__main__":
    main()
