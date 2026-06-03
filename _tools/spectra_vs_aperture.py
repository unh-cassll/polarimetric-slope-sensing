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
  4. For three centered circular apertures -- full frame, 0.5x, and 0.25x of
     the inscribed-circle diameter (min(frame width, height)) -- re-run only
     the (cheap) CWT-based long-wave inversion via the aperture's spatial-mean
     slope, yielding three eta_long(t) series. The per-frame g2s short-wave
     integration is NOT run: the averaging aperture only affects the long-wave
     inversion, and the resolved short-wave field is aperture-independent, so
     it is excluded from this comparison (and skipping it avoids the expensive
     g2s loop entirely). The aperture elevation series are thus eta_long only.
  5. Compute the omnidirectional frequency spectrum S(f) (Welch PSD, m^2/Hz)
     for each of the three series and for the full 10 min lidar elevation.
  5b. Compute the slope-inverted elevation spectrum (per-pixel, averaged over
     all pixels; the slope stack first decimated in time, anti-aliased, by
     --time-decimate to cut FFT cost): the slope-component frequency spectra
     summed and converted to elevation via deep-water dispersion,
     S_eta(f) = S_slope(f) * g^2 / (2 pi f)^4, band-limited to f >= --pss-fmin
     because that compensation amplifies low-frequency noise as f^-4.
  6. Plot all spectra on one log-log axis with labeled axes and legend. The
     three aperture curves are solid, the lidar dashed, and the slope-inverted
     dotted.
  7. Write the three PSS aperture elevation time series (full 30 Hz) to a
     NetCDF (--series-out) so a notebook can recompute the spectra itself. The
     lidar is left in its own committed file for the notebook to read alongside.

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
    #   --method M          spectral estimator: logband (default) or welch
    #   --bands-per-octave N  log-band resolution for logband (default: 12)
    #   --seg-seconds S     Welch segment length in seconds (welch only; def 30)
    #   --pss-fmin F        low-f cutoff (Hz) for ALL PSS elevation curves
    #                       (default: 0.08)
    #   --time-decimate N   anti-aliased time decimation for field spectra
    #                       (default: 3, i.e. 30 Hz -> 10 Hz; 1 disables)
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
from scipy.signal import welch, decimate, periodogram

# make examples/_data and the packages importable regardless of invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data  # noqa: E402
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


# Deep-water gravity constant for the slope->elevation compensation.
_G = 9.81


def logband_spectrum(eta, fs, bands_per_octave=12, fmin=None, detrend="linear"):
    """Log-frequency band-averaged spectrum (multiresolution).

    A single Welch (or pwelch) call uses ONE window length for all
    frequencies, forcing a single trade between low-frequency resolution and
    high-frequency averaging. This estimator instead computes ONE
    full-record periodogram -- the finest possible frequency resolution -- and
    then averages its bins within logarithmically-spaced frequency bands.

    The result keeps fine resolution at low frequency (few periodogram bins per
    band there, so the swell peak and its low side are preserved) while
    averaging many bins per band at high frequency (variance falls, the noisy
    tail smooths). Equivalently, the number of degrees of freedom per band
    grows with frequency -- the standard convention for omnidirectional wave
    spectra. The band value is the MEAN periodogram density in the band (a
    PSD, m^2/Hz), so curves remain directly comparable and variance-consistent.

    Args:
        eta : 1-D series (m, or slope if used on a component).
        fs : sample rate (Hz).
        bands_per_octave : log-band resolution. Higher = more points / less
            smoothing; lower = smoother. Default 12.
        fmin : lowest band edge (Hz). Defaults to the periodogram's first
            non-zero bin (1 / record length).
        detrend : passed to scipy.signal.periodogram.

    Returns:
        (fc, S, dof) : geometric band-center frequency (Hz), band-mean PSD
        (m^2/Hz), and degrees of freedom per band (~2 x bins averaged).
    """
    eta = np.asarray(eta, dtype=float)
    eta = eta[np.isfinite(eta)]
    if eta.size < 8:
        raise ValueError(f"series too short for a spectrum: {eta.size} samples")
    f, P = periodogram(eta - eta.mean(), fs=fs, window="hann", detrend=detrend)
    f, P = f[1:], P[1:]                      # drop the DC bin
    if fmin is None or fmin < f[0]:
        fmin = f[0]
    fmax = f[-1]
    nb = max(1, int(np.ceil(np.log2(fmax / fmin) * bands_per_octave)))
    edges = fmin * 2.0 ** (np.arange(nb + 1) / bands_per_octave)
    fc, S, dof = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        n = int(m.sum())
        if n > 0:
            fc.append(np.sqrt(lo * hi))      # geometric band center
            S.append(P[m].mean())            # band-mean PSD (density)
            dof.append(2 * n)                # ~2 dof per periodogram bin
    return np.asarray(fc), np.asarray(S), np.asarray(dof)


def field_logband_spectrum(field, fs, bands_per_octave=12, fmin=None):
    """Log-band spectrum of a (T, Ny, Nx) field, averaged over pixels.

    Periodogram along the time axis of every pixel, averaged over pixels, then
    log-band averaged in frequency -- the multiresolution analog of
    field_omnidirectional_spectrum. Per-pixel linear detrend.
    """
    field = np.asarray(field, dtype=float)
    f, P = periodogram(field, fs=fs, window="hann", detrend="linear", axis=0)
    P_omni = P.reshape(P.shape[0], -1).mean(axis=1)
    f, P_omni = f[1:], P_omni[1:]            # drop DC
    if fmin is None or fmin < f[0]:
        fmin = f[0]
    fmax = f[-1]
    nb = max(1, int(np.ceil(np.log2(fmax / fmin) * bands_per_octave)))
    edges = fmin * 2.0 ** (np.arange(nb + 1) / bands_per_octave)
    fc, S, dof = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        n = int(m.sum())
        if n > 0:
            fc.append(np.sqrt(lo * hi))
            S.append(P_omni[m].mean())
            dof.append(2 * n)
    return np.asarray(fc), np.asarray(S), np.asarray(dof)


def field_omnidirectional_spectrum(field, fs, seg_seconds=30.0):
    """Omnidirectional frequency spectrum of a (T, Ny, Nx) field, per pixel.

    Welch PSD along the TIME axis of every pixel independently, then averaged
    over all pixels. This is the frequency spectrum a dense array of co-located
    wave gauges would measure -- the omnidirectional elevation (or slope)
    spectrum of the resolved field. Each pixel series is linearly detrended
    (Welch detrend='linear'), which removes any static per-pixel offset/tilt
    that would otherwise dump energy into the lowest frequency bins.

    Args:
        field : (T, Ny, Nx) stack (elevation in m, or slope dimensionless).
        fs : sample rate (Hz).
        seg_seconds : Welch segment length in seconds (see
            omnidirectional_spectrum).

    Returns:
        (f, S) : frequency (Hz) and pixel-averaged PSD (units^2/Hz), 1-D.
    """
    field = np.asarray(field, dtype=float)
    T = field.shape[0]
    nps = int(min(round(seg_seconds * fs), T))
    # welch along the time axis for every pixel at once, then average over space
    f, S = welch(field, fs=fs, nperseg=nps, detrend="linear", axis=0)
    S_omni = S.reshape(S.shape[0], -1).mean(axis=1)
    return f, S_omni


def slope_inverted_elevation_spectrum(sx, sy, fs, seg_seconds=30.0,
                                      f_min=0.08, method="logband",
                                      bands_per_octave=12):
    """Elevation spectrum inferred from the slope fields' frequency content.

    Forms the per-pixel omnidirectional frequency spectrum of each slope
    component, sums them (S_sx + S_sy = the total slope-frequency PSD), and
    converts to an elevation spectrum via the DEEP-WATER dispersion relation:

        slope amplitude = k * elevation amplitude, with k = (2 pi f)^2 / g,
        so  S_eta(f) = S_slope(f) / k^2 = S_slope(f) * g^2 / (2 pi f)^4.

    This is distinct from the g2s-integrated short-wave elevation: it never
    does the spatial surface integration; it asks "what elevation spectrum do
    the observed slope FREQUENCIES imply under deep-water dispersion?"

    The g^2/(2 pi f)^4 factor amplifies low-frequency noise enormously (~10^4
    at 0.05 Hz), so the result is band-limited to f >= f_min on return: bins
    below f_min are set to NaN (not plotted) rather than displaying amplified
    noise. f_min default 0.08 Hz sits safely below the swell peak (~0.17 Hz).

    Args:
        sx, sy : (T, Ny, Nx) cross- and along-look slope stacks (dimensionless).
        fs : sample rate (Hz).
        seg_seconds : Welch segment length in seconds (method='welch' only).
        f_min : low-frequency cutoff (Hz); bins below are returned as NaN.
        method : 'logband' (multiresolution, default) or 'welch'.
        bands_per_octave : log-band resolution (method='logband' only).

    Returns:
        (f, S_eta) : frequency (Hz) and compensated elevation PSD (m^2/Hz),
        with S_eta = NaN for f < f_min and at f = 0.
    """
    if method == "logband":
        f, S_sx, _ = field_logband_spectrum(sx, fs, bands_per_octave)
        _, S_sy, _ = field_logband_spectrum(sy, fs, bands_per_octave)
    else:
        f, S_sx = field_omnidirectional_spectrum(sx, fs, seg_seconds)
        _, S_sy = field_omnidirectional_spectrum(sy, fs, seg_seconds)
    S_slope = S_sx + S_sy

    omega = 2.0 * np.pi * f
    S_eta = np.full_like(S_slope, np.nan)
    valid = (f >= f_min) & (omega > 0)
    S_eta[valid] = S_slope[valid] * (_G ** 2) / (omega[valid] ** 4)
    return f, S_eta


def inscribed_diameter_m(diag) -> float:
    """Inscribed-circle diameter (m) of the orthorectified output grid.

    The 'full frame' aperture: the largest centered disc that fits inside the
    rectangular frame, i.e. min(frame width, frame height) in meters on the
    downsampled grid that reconstruct_eta_field actually operates on.
    """
    dx_ds = float(diag["dx_ds"])
    Ny, Nx = diag["aperture_mask"].shape
    return dx_ds * min(Ny, Nx)


def _write_aperture_series(path, series_records, fs, dx_m, full_diam,
                           long_wave_ran):
    """Write the per-aperture long-wave elevation time series to NetCDF.

    One variable per aperture (eta_full, eta_0p5, eta_0p25), each the long-wave
    (spatial-mean slope inverted) elevation series eta_long(t) for that
    averaging aperture, at the full frame rate, plus a shared time vector and
    metadata (aperture diameter, fs, ground dx). The resolved short-wave field
    is aperture-independent and is not included. A notebook can open this
    alongside the committed lidar file and recompute the spectra.

    Args:
        path : output .nc path.
        series_records : list of (frac, diameter_m, eta_long) tuples.
        fs : frame rate (Hz) of the series.
        dx_m : ground pixel size (m) of the reconstructed grid.
        full_diam : inscribed-circle diameter (m) of the full-frame aperture.
        long_wave_ran : whether the long-wave inversion was active.
    """
    from netCDF4 import Dataset

    n = len(series_records[0][2])
    t = np.arange(n) / fs

    # frac -> safe variable-name suffix: 1.0->full, 0.5->0p5, 0.25->0p25
    def _suffix(frac):
        if frac == 1.0:
            return "full"
        # format the fraction, strip leading "0", turn the dot into "p"
        return ("%g" % frac).replace("0.", "0p").replace(".", "p")

    with Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.description = ("PSS aperture-averaged water-surface elevation time "
                          "series (frame-center), one per circular averaging "
                          "aperture. Companion to the committed lidar elevation "
                          "file; a notebook computes spectra from both.")
        ds.source = "spectra_vs_aperture.py"
        ds.long_wave_ran = "true" if long_wave_ran else "false"

        ds.createDimension("time", n)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "s"
        tv.long_name = "time from acquisition start"
        tv[:] = t

        for frac, diam, eta in series_records:
            name = f"eta_{_suffix(frac)}"
            v = ds.createVariable(name, "f8", ("time",))
            v.units = "m"
            v.long_name = (f"long-wave elevation, "
                           f"{('full frame' if frac == 1.0 else f'{frac:g}x')} "
                           f"aperture")
            v.aperture_diameter_m = float(diam)
            v.aperture_fraction = float(frac)
            v[:] = np.asarray(eta, dtype=float)

        for sname, val, units in [
            ("framerate", fs, "Hz"),
            ("ground_dx", dx_m, "m"),
            ("full_frame_diameter", full_diam, "m"),
        ]:
            sv = ds.createVariable(sname, "f8")
            sv.units = units
            sv[...] = float(val)

    print(f"wrote aperture elevation series -> {path} "
          f"({len(series_records)} apertures, {n} samples @ {fs:.0f} Hz)")


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
    p.add_argument("--method", choices=("logband", "welch"), default="logband",
                   help="spectral estimator (default: logband). 'logband' is "
                        "multiresolution: one full-record periodogram averaged "
                        "into log-frequency bands, so low frequencies keep fine "
                        "resolution while high frequencies are smoothed (the "
                        "wave-spectrum convention; what a single Welch/pwelch "
                        "call CANNOT do). 'welch' is the classic fixed-window "
                        "estimator (uses --seg-seconds).")
    p.add_argument("--bands-per-octave", type=int, default=12,
                   help="log-band resolution for --method logband (default: 12). "
                        "Higher = more detail / less smoothing.")
    p.add_argument("--seg-seconds", type=float, default=30.0,
                   help="Welch segment length in SECONDS (default: 30). "
                        "Converted to samples per series via each instrument's "
                        "sample rate, so PSS (30 Hz) and lidar (10 Hz) get the "
                        "same low-frequency resolution (~1/seg_seconds). Longer "
                        "= lower frequencies resolved but fewer averages.")
    p.add_argument("--out", default="spectra_vs_aperture.png",
                   help="output figure path")
    p.add_argument("--series-out",
                   default="_data/asit2019_aperture_elevation_60s.nc",
                   help="NetCDF path for the per-aperture elevation time series "
                        "(full 30 Hz), written so a notebook can recompute the "
                        "spectra. The lidar is left in its own committed file. "
                        "Default: _data/asit2019_aperture_elevation_60s.nc "
                        "(beside the other example artifacts; relative to the "
                        "cwd, i.e. run from the repo root).")
    p.add_argument("--pss-fmin", type=float, default=0.05,
                   help="low-frequency cutoff (Hz) for ALL PSS elevation curves "
                        "(default: 0.06). The long-wave inversion has no content "
                        "below its 0.05 Hz CWT floor and the 1/k (and slope-"
                        "inverted 1/k^2) dispersion division amplifies low-f "
                        "noise, so PSS curves are not plotted below this. 0.06 "
                        "matches the 0.05 Hz CWT floor of the inversion. The "
                        "lidar, which legitimately resolves lower, is NOT "
                        "clipped. Sits below the ~0.17 Hz swell peak.")
    p.add_argument("--time-decimate", type=int, default=3,
                   help="decimate the slope stack in time by this factor "
                        "(anti-aliased) before the field spectra (default: 3, "
                        "i.e. 30 Hz -> 10 Hz). Cuts the per-pixel FFT cost; the "
                        "discarded high frequencies are above what the footprint "
                        "resolves spatially anyway. Use 1 to disable.")
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
    # 3+4. The averaging aperture only affects the long-wave (spatial-mean
    #      slope) inversion eta_long(t); the resolved short-wave field is
    #      aperture-INDEPENDENT and is NOT part of this comparison, so we skip
    #      the (expensive) per-frame g2s integration entirely (short_wave=False).
    #      One full-frame pass gives eta_long_full + diag (for dx_ds and the
    #      inscribed diameter); the other apertures recompute only the cheap
    #      1-D long-wave inversion.
    # ------------------------------------------------------------------
    print("reconstructing long-wave eta (full frame; g2s skipped) ...")
    _, eta_long_full, _, _, diag = reconstruct_eta_field(
        base.slope_x, base.slope_y, dx=dx_m, fs=fs, downsample=1,
        aperture_diameter_m=None, long_wave=base.long_wave_ran,
        short_wave=False, verbose=True)

    full_diam = inscribed_diameter_m(diag)

    # The downsampled slope grid and freqs the long-wave step operates on. The
    # driver already downsampled (downsample=1 here), so dx_ds == dx_m and the
    # grid is base.slope_x itself.
    sx_ds, sy_ds = base.slope_x, base.slope_y
    dx_ds = diag["dx_ds"]
    freqs_cwt = np.linspace(0.05, 2.0, 80)           # reconstruct_eta_field default
    water_depth_m = 100.0                            # recon default (deep water)

    series = {}   # label -> eta_long series (for plotting)
    series_records = []   # (frac, diameter_m, eta_long) for the NetCDF writer
    for frac in APERTURE_FRACTIONS:
        if frac == 1.0:
            eta_long = eta_long_full
            diam = full_diam
            label = f"PSS full frame (D={full_diam:.2f} m)"
        elif base.long_wave_ran:
            diam = frac * full_diam
            mask = _circular_aperture_mask(sx_ds.shape[1], sx_ds.shape[2],
                                           dx_ds, diam)
            eta_long = _eta_long_for_aperture(
                sx_ds, sy_ds, fs, mask, freqs_cwt, water_depth_m)
            label = f"PSS {frac:g}x (D={frac*full_diam:.2f} m)"
        else:
            diam = frac * full_diam
            eta_long = np.zeros(sx_ds.shape[0])
            label = f"PSS {frac:g}x (D={frac*full_diam:.2f} m, no long-wave)"

        # The aperture series are the long-wave (mean-slope-inverted) elevation
        # only; the resolved short-wave field is aperture-independent and is
        # deliberately excluded (see comment above).
        series[label] = eta_long
        series_records.append((frac, diam, eta_long))
        print(f"  aperture {frac:>4g}x: eta_long std = "
              f"{np.std(eta_long)*100:.2f} cm")

    # ------------------------------------------------------------------
    # Write the aperture elevation time series (full fs) to NetCDF so a
    # notebook can recompute the spectra. Lidar left in its own file.
    # ------------------------------------------------------------------
    _write_aperture_series(args.series_out, series_records, fs, dx_m,
                           full_diam, base.long_wave_ran)

    # ------------------------------------------------------------------
    # 4b. Slope-inverted elevation spectrum (per-pixel, averaged over pixels):
    #     S_eta(f) from the slope frequency content via deep-water dispersion
    #     (g^2/(2 pi f)^4), band-limited. The slope stack is first decimated in
    #     time (anti-aliased) by --time-decimate -- the discarded high
    #     frequencies are above what the footprint resolves spatially anyway.
    # ------------------------------------------------------------------
    td = max(1, args.time_decimate)
    if td > 1:
        print(f"decimating slope stack in time by {td} "
              f"({fs:.0f} -> {fs/td:.0f} Hz, anti-aliased) ...")
        sx_dec = decimate(sx_ds, td, axis=0, ftype="fir").astype(float)
        sy_dec = decimate(sy_ds, td, axis=0, ftype="fir").astype(float)
        fs_field = fs / td
    else:
        sx_dec, sy_dec, fs_field = sx_ds, sy_ds, fs

    print(f"computing slope-inverted spectrum (per-pixel {args.method} over "
          f"{sx_dec.shape[1]*sx_dec.shape[2]} pixels @ {fs_field:.0f} Hz) ...")
    f_inv, S_inv = slope_inverted_elevation_spectrum(
        sx_dec, sy_dec, fs_field, seg_seconds=args.seg_seconds,
        f_min=args.pss_fmin, method=args.method,
        bands_per_octave=args.bands_per_octave)
    print(f"  slope-inverted : {np.sum(np.isfinite(S_inv))} valid bins "
          f"(f >= {args.pss_fmin} Hz)")

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

    def _clip_low(f, S, fmin):
        """Mask a PSS spectrum below fmin (set NaN) so it is not plotted there.

        The long-wave inversion has no content below its 0.05 Hz CWT floor, and
        the 1/k dispersion division inflates low-f slope noise, so PSS curves
        are not trustworthy below ~fmin. The lidar is exempt (it resolves
        lower). NaN values simply break the line; they do not draw at zero.
        """
        S = np.asarray(S, dtype=float).copy()
        S[np.asarray(f) < fmin] = np.nan
        return S

    def _point_spectrum(series_1d, sample_rate):
        """Dispatch a 1-D series to the chosen estimator -> (f, S)."""
        if args.method == "logband":
            f, S, _ = logband_spectrum(series_1d, sample_rate,
                                       bands_per_octave=args.bands_per_octave)
            return f, S
        return omnidirectional_spectrum(series_1d, sample_rate,
                                        seg_seconds=args.seg_seconds)

    pss_colors = ["#1f77b4", "#2ca02c", "#d62728"]   # full, 0.5x, 0.25x
    for (label, eta_c), color in zip(series.items(), pss_colors):
        f, S = _point_spectrum(eta_c, fs)
        ax.loglog(f, _clip_low(f, S, args.pss_fmin), color=color, lw=1.6,
                  label=label)

    f_lid, S_lid = _point_spectrum(eta_lid, fs_lid)
    ax.loglog(f_lid, S_lid, color="k", lw=2.0, ls="--",
              label=f"lidar (10 min, {fs_lid:.0f} Hz)")

    # Slope-inverted field spectrum, dotted so it reads apart from the solid
    # aperture curves and the dashed lidar. Already NaN'd below f_min in its
    # helper.
    ax.loglog(f_inv, S_inv, color="#ff7f0e", lw=1.6, ls=":",
              label=f"PSS slope-inverted (deep-water, f$\\geq${args.pss_fmin:g} Hz)")

    ax.set_xlabel("frequency  $f$  (Hz)")
    ax.set_ylabel(r"elevation spectral density  $S(f)$  (m$^2$/Hz)")
    method_note = (f"log-band {args.bands_per_octave}/oct"
                   if args.method == "logband"
                   else f"Welch {args.seg_seconds:g}s")
    ax.set_title("Omnidirectional elevation spectrum vs averaging aperture\n"
                 f"({method_note})", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    # Focus on the gravity-wave band; trim the empty decades.
    ax.set_xlim(0.03, max(fs, fs_lid) / 2.0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
