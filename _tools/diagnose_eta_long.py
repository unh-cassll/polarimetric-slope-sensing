#!/usr/bin/env python3
"""
diagnose_eta_long.py -- localize the eta_long amplitude on the REAL artifact.

The synthetic single-wave test recovers eta amplitude correctly, but the real
60 s record's eta_long came out ~9x smaller than the slope amplitudes imply.
This script loads the committed spatial-mean slope artifact and dumps the
intermediate quantities of the long-wave inversion so we can see WHERE the
amplitude goes, on the actual data (not a synthetic stand-in).

Run:
    uv run python tools/diagnose_eta_long.py
    # or point at a specific artifact:
    uv run python tools/diagnose_eta_long.py --artifact examples/asit2019_mean_slope_60s.nc

It prints, stage by stage:
  1. input slope std (sx_mean, sy_mean) in rad and deg
  2. the CWT power spectrum vs frequency -- WHERE is the slope energy? If the
     dominant period is < ~1/freqs.min() the long waves fall outside the band.
  3. k(f) across the band, and the 1/k scaling factor at the energy peak
  4. eta_long std, and the naive single-frequency expectation slope_std/k_peak
  5. the ratio, to quantify the loss and point at the cause
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from eta_field_recon.wavelet_core import lindisp_with_current, _cwt, _inverse_cwt
from eta_field_recon.recon import _make_temporal_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path,
                    default=Path("_data/asit2019_mean_slope_60s.nc"))
    args = ap.parse_args()

    with Dataset(str(args.artifact)) as ds:
        sx = np.asarray(ds.variables["sx_mean"][...], float)
        sy = np.asarray(ds.variables["sy_mean"][...], float)
        fs = float(np.asarray(ds.variables["fs"][...]))
        depth = float(np.asarray(ds.variables["water_depth"][...]))

    T = sx.size
    print(f"artifact: {args.artifact.name}")
    print(f"  {T} samples @ {fs} Hz ({T/fs:.1f} s), depth {depth} m")
    print()

    # 1. input slope amplitude
    print("1. INPUT SLOPE")
    print(f"   sx_mean std = {sx.std():.5f} rad ({np.degrees(sx.std()):.4f} deg)")
    print(f"   sy_mean std = {sy.std():.5f} rad ({np.degrees(sy.std()):.4f} deg)")
    print(f"   combined    = {np.sqrt(sx.std()**2+sy.std()**2):.5f} rad")
    print()

    # 2. where is the slope energy in frequency?
    freqs = np.linspace(0.05, 2.0, 80)
    win = _make_temporal_window(T, "tukey", 0.25)
    Wsx = _cwt((sx - sx.mean()) * win, freqs, fs, None).values
    Wsy = _cwt((sy - sy.mean()) * win, freqs, fs, None).values
    power = (np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2).mean(axis=1)  # avg over time
    pk = int(np.argmax(power))
    print("2. SLOPE ENERGY vs FREQUENCY (time-averaged CWT power)")
    print(f"   band: {freqs.min():.3f}..{freqs.max():.3f} Hz "
          f"(longest resolvable period {1/freqs.min():.0f} s)")
    print(f"   peak at f = {freqs[pk]:.3f} Hz (period {1/freqs[pk]:.1f} s)")
    # show the low-frequency tail -- is energy piled up at the band edge?
    print("   power in lowest 5 bins:",
          np.array2string(power[:5] / power.max(), precision=2))
    print("   power at bins near peak:",
          np.array2string(power[max(0, pk-2):pk+3] / power.max(), precision=2))
    frac_lowband = power[freqs < 0.25].sum() / power.sum()
    print(f"   fraction of slope energy below 0.25 Hz: {frac_lowband:.2f}")
    print()

    # 3. dispersion scaling
    _, kk = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)
    print("3. DISPERSION k(f) AND 1/k SCALING")
    print(f"   k at peak f={freqs[pk]:.3f} Hz: {kk[pk]:.4f} rad/m "
          f"(wavelength {2*np.pi/kk[pk]:.1f} m)")
    print(f"   1/k at peak: {1/kk[pk]:.2f} m/rad")
    print()

    # 4. run the actual inversion
    eps = 1e-30
    mag = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + eps
    cos_th = np.abs(Wsx) / mag
    sin_th = (np.abs(Wsy) / mag) * np.sign(np.real(Wsy * np.conj(Wsx)))
    W_eta = 1j * (cos_th * Wsx + sin_th * Wsy) / kk[:, None]
    W_eta = np.where(np.isfinite(W_eta), W_eta, 0.0)
    eta = np.real(_inverse_cwt(W_eta, freqs, fs, None))

    slope_std = np.sqrt(sx.std() ** 2 + sy.std() ** 2)
    naive = slope_std / kk[pk]
    print("4. ELEVATION")
    print(f"   eta_long std            = {eta.std()*100:.3f} cm")
    print(f"   naive slope_std / k_peak = {naive*100:.3f} cm")
    print(f"   ratio (eta / naive)     = {eta.std()/naive:.3f}")
    print()
    print("INTERPRETATION")
    if frac_lowband < 0.3:
        print("   Most slope energy is at HIGHER freq than swell -> short waves")
        print("   dominate the spatial mean. Their small 1/k gives small eta.")
        print("   The mean-wave (long-wave) signal may be genuinely small, OR")
        print("   the spatial mean is being dominated by short-wave residual.")
    else:
        print("   Slope energy IS concentrated at low freq (swell).")
        print("   If eta is still small, suspect the inversion scaling.")


if __name__ == "__main__":
    main()
