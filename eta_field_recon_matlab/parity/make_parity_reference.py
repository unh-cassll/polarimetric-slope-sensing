"""Parity reference for the MATLAB port of eta_field_recon.

Synthesizes a (T, Ny, Nx) slope-field stack (three directional linear waves
spanning the long- and short-wave bands, plus seeded noise), runs the Python
reconstruct_eta_field on it, and writes inputs + outputs to a single .mat
file for parity/run_parity_check.m.

Usage: python make_parity_reference.py [output.mat]
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eta_field_recon import reconstruct_eta_field

# Grid: rectangular (Ny != Nx) to exercise axis ordering in the port.
NY, NX, T = 96, 80, 256
DX = 0.025          # m
FS = 10.0           # Hz
DEPTH = 15.0        # m
DOWNSAMPLE = 2

# (amplitude m, wavelength m, direction deg, phase rad); first component is
# the long wave (lambda >> frame), the rest are in-frame short waves.
WAVES = [
    (0.050, 40.00, 20.0, 0.3),
    (0.004, 1.20, 65.0, 1.1),
    (0.002, 0.60, -30.0, 2.5),
    (0.001, 0.35, 130.0, 4.2),
]
NOISE_STD = 1e-4    # rad, on each slope channel


def dispersion_omega(k, depth):
    g, rho_w, sigma = 9.806, 1020.0, 0.072
    return np.sqrt((g * k + sigma / rho_w * k ** 3) * np.tanh(k * depth))


def build_slope_fields():
    x = (np.arange(NX) - NX / 2) * DX
    y = (np.arange(NY) - NY / 2) * DX
    t = np.arange(T) / FS
    X, Y = np.meshgrid(x, y)
    sx = np.zeros((T, NY, NX))
    sy = np.zeros((T, NY, NX))
    for a, lam, deg, ph0 in WAVES:
        k = 2 * np.pi / lam
        om = dispersion_omega(k, DEPTH)
        kx, ky = k * np.cos(np.radians(deg)), k * np.sin(np.radians(deg))
        phase = kx * X[None] + ky * Y[None] - om * t[:, None, None] + ph0
        sx += -a * kx * np.sin(phase)
        sy += -a * ky * np.sin(phase)
    rng = np.random.default_rng(42)
    sx += NOISE_STD * rng.standard_normal(sx.shape)
    sy += NOISE_STD * rng.standard_normal(sy.shape)
    return sx, sy


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("parity_case.mat")
    sx, sy = build_slope_fields()

    t0 = time.perf_counter()
    eta_xyt, eta_long, eta_short, confidence, diag = reconstruct_eta_field(
        sx, sy, dx=DX, fs=FS, water_depth_m=DEPTH, downsample=DOWNSAMPLE,
        verbose=True)
    print(f"  python reconstruction: {time.perf_counter() - t0:.1f} s")
    print(f"  std(eta_long)={eta_long.std():.4e} m, "
          f"std(eta_short)={eta_short.std():.4e} m, "
          f"std(eta_xyt)={eta_xyt.std():.4e} m")

    sio.savemat(out, dict(
        slope_x_field=sx, slope_y_field=sy,
        dx=DX, fs=FS, water_depth_m=DEPTH, downsample=DOWNSAMPLE,
        eta_xyt_py=eta_xyt, eta_long_py=eta_long, eta_short_py=eta_short,
        confidence_py=confidence,
        sx_mean_py=diag["sx_mean"], sy_mean_py=diag["sy_mean"],
    ), do_compression=True, oned_as="column")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
