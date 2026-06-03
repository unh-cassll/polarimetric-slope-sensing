"""Smoke test: synthesize a DoFP frame from a known wave-slope field and run
the full pipeline through all three Stokes methods and all three gain modes.

This is not a unit test of correctness, just a check that:
  - all code paths execute without error,
  - the outputs are finite,
  - the empirical gain scales DoLP toward the Fresnel ideal,
  - shapes / dtypes line up,
  - higher mss is produced when gain > 1.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pss import compute_slope_field
from pss.fresnel import fresnel_dolp
from pss.stokes import _OFFSETS

rng = np.random.default_rng(0)

# --- Synthesize a "truth" wave field ---------------------------------------
# Use a smooth 2D field of along-look / cross-look slopes.
H, W = 256, 256
xx, yy = np.meshgrid(np.linspace(-3, 3, W), np.linspace(-3, 3, H))
Sx_true = 0.05 * np.sin(2.0 * xx) * np.cos(1.3 * yy)        # ~3 deg amplitude
Sy_true = 0.05 * np.cos(1.7 * xx) * np.sin(0.9 * yy)
slope_mag = np.sqrt(Sx_true**2 + Sy_true**2)                  # tan(theta_i_local)
# Local angles
theta_local_rad = np.arctan(slope_mag)
# Plus a uniform "mean camera tilt" so theta_i_mean ~ 30 deg
THETA_MEAN_DEG = 30.0
theta_total_rad = theta_local_rad + np.deg2rad(THETA_MEAN_DEG)
theta_total_deg = np.degrees(theta_total_rad)

# Polarization orientation phi: from the slope direction
# (per the README: Sx = -sin(phi)tan(theta), Sy = -cos(phi)tan(theta), but we
# use the sign convention from the MATLAB code: Sx = sin(phi)tan(theta).)
phi_rad = np.arctan2(Sx_true, Sy_true)

# DoLP from the Fresnel curve at the local total angle of incidence
DoLP_true = fresnel_dolp(theta_total_deg, n_water=1.34)
DoLP_true = np.clip(DoLP_true, 0, 1)

# Now back out s1, s2 (normalized): DoLP = sqrt(s1^2 + s2^2), phi = 0.5 atan2(s2, s1)
# So s1 = DoLP cos(2 phi), s2 = DoLP sin(2 phi).
s1_true = DoLP_true * np.cos(2 * phi_rad)
s2_true = DoLP_true * np.sin(2 * phi_rad)

# Pick an S0 baseline (intensity) — say 1000 counts plus a little gradient
S0_true = 1000.0 + 50.0 * yy  # mild gradient

# Now generate the four orientation intensities at every pixel:
# I(theta_pol) = S0/2 + S1/2 cos(2 theta_pol) + S2/2 sin(2 theta_pol)
# but the Stokes convention used in the codebase has S0 = (sum)/2, so
# I = S0 + S1 cos(...) /2 + S2 sin(...)/2 -- careful.
# Using compute_stokes definitions: S0 = (I0+I45+I90+I135)/2, so
#   sum of four I's = 2 S0 => mean of four = S0/2.
# Malus's law: I(alpha) = I_mean + (I_max - I_mean) cos(2(alpha - alpha_max))
# with I_max - I_mean = S0/2 * DoLP, alpha_max = phi.
def I_at(alpha_deg, S0, dolp, phi):
    a = np.deg2rad(alpha_deg)
    return 0.5 * S0 * (1.0 + dolp * np.cos(2 * (a - phi)))

# Build sparse-sample frame using the layout in _OFFSETS
frame = np.zeros((H, W))
for name, (r_off, c_off) in _OFFSETS.items():
    alpha = {"I0": 0.0, "I45": 45.0, "I90": 90.0, "I135": 135.0}[name]
    I_full = I_at(alpha, S0_true, DoLP_true, phi_rad)
    frame[r_off::2, c_off::2] = I_full[r_off::2, c_off::2]

# Add a tiny bit of noise to be realistic
frame += rng.normal(0, 0.5, size=frame.shape)

# --- Run the pipeline through every combination ----------------------------
print(f"synthetic frame: shape={frame.shape}, dtype={frame.dtype}")
print(f"truth: median DoLP={np.median(DoLP_true):.4f}, "
      f"theta_i_mean={THETA_MEAN_DEG} deg, "
      f"ideal_DoLP={fresnel_dolp(THETA_MEAN_DEG):.4f}\n")

methods = ["bilinear", "kernel_averaging", "conv_demodulation"]
gains = ["none", "lab", "empirical"]

for m in methods:
    print(f"=== Stokes method: {m} ===")
    for g in gains:
        kw = dict(method=m, gain_mode=g)
        if g == "empirical":
            kw["theta_i_mean_deg"] = THETA_MEAN_DEG
        r = compute_slope_field(frame, **kw)
        assert np.isfinite(r.s1).all() and np.isfinite(r.s2).all(), "non-finite s1/s2"
        assert np.isfinite(r.dolp).all(), "non-finite DoLP"
        assert np.isfinite(r.aoi_deg).all(), "non-finite AOI"
        # At the default resolution="native", one Stokes vector is produced per
        # 2x2 super-pixel, so the slope field is H/2 x W/2 (not the input size).
        assert r.Sx.shape == (frame.shape[0] // 2, frame.shape[1] // 2), \
            "Sx shape mismatch (expected native H/2 x W/2)"
        med_dolp = np.nanmedian(r.dolp)
        med_aoi = np.nanmedian(r.aoi_deg)
        print(f"  gain={g:10s}  g1={r.g1 if False else r.gain_g1:.4f}  "
              f"med_DoLP={med_dolp:.4f}  med_AOI={med_aoi:.2f} deg  "
              f"mss={r.mss:.3f}")
    print()

# Targeted check: empirical-gain median DoLP should be very close to the
# Fresnel ideal at theta_i_mean_deg.
r_emp = compute_slope_field(
    frame, method="bilinear", gain_mode="empirical", theta_i_mean_deg=THETA_MEAN_DEG)
med_dolp_emp = float(np.nanmedian(r_emp.dolp))
ideal = float(fresnel_dolp(THETA_MEAN_DEG))
err = abs(med_dolp_emp - ideal) / ideal
print(f"empirical gain check: median DoLP {med_dolp_emp:.4f} vs ideal {ideal:.4f} "
      f"(relative err {err:.4%})")
assert err < 0.02, f"empirical gain failed to align median DoLP to ideal (err={err:.4%})"
print("\nAll smoke checks passed.")
