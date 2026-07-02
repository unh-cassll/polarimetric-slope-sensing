"""
Slope-domain empirical anchor for the sky-aware inversion.

seapol's sky-aware inverter returns TRUE facet slopes with the correct short-wave
/ tail SHAPE but an absolute amplitude set by the inferred environment (which the
blind sky inference can get wrong). This module rescales that stack by a single
isotropic slope gain `f` so its long-wave (FOV-mean) tilt-fluctuation amplitude
matches the in-scene empirical DoLP-gain pipeline -- which itself is derived only
from the in-scene polarimetry (median DoLP + the known camera incidence), no
second reference measurement. Because `f` is a single scalar, it is variance-
normalized-invariant: the Gram-Charlier tail shape is preserved; only the
amplitude (mss, wavenumber and elevation energy) rescales, which also removes the
long/short-wave elevation seam.

`f` requires the temporal variance of the per-frame FOV-mean slope across the
whole stack, so it is a STACK-level operation (it cannot be computed per frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fresnel import build_lookup_table, dolp_to_aoi
from .gain import apply_gain

_F_BOUNDS = (0.2, 6.0)


@dataclass
class AnchorResult:
    f: float                 # isotropic slope-domain gain applied to seapol slopes
    mode: str                # "none" | "lab" | "empirical"
    var_present: float       # temporal var of present-pipeline FOV-mean tilt
    var_seapol: float        # temporal var of seapol FOV-mean tilt
    notes: str = ""


def present_slope_from_stokes(s1, s2, *, gain_mode="empirical",
                              lab_gain=(1.2185, 1.2197), theta_i_mean_deg=None,
                              n_water=1.34, lookup_table=None,
                              dolp_obs_median=None):
    """Present (Fresnel) pipeline slope field FROM Stokes, matching
    pss.slope.compute_slope_field steps 2-5 exactly (apply_gain -> DoLP ->
    dolp_to_aoi -> Sx=sin(phi)tan(aoi), Sy=cos(phi)tan(aoi)). Reused as the
    amplitude reference for the sky-aware anchor."""
    gres = apply_gain(s1, s2, mode=gain_mode, lab_gain=lab_gain,
                      theta_i_mean_deg=theta_i_mean_deg, n_water=n_water,
                      dolp_obs_median=dolp_obs_median)
    s1c, s2c = gres.s1, gres.s2
    dolp = np.clip(np.sqrt(s1c * s1c + s2c * s2c), 0.0, 1.0)
    phi_deg = 0.5 * np.degrees(np.arctan2(s2c, s1c))
    if lookup_table is None:
        lookup_table = build_lookup_table(n_water=n_water)
    DOLP_full, theta_full = lookup_table
    aoi_deg = dolp_to_aoi(dolp, DOLP_full, theta_full)
    Sx = np.sin(np.deg2rad(phi_deg)) * np.tan(np.deg2rad(aoi_deg))
    Sy = np.cos(np.deg2rad(phi_deg)) * np.tan(np.deg2rad(aoi_deg))
    return Sx, Sy


def fov_mean_tilt_series(slope_x, slope_y):
    """Per-frame FOV-mean (spatial-mean) slope -> (mx(t), my(t)), each (T,)."""
    sx = np.asarray(slope_x); sy = np.asarray(slope_y)
    ax = tuple(range(1, sx.ndim))
    return np.nanmean(sx, axis=ax), np.nanmean(sy, axis=ax)


def _longwave_amp2(slope_x, slope_y):
    """Long-wave slope amplitude^2 = temporal variance of the FOV-mean tilt
    summed over the two components (DC view-tilt removed by the variance)."""
    mx, my = fov_mean_tilt_series(slope_x, slope_y)
    return float(np.nanvar(mx) + np.nanvar(my))


def slope_anchor_gain(seapol_x, seapol_y, present_x=None, present_y=None,
                      *, mode="empirical") -> AnchorResult:
    """Isotropic slope gain f matching the seapol long-wave amplitude to the
    present pipeline's. mode='none' -> f=1 (native seapol amplitude); modes
    'empirical'/'lab' differ only in which present pipeline produced
    (present_x, present_y) upstream."""
    if mode == "none" or present_x is None:
        return AnchorResult(f=1.0, mode="none", var_present=float("nan"),
                            var_seapol=_longwave_amp2(seapol_x, seapol_y),
                            notes="native seapol amplitude (f=1)")
    vp = _longwave_amp2(present_x, present_y)
    vs = _longwave_amp2(seapol_x, seapol_y)
    # Degenerate temporal variance (single frame or constant/all-NaN tilt
    # series): amplitude ratio undefined; fall back to f=1 rather than
    # railing to the lower clip bound via 0/eps.
    if not (np.isfinite(vs) and vs > 1e-12 and np.isfinite(vp)):
        return AnchorResult(f=1.0, mode=mode, var_present=vp, var_seapol=vs,
                            notes="degenerate FOV-mean tilt variance (single "
                                  "frame or constant series); f=1 fallback")
    f = float(np.clip(np.sqrt(vp / vs), *_F_BOUNDS))
    return AnchorResult(f=f, mode=mode, var_present=vp, var_seapol=vs,
                        notes=f"f matches long-wave tilt amplitude ({mode})")
