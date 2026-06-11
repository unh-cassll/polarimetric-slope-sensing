"""
DoLP gain correction.

In the original MATLAB implementation, the normalized Stokes parameters were
multiplied by hard-coded scalars (1.2185 for s1, 1.2197 for s2) obtained from
a one-time laboratory calibration with an integrating sphere and rotating
linear polarizer. This module exposes three modes that replace that hard-coded
behavior, following the E-PSS framework of Laxague et al. [2026]:

    "none"       — no gain applied; s1, s2 pass through unchanged.

    "lab"        — fixed lab-calibrated gains. Defaults to the original
                   MATLAB values (1.2185, 1.2197); override with the
                   `lab_gain` argument if you have your own calibration.

    "empirical"  — compute the gain so that a TEMPORAL-MEDIAN reference DoLP
                   lands at the value the ideal Fresnel curve predicts for the
                   camera's known mean angle of incidence. This is the
                   single-camera empirical correction described in section II
                   of the E-PSS paper. Requires `theta_i_mean_deg`, a
                   Fresnel-curve lookup, AND a reference (a temporal-median
                   frame via s1_ref/s2_ref, or a precomputed dolp_obs_median).
                   The gain is NEVER self-referenced against the frame being
                   corrected: that would assume the instantaneous surface is
                   flat on average, which is false frame-by-frame. With no
                   reference supplied, no gain is applied.

Conceptually, the per-Stokes lab gains and the empirical DoLP gain are
equivalent up to the (small) asymmetry between s1 and s2 gains — since
DoLP = sqrt(s1^2 + s2^2), scaling both s1 and s2 by g scales DoLP by g. We
expose them via a single `apply_gain` function that returns the
gain-corrected s1, s2 and reports the effective scalar applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default lab-calibrated gains from the MATLAB implementation
# (sample_slope_field_calculations.m, lines 35-36, dated 2023-11-21).
DEFAULT_LAB_GAIN = (1.2185, 1.2197)


@dataclass
class GainResult:
    s1: np.ndarray
    s2: np.ndarray
    g1: float   # scalar applied to s1
    g2: float   # scalar applied to s2
    mode: str
    notes: str = ""


def _ideal_dolp_from_theta(theta_deg: float, n_water: float = 1.34) -> float:
    """DoLP predicted by Fresnel theory for an unpolarized, no-upwelling sky.

    Uses the closed form:
        DoLP(theta) = 2 sin^2(theta) cos(theta) sqrt(n^2 - sin^2(theta))
                      / (n^2 - sin^2(theta) - n^2 sin^2(theta) + 2 sin^4(theta))
    matching the formula in the repo README.
    """
    th = np.deg2rad(theta_deg)
    s, c = np.sin(th), np.cos(th)
    s2 = s * s
    n2 = n_water * n_water
    num = 2.0 * s2 * c * np.sqrt(n2 - s2)
    den = n2 - s2 - n2 * s2 + 2.0 * s2 * s2
    return float(num / den)


def apply_gain(
    s1: np.ndarray,
    s2: np.ndarray,
    mode: str = "none",
    *,
    lab_gain: tuple[float, float] = DEFAULT_LAB_GAIN,
    theta_i_mean_deg: float | None = None,
    n_water: float = 1.34,
    clip_gain: tuple[float, float] = (0.5, 3.0),
    s1_ref: np.ndarray | None = None,
    s2_ref: np.ndarray | None = None,
    dolp_obs_median: float | None = None,
) -> GainResult:
    """Apply a DoLP-correcting gain to normalized Stokes parameters.

    Parameters
    ----------
    s1, s2 : ndarray
        Normalized Stokes parameters from compute_stokes(). These are the
        arrays the gain is *applied to*.
    mode : {"none", "lab", "empirical"}
        Gain-selection strategy.
    lab_gain : (g1, g2)
        Per-component gains used when mode == "lab". Defaults to the MATLAB
        values (1.2185, 1.2197).
    theta_i_mean_deg : float, optional
        Required when mode == "empirical". Mean angle of incidence across the
        frame, in degrees, computed from the camera-lens geometry.
    n_water : float
        Refractive index of seawater (default 1.34).
    clip_gain : (lo, hi)
        Safety clamp for the empirical gain. Field values reported in the
        E-PSS paper are in the range 1.2-1.7; the window is wider but refuses
        a gain that implies an invalid calibration.
    s1_ref, s2_ref : ndarray, optional
        Reference Stokes parameters from which to derive the *observed* DoLP
        median for the empirical gain. These should come from the per-pixel
        temporal-median background frame (the E-PSS workflow). Both must be
        given together. Takes precedence over `dolp_obs_median`. Ignored
        unless mode == "empirical".
    dolp_obs_median : float, optional
        A pre-computed observed-DoLP median (e.g. a calibration constant for
        the single-frame case where a temporal median can't be formed). Used
        only if `s1_ref`/`s2_ref` are not supplied. Ignored unless
        mode == "empirical".

    Returns
    -------
    GainResult

    Notes
    -----
    The empirical gain calibrates against a TEMPORAL AVERAGE, never the frame
    itself: its `median(DoLP) -> DoLP_ideal(theta_i)` step assumes the
    reference's median surface tilt equals the mean viewing angle (a flat
    surface on average). That holds for a temporal median but not for any
    single instantaneous frame, so self-referencing is not supported.

    Empirical-gain reference precedence (highest first):
        1. `s1_ref`, `s2_ref`   (temporal-median reference frame)
        2. `dolp_obs_median`    (pre-computed scalar; single-frame fallback)
        3. none supplied        -> NO gain is applied (never self-reference)
    """
    mode = mode.lower()
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)

    if mode == "none":
        return GainResult(s1=s1.copy(), s2=s2.copy(), g1=1.0, g2=1.0, mode=mode,
                          notes="no gain applied")

    if mode == "lab":
        g1, g2 = float(lab_gain[0]), float(lab_gain[1])
        return GainResult(s1=s1 * g1, s2=s2 * g2, g1=g1, g2=g2, mode=mode,
                          notes=f"lab gains {g1:.4f}, {g2:.4f}")

    if mode == "empirical":
        if theta_i_mean_deg is None:
            raise ValueError(
                "mode='empirical' requires theta_i_mean_deg "
                "(the camera's mean angle of incidence in degrees)"
            )
        # Determine the observed-DoLP median from a REFERENCE only. The
        # empirical gain assumes the reference's median surface tilt equals the
        # mean viewing angle (a flat surface on average) -- this holds for a
        # temporal average but NOT for a single frame, so we never self-
        # reference. Precedence: a full temporal-median reference FRAME takes
        # priority over a precomputed scalar; if neither is supplied we fall
        # back to NO gain rather than guess.
        if s1_ref is not None or s2_ref is not None:
            if s1_ref is None or s2_ref is None:
                raise ValueError(
                    "s1_ref and s2_ref must be supplied together"
                )
            s1r = np.asarray(s1_ref, dtype=np.float64)
            s2r = np.asarray(s2_ref, dtype=np.float64)
            dolp_obs = float(np.nanmedian(np.sqrt(s1r * s1r + s2r * s2r)))
            ref_note = "ref=temporal-median frame"
        elif dolp_obs_median is not None:
            dolp_obs = float(dolp_obs_median)
            ref_note = "ref=precomputed"
        else:
            # No reference average available -> do NOT self-reference (the
            # single-frame flat-surface assumption is invalid). Apply no gain.
            return GainResult(
                s1=s1.copy(), s2=s2.copy(), g1=1.0, g2=1.0, mode="none",
                notes=(
                    "empirical gain requested but no reference supplied "
                    "(need a temporal-median frame via s1_ref/s2_ref, or a "
                    "precomputed dolp_obs_median); no gain applied"
                ),
            )
        dolp_ideal = _ideal_dolp_from_theta(theta_i_mean_deg, n_water=n_water)
        if not np.isfinite(dolp_obs) or dolp_obs <= 0:
            raise ValueError(
                f"observed DoLP median is {dolp_obs!r} (non-finite or "
                f"non-positive); cannot compute empirical gain -- check the "
                f"reference frame for all-NaN content")
        g = dolp_ideal / dolp_obs
        lo, hi = clip_gain
        g_clipped = float(np.clip(g, lo, hi))
        note = (
            f"empirical gain {g:.4f} "
            f"(median DoLP {dolp_obs:.4f} -> ideal {dolp_ideal:.4f} "
            f"at theta_i={theta_i_mean_deg:.2f} deg, {ref_note})"
        )
        if g != g_clipped:
            note += f"; CLIPPED to {g_clipped:.4f}"
        # Apply the same scalar to both components — this is the single
        # parameter empirical correction of E-PSS section II.
        return GainResult(s1=s1 * g_clipped, s2=s2 * g_clipped,
                          g1=g_clipped, g2=g_clipped, mode=mode, notes=note)

    raise ValueError(f"unknown gain mode {mode!r}; expected 'none', 'lab', or 'empirical'")
