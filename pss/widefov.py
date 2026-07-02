"""
Wide-FOV calibration of the DoLP->angle-of-incidence (AOI) relationship.

A wide-FOV polarimeter views water across a large range of incidence angles in a
single (time-averaged) frame -- one image edge looks toward grazing incidence,
the opposite edge toward nadir (which edge is which depends on the mount; see
`plane_normal_R`'s row_sign) -- so one frame measures the whole DoLP(theta)
curve. A narrow telephoto imager sees only a few degrees of incidence and cannot
self-calibrate that curve; inverting its DoLP through the ideal Fresnel relation
is biased by the real sky polarization and the unpolarized water-leaving pedestal.

This module turns a wide-FOV mean frame into DoLP->AOI lookup tables that a narrow
imager's `compute_slope_field` / `run_epss` reduction can consume in place of the
ideal Fresnel table. Four tables are produced (when their inputs are available):

    lut_fresnel    ideal Fresnel DoLP(theta)            (sky-blind baseline)
    lut_empirical  the wide camera's MEASURED DoLP(theta) (the winner in practice)
    lut_seapol     pure seapol forward prediction        (needs sun geometry + seapol)
    lut_hybrid     seapol prediction with a fitted water-leaving pedestal and a
                   sky-polarization scale anchored to the measured curve

All four are returned in the `pss.fresnel` (DOLP_full, theta_full) format, so any
of them drops straight into `compute_slope_field(lookup_table=...)`.

The wide frame is Stokes-reduced with the Pistellato projective correction
(`pss.pistellato`), because on a wide lens the polarizer-tilt bias is exactly the
several-degree effect this calibration must not inherit.

seapol is an OPTIONAL dependency: the Fresnel and empirical tables never need it;
without it (or without sun geometry) the seapol/hybrid tables are returned as None
with a note, and only a downstream request for them raises.

References:
    Zappa et al. (2008); Laxague et al. (2026, IEEE JOE); Pistellato & Bergamasco
    (2024). The dual-camera recipe follows the Piermont 2025 analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fresnel import build_lookup_table, lut_from_curve
from .pistellato import build_K, corrected_stokes_superpixel, get_camera_rays

# ---- optional seapol dependency (guarded; same pattern as pss.skyaware) ----
_SEAPOL_IMPORT_ERROR = None
try:
    from seapol.render import make_rayleigh_sky, make_clear_sky
    from seapol.polarization import reflection_chain, apply_mueller
    _HAS_SEAPOL = True
except Exception as _e:                       # ImportError or transitive failure
    _SEAPOL_IMPORT_ERROR = _e
    _HAS_SEAPOL = False


def require_seapol() -> None:
    """Raise an actionable ImportError if the optional seapol dep is unusable."""
    if not _HAS_SEAPOL:
        raise ImportError(
            "The wide-FOV seapol/hybrid LUTs require the optional 'seapol' "
            "package, which is not importable:\n"
            f"    {_SEAPOL_IMPORT_ERROR!r}\n"
            "Install it with:  pip install 'epss[skyaware]'.\n"
            "The ideal-Fresnel and empirical wide LUTs do NOT need seapol."
        )


# ==========================================================================
# Geometry helpers (ported from the Piermont full-set driver).
# ==========================================================================
def plane_normal_R(incidence_deg: float, row_sign: float = -1.0) -> np.ndarray:
    """Rotation whose 3rd column is the camera-facing water-plane normal.

    For a camera tilted `incidence_deg` from nadir about the horizontal (rows)
    axis. Only R[:, 2] is consumed downstream. `row_sign` sets which way
    incidence increases along the image rows: row_sign=-1 (the default, e.g. a
    mount flipped about the horizontal) puts increasing incidence toward the
    TOP of the frame; row_sign=+1 toward the BOTTOM. Verify the sign once per
    mount against the Fresnel DoLP-vs-incidence slope on below-Brewster rows; a
    wrong sign inverts the AOI-vs-row mapping (the measured curve looks like a
    falling branch and the empirical LUT is garbage).
    """
    t = np.radians(incidence_deg)
    n = np.array([0.0, row_sign * np.sin(t), -np.cos(t)])   # faces the camera (-z)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], axis=1)


def per_row_incidence(W: int, H: int, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Angle of incidence (deg) per image row, from the real ray geometry."""
    rays = get_camera_rays(W, H, K).reshape(3, H, W)
    n = R[:, 2]
    col = rays[:, :, W // 2]                          # 3 x H, central column
    cos_t = np.clip(np.sum(-col * n[:, None], axis=0), -1, 1)
    return np.degrees(np.arccos(cos_t))              # H,


def strip_profile(field2d: np.ndarray, strip_half: int = 25) -> np.ndarray:
    """Mean over the central column strip -> per-row profile.

    The strip is clipped to the frame; a negative start index would wrap
    around and average the wrong columns on narrow frames.
    """
    c = field2d.shape[1] // 2
    lo = max(c - strip_half, 0)
    return np.nanmean(field2d[:, lo:c + strip_half + 1], axis=1)


def valid_incidence_mask(theta_row: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-row keep mask for the DoLP-vs-incidence profile.

    When the vertical FOV crosses nadir (low incidence, wide lens) arccos folds
    `theta_row`: it dips toward 0 and rises again, double-mapping rows to the
    same incidence. Keep the LONGER branch (it contains the frame center and
    reaches nadir) and drop the short past-nadir stub, then clip to [lo, hi].
    """
    H = len(theta_row)
    nadir = int(np.argmin(theta_row))
    keep = np.ones(H, bool)
    if 1 < nadir < H - 2 and theta_row[nadir] < 3.0:     # real nadir crossing -> fold
        if nadir < (H - 1 - nadir):                       # shorter stub is above
            keep[:nadir] = False
        else:                                             # shorter stub is below
            keep[nadir + 1:] = False
    keep &= (theta_row >= lo) & (theta_row <= hi)
    return keep


def rolling_nanmedian(a: np.ndarray, w: int) -> np.ndarray:
    """Despike a 1-D profile with a centered rolling median (NaN-aware)."""
    H = len(a)
    h = w // 2
    out = np.full(H, np.nan)
    for i in range(H):
        seg = a[max(0, i - h):min(H, i + h + 1)]
        if np.any(~np.isnan(seg)):
            out[i] = np.nanmedian(seg)
    return out


# ==========================================================================
# seapol forward DoLP(theta) (ported from the Piermont sky-aware demo).
# ==========================================================================
def _reflected_components(theta_deg, sky, camera_heading_deg, n_water):
    """Reflected-sky intensity I and linear-polarized magnitude sqrt(Q^2+U^2)
    per incidence angle (before any water-leaving pedestal)."""
    phi = np.radians(camera_heading_deg)
    I, pol = [], []
    for t in np.radians(np.asarray(theta_deg, float)):
        az = phi + np.pi   # d_out (surface->camera) azimuth = heading + 180
        d_out = np.array([np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.cos(t)])
        M, d_in, _ = reflection_chain(d_out, np.array([0, 0, 1.0]), n_water)
        S_sky = np.asarray(sky((-d_in)[None, :]))[0]
        S = apply_mueller(M, S_sky)
        I.append(S[0])
        pol.append(np.hypot(S[1], S[2]))
    return np.array(I), np.array(pol)


def _make_sky(sun_zenith_deg, sun_azimuth_deg, sky_kind="rayleigh"):
    if sky_kind == "clear":
        return make_clear_sky(sun_zenith_deg, sun_azimuth_deg)
    return make_rayleigh_sky(sun_zenith_deg, sun_azimuth_deg)


def seapol_lut(sun_zenith_deg, sun_azimuth_deg, heading_deg, *,
               pedestal=0.0, scale=1.0, n_water=1.34, sky_kind="rayleigh",
               n_theta=400):
    """Build a DoLP->AOI LUT from the seapol reflected-sky forward model.

    DoLP(theta) = scale * pol(theta) / (I(theta) + pedestal), reduced to the
    `pss.fresnel` table format. pedestal=0, scale=1 gives the pure prediction;
    fitted (pedestal, scale) gives the hybrid.
    """
    require_seapol()
    thg = np.linspace(0.1, 85.0, n_theta)
    sky = _make_sky(sun_zenith_deg, sun_azimuth_deg, sky_kind)
    Ig, polg = _reflected_components(thg, sky, heading_deg, n_water)
    return lut_from_curve(thg, scale * polg / (Ig + pedestal))


def hybrid_fit(theta_deg, dolp_meas, sun_zenith_deg, sun_azimuth_deg,
               heading_deg, *, n_water=1.34, sky_kind="rayleigh"):
    """Fix sun geometry + heading from seapol, then fit only the unpolarized
    water-leaving pedestal P and a sky-polarization scale s:

        DoLP(theta) = s * pol(theta) / (I(theta) + P).

    Returns dict(pedestal, sky_scale, rmse).
    """
    require_seapol()
    from scipy.optimize import least_squares
    th = np.asarray(theta_deg, float)
    y = np.asarray(dolp_meas, float)
    ok = np.isfinite(th) & np.isfinite(y)
    sky = _make_sky(sun_zenith_deg, sun_azimuth_deg, sky_kind)
    I, pol = _reflected_components(th[ok], sky, heading_deg, n_water)

    def resid(p):
        P, s = p
        return s * pol / (I + P) - y[ok]

    r = least_squares(resid, [0.3, 1.0], bounds=([0, 0.2], [5, 3]), max_nfev=2000)
    P, s = r.x
    model = s * pol / (I + P)
    rmse = float(np.sqrt(np.nanmean((model - y[ok]) ** 2)))
    return dict(pedestal=float(P), sky_scale=float(s), rmse=rmse)


# ==========================================================================
# Calibration object + builder.
# ==========================================================================
@dataclass
class WideFOVCalibration:
    """DoLP->AOI lookup tables derived from a wide-FOV mean frame.

    Each `lut_*` is a (DOLP_full, theta_full) tuple in the `pss.fresnel` format
    (or None when its inputs were unavailable). Pass any one as
    `compute_slope_field(lookup_table=...)` or via run_epss's inversion modes.
    """
    theta_deg: np.ndarray            # per-row AOI of the wide frame (valid rows)
    dolp_measured: np.ndarray        # measured DoLP(AOI) rising curve
    lut_fresnel: tuple               # ideal Fresnel
    lut_empirical: tuple             # from this frame's measured curve
    lut_seapol: tuple | None = None  # pure seapol prediction
    lut_hybrid: tuple | None = None  # seapol + fitted pedestal/scale
    n_water: float = 1.34
    hybrid_params: dict | None = None
    notes: str = ""


def calibrate_widefov(
    mean_frame: np.ndarray, *,
    focal_length_m: float,
    pixel_pitch_m: float,
    incidence_mean_deg: float,
    n_water: float = 1.34,
    row_sign: float = -1.0,
    dolp_gain: float = 1.0,
    strip_half: int = 25,
    smooth_window: int = 9,
    aoi_lo: float = 0.0,
    aoi_hi: float = 85.0,
    sun_zenith_deg: float | None = None,
    sun_azimuth_deg: float | None = None,
    heading_deg: float | None = None,
    sky_kind: str = "rayleigh",
    keep_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> WideFOVCalibration:
    """Reduce a (time-averaged) wide-FOV raw frame to DoLP->AOI lookup tables.

    Parameters
    ----------
    mean_frame : ndarray
        Raw DoFP mean frame (2-D) from the wide camera.
    focal_length_m, pixel_pitch_m : float
        Lens focal length and SENSOR pixel pitch (m); build the intrinsics K.
    incidence_mean_deg : float
        Mean incidence angle (deg from nadir) of the frame center.
    n_water : float
        Water refractive index (default 1.34).
    row_sign : float
        Direction incidence increases down the rows (see `plane_normal_R`).
    dolp_gain : float
        Scalar applied to the measured DoLP (e.g. a lab calibration gain).
    strip_half : int
        Half-width (cols) of the central strip averaged into the profile.
    smooth_window : int
        Rolling-median window (rows) used to despike the profile.
    aoi_lo, aoi_hi : float
        Valid-incidence window (deg); rows outside are dropped.
    sun_zenith_deg, sun_azimuth_deg, heading_deg : float, optional
        Sun position (as consumed by seapol's sky model) and camera compass
        heading. All three AND seapol are required to build the seapol/hybrid
        LUTs; otherwise those are left None.
    sky_kind : {"rayleigh", "clear"}
        seapol sky model for the forward prediction.
    keep_mask : ndarray, optional
        Boolean half-resolution mask (True = water to keep). Use it to exclude
        dock/foam/glint; build it however your scene needs (this package does
        not include a scene-specific detector).
    verbose : bool
        Print a short summary.

    Returns
    -------
    WideFOVCalibration
    """
    frame = np.asarray(mean_frame, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"mean_frame must be 2-D; got shape {frame.shape}")

    # 1. Pistellato-corrected Stokes at native (half) resolution.
    S0, s1, s2 = corrected_stokes_superpixel(
        frame, focal_length_m=focal_length_m, pixel_pitch_m=pixel_pitch_m)
    ny, nx = s1.shape

    # 2. Measured DoLP, gained and (optionally) masked.
    dolp = np.sqrt(s1 * s1 + s2 * s2) * dolp_gain
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape != dolp.shape:
            raise ValueError(
                f"keep_mask shape {keep_mask.shape} must match the half-res "
                f"Stokes grid {dolp.shape}")
        dolp = np.where(keep_mask, dolp, np.nan)

    # 3. Per-row angle of incidence from the real ray geometry.
    K = build_K(focal_length_m, pixel_pitch_m, nx, ny, half_res=True)
    R = plane_normal_R(incidence_mean_deg, row_sign=row_sign)
    theta_row = per_row_incidence(nx, ny, K, R)

    # 4. Central-strip profile, despiked, restricted to valid incidence.
    profile = rolling_nanmedian(strip_profile(dolp, strip_half), smooth_window)
    vm = valid_incidence_mask(theta_row, aoi_lo, aoi_hi)
    profile = profile.copy()
    profile[~vm] = np.nan

    # 5. Ideal + empirical LUTs (always available).
    lut_fresnel = build_lookup_table(n_water=n_water)
    lut_empirical = lut_from_curve(theta_row, profile)

    # 6. seapol + hybrid LUTs (optional).
    lut_seapol = lut_hybrid = None
    hybrid_params = None
    notes = ""
    have_geom = (sun_zenith_deg is not None and sun_azimuth_deg is not None
                 and heading_deg is not None)
    if have_geom and _HAS_SEAPOL:
        lut_seapol = seapol_lut(sun_zenith_deg, sun_azimuth_deg, heading_deg,
                                pedestal=0.0, scale=1.0, n_water=n_water,
                                sky_kind=sky_kind)
        m = np.isfinite(theta_row) & np.isfinite(profile)
        hybrid_params = hybrid_fit(theta_row[m], profile[m], sun_zenith_deg,
                                   sun_azimuth_deg, heading_deg,
                                   n_water=n_water, sky_kind=sky_kind)
        lut_hybrid = seapol_lut(sun_zenith_deg, sun_azimuth_deg, heading_deg,
                                pedestal=hybrid_params["pedestal"],
                                scale=hybrid_params["sky_scale"],
                                n_water=n_water, sky_kind=sky_kind)
    elif have_geom and not _HAS_SEAPOL:
        notes = ("sun geometry supplied but seapol unavailable -> "
                 "seapol/hybrid LUTs not built")
    else:
        notes = ("no sun geometry supplied -> seapol/hybrid LUTs not built")

    if verbose:
        nvalid = int(np.isfinite(profile).sum())
        print(f"calibrate_widefov: {nvalid} valid rows, "
              f"AOI {np.nanmin(theta_row[vm]):.1f}-{np.nanmax(theta_row[vm]):.1f} deg, "
              f"peak DoLP {np.nanmax(profile):.3f}"
              + (f"; hybrid pedestal={hybrid_params['pedestal']:.2f} "
                 f"scale={hybrid_params['sky_scale']:.2f}"
                 if hybrid_params else f"; {notes}"))

    return WideFOVCalibration(
        theta_deg=theta_row, dolp_measured=profile,
        lut_fresnel=lut_fresnel, lut_empirical=lut_empirical,
        lut_seapol=lut_seapol, lut_hybrid=lut_hybrid,
        n_water=n_water, hybrid_params=hybrid_params, notes=notes)
