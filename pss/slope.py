"""
End-to-end slope-field computation from a DoFP raw frame.

Pipeline (matches sample_slope_field_calculations.m):
    raw frame  ->  Stokes (S0, s1, s2)
              ->  apply DoLP gain (none / lab / empirical)
              ->  DoLP, orientation
              ->  angle of incidence via Fresnel lookup
              ->  along-look and cross-look slopes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fresnel import build_lookup_table, dolp_to_aoi
from .gain import apply_gain
from .stokes import compute_stokes, by_superpixel


@dataclass
class SlopeResult:
    S0: np.ndarray      # total-intensity Stokes parameter, full resolution
    s1: np.ndarray      # gain-corrected normalized S1
    s2: np.ndarray      # gain-corrected normalized S2
    dolp: np.ndarray    # degree of linear polarization (clipped to [0, 1])
    orientation_deg: np.ndarray  # polarization orientation phi in degrees
    aoi_deg: np.ndarray          # angle of incidence theta_i in degrees
    Sx: np.ndarray      # cross-look slope (mean retained; carries the swell tilt)
    Sy: np.ndarray      # along-look slope (mean retained; carries the swell tilt)
    Ax_deg: np.ndarray  # cross-look angle = atan(Sx) in degrees
    Ay_deg: np.ndarray  # along-look angle  = atan(Sy) in degrees
    mss: float          # mean-square slope = var(Sx) + var(Sy)  [dimensionless]
    gain_g1: float
    gain_g2: float
    gain_mode: str
    gain_notes: str


def compute_slope_field(
    frame: np.ndarray,
    *,
    resolution: str = "native",
    method: str = "bilinear",
    method_kwargs: dict | None = None,
    gain_mode: str = "none",
    lab_gain: tuple[float, float] = (1.2185, 1.2197),
    theta_i_mean_deg: float | None = None,
    n_water: float = 1.34,
    lookup_table: tuple[np.ndarray, np.ndarray] | None = None,
    gain_reference_frame: np.ndarray | None = None,
    dolp_obs_median: float | None = None,
    focal_length_m: float | None = None,
    pixel_pitch_m: float | None = None,
) -> SlopeResult:
    """Compute the full slope-field result from a raw DoFP frame.

    Parameters
    ----------
    frame : ndarray
        Raw DoFP image, 2D.
    resolution : {"native", "pistellato", "full"}
        Output spatial resolution.

        - "native" (default): one Stokes vector per 2x2 super-pixel, returned
          at HALF resolution (H/2, W/2) with no interpolation; the output grid
          equals the measurement grid. The `method` argument is ignored. The effective
          pixel pitch is twice the sensor pitch (pass the doubled dx to any
          downstream wavelength/spectrum computation).
        - "pistellato": like "native" (half resolution, per super-pixel) but the
          Stokes parameters are solved with the Pistellato & Bergamasco (2024)
          projective polarizer-tilt correction. Requires `focal_length_m` and
          `pixel_pitch_m` (to build the camera intrinsics K). Meaningful for
          wide-FOV lenses, where peripheral rays tilt each micropolarizer
          relative to its ray; effectively a no-op for telephoto lenses.
        - "full": interpolate each orientation back to the full (H, W) grid
          using the chosen `method`. This implies 2x the real linear
          resolution but partially corrects the IFOV half-pixel offset.
    method : {"bilinear", "kernel_averaging", "conv_demodulation"}
        Stokes reconstruction method. Only used when resolution == "full".
        ("conv_demodulation" is the exact Ratliff 2009 Method 4 interpolator.)
    method_kwargs : dict, optional
        Extra kwargs forwarded to the chosen Stokes method (full only).
    gain_mode : {"none", "lab", "empirical"}
        DoLP gain strategy. See pss.gain for details.
    lab_gain : (g1, g2)
        Used when gain_mode == "lab".
    theta_i_mean_deg : float, optional
        Required when gain_mode == "empirical".
    n_water : float
        Water refractive index used for the Fresnel curve (default 1.34).
    lookup_table : (DOLP_full, theta_full), optional
        Pre-built DoLP -> AOI lookup. If None, one is generated from
        first-principles Fresnel theory at the given n_water.
    gain_reference_frame : ndarray, optional
        A separate raw DoFP frame (same layout as `frame`) from which to
        derive the observed-DoLP median for the empirical gain, instead of
        `frame` itself. This is the E-PSS workflow: calibrate the gain
        against the per-pixel temporal-median background frame, then apply
        that single scalar to the individual frame being reduced. Only used
        when gain_mode == "empirical". Takes precedence over
        `dolp_obs_median`.
    dolp_obs_median : float, optional
        Pre-computed reference DoLP median for the empirical gain (see
        pss.gain.apply_gain). Lets a driver reduce the reference frame once
        and reuse the scalar across a record instead of re-reducing the
        reference per frame. Only used when gain_mode == "empirical" and
        `gain_reference_frame` is None.
    """
    method_kwargs = method_kwargs or {}
    resolution = resolution.lower()
    if resolution not in ("native", "pistellato", "full"):
        raise ValueError(
            f"resolution must be 'native', 'pistellato', or 'full'; "
            f"got {resolution!r}"
        )
    if resolution == "pistellato" and (focal_length_m is None
                                       or pixel_pitch_m is None):
        raise ValueError(
            "resolution='pistellato' requires focal_length_m and pixel_pitch_m "
            "(to build the camera intrinsics K for the projective correction)."
        )

    def _stokes(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if resolution == "native":
            return by_superpixel(f)
        if resolution == "pistellato":
            from .pistellato import corrected_stokes_superpixel
            return corrected_stokes_superpixel(
                f, focal_length_m=focal_length_m, pixel_pitch_m=pixel_pitch_m)
        return compute_stokes(f, method=method, **method_kwargs)

    # 1. Stokes parameters from the raw frame.
    S0, s1, s2 = _stokes(frame)

    # 1b. Optional reference Stokes for the empirical gain (E-PSS: derive the
    #     DoLP-median from a separate temporal-median background frame).
    s1_ref = s2_ref = None
    if gain_reference_frame is not None and gain_mode.lower() == "empirical":
        _, s1_ref, s2_ref = _stokes(np.asarray(gain_reference_frame))

    # 2. Apply DoLP gain.
    gres = apply_gain(
        s1, s2,
        mode=gain_mode,
        lab_gain=lab_gain,
        theta_i_mean_deg=theta_i_mean_deg,
        n_water=n_water,
        s1_ref=s1_ref,
        s2_ref=s2_ref,
        dolp_obs_median=dolp_obs_median,
    )
    s1c, s2c = gres.s1, gres.s2

    # 3. DoLP and polarization orientation phi.
    dolp = np.sqrt(s1c * s1c + s2c * s2c)
    dolp = np.clip(dolp, 0.0, 1.0)
    phi_deg = 0.5 * np.degrees(np.arctan2(s2c, s1c))

    # 4. DoLP -> AOI via Fresnel lookup.
    if lookup_table is None:
        lookup_table = build_lookup_table(n_water=n_water)
    DOLP_full, theta_full = lookup_table
    aoi_deg = dolp_to_aoi(dolp, DOLP_full, theta_full)

    # 5. Along-look / cross-look slopes (in radians of tilt -> tan).
    Sx = np.sin(np.deg2rad(phi_deg)) * np.tan(np.deg2rad(aoi_deg))
    Sy = np.cos(np.deg2rad(phi_deg)) * np.tan(np.deg2rad(aoi_deg))

    # Do NOT subtract the per-frame spatial mean. The mean slope is the
    # swell-induced tilt of the whole footprint, i.e. the long-wave signal
    # eta_long(t) is built from; per-frame de-meaning would destroy it. The
    # constant camera viewing tilt is removed once at the record level
    # downstream (eta_field_recon.recon subtracts the time-mean of the
    # spatial-mean slope before the long-wave inversion). The short-wave path
    # (g2s) discards the spatial mean by construction.

    # 6. Angles (degrees) and mean-square slope.
    Ax = np.degrees(np.arctan(Sx))
    Ay = np.degrees(np.arctan(Sy))
    # Mean-square slope (mss): the classic air-sea definition is the variance
    # of the dimensionless surface slope itself (Sx, Sy are tan of the tilt,
    # i.e. rise-over-run), summed over the two orthogonal components:
    #     mss = var(Sx) + var(Sy)
    # This is dimensionless. Because Sx = tan(theta) ~ theta (radians) for
    # small tilts, mss is numerically close to the tilt-angle variance in
    # rad^2, but the slope variance is the physically standard quantity.
    #
    # (The original MATLAB driver computed var(atand(Ax)) with Ax = atand(Sx):
    # atand applied twice. nanvar(Ax) + nanvar(Ay) applies atand once and does
    # not reproduce that number; mss below is the standard slope variance.)
    #
    # mss uses nanvar, which subtracts the mean internally, so it is invariant
    # to the constant camera-tilt offset present in Sx/Sy. nanmean(Sx**2) would
    # instead add that tilt back as a spurious ~tan(theta_i)^2 term.
    mss = float(np.nanvar(Sx) + np.nanvar(Sy))

    return SlopeResult(
        S0=S0, s1=s1c, s2=s2c,
        dolp=dolp, orientation_deg=phi_deg, aoi_deg=aoi_deg,
        Sx=Sx, Sy=Sy, Ax_deg=Ax, Ay_deg=Ay,
        mss=mss,
        gain_g1=gres.g1, gain_g2=gres.g2,
        gain_mode=gres.mode, gain_notes=gres.notes,
    )
