"""
epss -- top-level entry point for the E-PSS framework.

One function, `run_epss`, ties the whole chain together for in-memory data:

    raw DoFP frames  --(pss)-->  slope fields  --(optional ortho + eta)-->  eta(x,y,t)

The behavior scales with what you give it:

  * Pass only an array of raw frames -> every frame is reduced with `pss`
    and you get the stack of slope fields back. Nothing else runs.

  * Additionally pass ALL FIVE acquisition parameters
    (fs, theta_i, freeboard, pixel_pitch, focal_length) -> the slope stack is
    also orthorectified onto a uniform ground grid (static-platform geometry)
    and the surface elevation eta(x, y, t) is reconstructed, including the
    long-wave (mean-wave) inference. The long-wave path remains gated on
    record length exactly as in eta_pipeline (a short record returns
    eta_long = 0).

The three geometry parameters are all-or-nothing: supplying some but not
all of them raises a clear error rather than silently doing half the job,
because orthorectification and the dispersion-relation inversion each need
the full set to be physically meaningful.

This is the array/in-memory sibling of `eta_field_recon.reconstruct_eta_from
_record`, which does the same chain starting from a NetCDF file on disk.

Defaults for the pss reduction
------------------------------
The reduction parameters not in the acquisition list take `pss`'s own
defaults, which match the common case: L0 super-pixel layout, n_water = 1.34,
and gain_mode = "none" unless a `gain_reference_frame` is supplied (in which
case "empirical" is the sensible choice). All are overridable via kwargs, and
`run_epss` prints what it assumed when verbose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pss import build_lookup_table, compute_slope_field
from pss.stokes import by_superpixel, compute_stokes
from pss.skyaware import (skyaware_slope_stack, solar_position,
                          scene_azimuth_deg, require_seapol)

from eta_field_recon.orthorectify import orthorectify_static
from eta_field_recon.recon import reconstruct_eta_field, long_wave_gate
from eta_field_recon.eta_pipeline import DEFAULT_MIN_PERIODS


# Minimum record length (seconds) for the auto empirical-gain trigger. When no
# explicit gain-reference frame is supplied, the empirical DoLP gain can still
# be calibrated against a temporal-median background formed FROM the stack --
# but only if the record is long enough that its median surface is plausibly
# flat-on-average (the assumption the empirical gain rests on). Below this, a
# stack-median is not a trustworthy flat reference, so no gain is applied. A
# supplied mean/median frame bypasses this gate entirely (it can be used at any
# record length). Overridable per call via `min_gain_seconds`.
DEFAULT_MIN_GAIN_SECONDS = 30.0


@dataclass
class EpssResult:
    """Result of run_epss.

    `slope_x` / `slope_y` are always populated (the (T, Ny, Nx) reduced
    slope stack). The elevation fields are populated only when the
    acquisition parameters enabled the eta stage; otherwise they are None.
    """
    # always present
    slope_x: np.ndarray          # (T, Ny, Nx) cross-look slope stack (rad)
    slope_y: np.ndarray          # (T, Ny, Nx) along-look slope stack (rad)
    slope_results: list          # per-frame pss SlopeResult objects; empty
                                 # unless keep_slope_results=True (each holds
                                 # ~10 full-resolution arrays per frame)

    # eta stage (None unless the acquisition params were supplied)
    eta_xyt: np.ndarray | None = None
    eta_long: np.ndarray | None = None
    eta_short: np.ndarray | None = None
    confidence: np.ndarray | None = None
    diag: dict[str, Any] | None = None
    ortho: Any = None            # OrthoResult, when orthorectified

    # bookkeeping
    eta_ran: bool = False        # whether the eta stage ran
    long_wave_ran: bool = False  # whether the long-wave inversion ran
    dx_m: float | None = None    # ground dx used (from orthorectification)
    inversion: str = "fresnel"   # fresnel | skyaware | empirical_wide |
                                 # seapol_lut | hybrid
    skyaware_env: Any = None     # SkyAwareEnv used (skyaware path only)
    anchor: Any = None           # AnchorResult (skyaware path only)
    gain_mode: str | None = None # resolved gain mode actually used
    gain_auto_median: bool = False  # True if the >=min_gain_seconds auto
                                    # temporal-median reference was used
    notes: str = ""              # human-readable summary of what ran


# The eta stage needs the full imaging geometry. These three are all-or-
# nothing among themselves: supplying some but not all is an error, because
# orthorectification and the dispersion-relation inversion each need the full
# set to be physically meaningful.
#
# Two acquisition params are deliberately NOT in this set, because each is
# meaningful on its own (independent of running eta):
#   - fs               : sets the record duration, which gates the empirical-
#                        gain auto-median trigger;
#   - theta_i_mean_deg : the mean incidence angle, required by the empirical
#                        DoLP gain itself.
# Both may be supplied alone to enable the gain path without running eta. The
# eta stage requires these two AND the full geometry set below.
_ETA_GEOM_NAMES = ("freeboard_m", "pixel_pitch_m", "focal_length_m")


def run_epss(
    frames,
    *,
    # acquisition parameters: all five together enable ortho + eta.
    fs: float | None = None,
    theta_i_mean_deg: float | None = None,
    freeboard_m: float | None = None,
    pixel_pitch_m: float | None = None,
    focal_length_m: float | None = None,
    # eta-stage options (used only when the eta stage runs)
    water_depth_m: float = 100.0,
    camera_azimuth_deg: float | None = None,
    downsample: int = 4,
    freqs_cwt: np.ndarray | None = None,
    min_periods: float = DEFAULT_MIN_PERIODS,
    force_long_wave: bool | None = None,
    spatial_alpha: float = 0.1,
    spatial_pad_frac: float = 0.10,
    temporal_window: str = "tukey",
    temporal_alpha: float = 0.25,
    aperture_diameter_m: float | None = None,
    short_wave: bool = False,
    # pss reduction options (defaults match pss; all overridable)
    resolution: str = "native",
    method: str = "conv_demodulation",
    gain_mode: str | None = None,
    n_water: float = 1.34,
    lab_gain: tuple[float, float] = (1.2185, 1.2197),
    gain_reference_frame=None,
    min_gain_seconds: float = DEFAULT_MIN_GAIN_SECONDS,
    # single-camera sky-aware inversion (mainline option; needs the optional
    # seapol package). inversion="skyaware" replaces the Fresnel DoLP->AoI
    # reduction with seapol's forward-model inversion + the empirical slope
    # anchor; gain_mode then selects the anchor amplitude (none/lab/empirical).
    inversion: str = "fresnel",
    wide_calibration=None,
    acquisition_time_utc=None,
    site_lat: float | None = None,
    site_lon: float | None = None,
    camera_azimuth_compass_deg: float | None = None,
    skyaware_env=None,
    water_case: int = 2,
    theta_v_deg: float | None = None,
    keep_slope_results: bool = False,
    verbose: bool = True,
) -> EpssResult:
    """Reduce raw DoFP frames to slopes, and (optionally) reconstruct eta.

    Args:
        frames : array-like of raw DoFP frames. Either a single 2-D frame
            (H, W) or a stack (T, H, W). A single frame is treated as a
            length-1 record.

        fs, theta_i_mean_deg, freeboard_m, pixel_pitch_m, focal_length_m :
            the acquisition parameters. Supply ALL FIVE to enable
            orthorectification and eta reconstruction; supply NONE to stop
            after producing slope fields. Supplying some-but-not-all raises
            ValueError.
              fs            : frame rate (Hz)
              theta_i_mean_deg : mean incidence angle (deg from vertical)
              freeboard_m   : camera height above the mean surface (m)
              pixel_pitch_m : SENSOR pixel pitch (m). The native-resolution
                              doubling (one Stokes vector per 2x2 super-pixel)
                              is applied internally; pass the raw sensor pitch.
              focal_length_m : lens focal length (m). Required for the ground
                              projection: it sets the image's angular scale
                              (a pixel subtends atan(pitch/focal_length)), so
                              the footprint and ground dx cannot be computed
                              without it.

        water_depth_m : depth for the dispersion relation (default 100 m,
            i.e. effectively deep water). Set to your actual depth in shallows.
        camera_azimuth_deg : look azimuth (deg); recorded, axis-labeling only.
        downsample, freqs_cwt, min_periods, force_long_wave,
        spatial_alpha, spatial_pad_frac, temporal_window, temporal_alpha,
        aperture_diameter_m :
            forwarded to the eta reconstruction (see reconstruct_eta_field /
            reconstruct_eta_from_record). aperture_diameter_m sets the
            diameter (m) of the centered circular aperture over which the
            spatial-mean slope is averaged for the long-wave inversion;
            None (default) uses the full frame.

        resolution, method, gain_mode, n_water, lab_gain,
        gain_reference_frame : forwarded to pss.compute_slope_field for the
            per-frame reduction.
        min_gain_seconds : minimum record length (s) for the empirical gain to
            be auto-enabled WITHOUT an explicit reference frame. Default 30.
            When gain_mode is None (the default), the empirical DoLP gain is
            resolved as follows, given theta_i_mean_deg is present:
              - an explicit gain_reference_frame -> empirical, any length
                (fs is NOT needed in this case);
              - else, record >= min_gain_seconds -> empirical, with the
                reference formed as np.nanmedian over the supplied stack
                (this branch needs fs to compute the record duration);
              - else -> no gain (a single frame is never self-referenced).
            Without theta_i_mean_deg, no empirical gain is applied. If fs is
            None only the auto-median branch is unavailable; an explicit
            reference frame still enables the gain. The resolved mode and
            whether the auto-median was used are reported on EpssResult.

        inversion : DoLP->AOI strategy.
            - "fresnel" (default): ideal Fresnel lookup.
            - "skyaware": seapol forward-model inversion (needs the optional
              seapol package; see the sky-aware params below).
            - "empirical_wide" / "seapol_lut" / "hybrid": use a DoLP->AOI table
              measured/derived from a WIDE-FOV camera. Pass the calibration as
              `wide_calibration`; the table is selected by the mode
              (empirical / pure-seapol / seapol+fitted-pedestal-scale). These
              run the same fast Fresnel reduction loop, just with a different
              lookup table, and need NO seapol at run time (any seapol work
              happened when the calibration was built).
        wide_calibration : a pss.widefov.WideFOVCalibration, required by the
            three wide-FOV inversion modes. Build it with
            pss.widefov.calibrate_widefov on the wide camera's mean frame; the
            narrow camera is then a normal run_epss record that consumes it.
            Ignored by the "fresnel" and "skyaware" modes.

        keep_slope_results : retain the full per-frame SlopeResult objects on
            EpssResult.slope_results. Each holds ~10 full-resolution arrays,
            so a long record at sensor resolution multiplies memory use ~10x;
            off by default.
        verbose : print progress and the assumptions made.

    Returns:
        EpssResult. `slope_x` / `slope_y` are always present; the eta fields
        are present only when the acquisition parameters enabled that stage.
    """
    resolution = str(resolution).lower()
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(
            f"`frames` must be (H, W) or (T, H, W); got shape {frames.shape}."
        )
    T = frames.shape[0]

    # ------------------------------------------------------------------
    # Eta-stage gate.
    #
    # The eta stage (orthorectification + elevation inversion) runs only when
    # the full imaging geometry, the mean incidence angle, AND the frame rate
    # are all available:
    #   - the three geometry params (_ETA_GEOM_NAMES) are all-or-nothing among
    #     themselves -- supplying some but not all is an error;
    #   - theta_i_mean_deg and fs are decoupled from that all-or-nothing set
    #     because each is meaningful on its own (they enable the empirical-gain
    #     path), so either may be supplied alone without running eta.
    # ------------------------------------------------------------------
    geom = dict(freeboard_m=freeboard_m, pixel_pitch_m=pixel_pitch_m,
                focal_length_m=focal_length_m)
    geom_supplied = [n for n in _ETA_GEOM_NAMES if geom[n] is not None]
    geom_complete = len(geom_supplied) == len(_ETA_GEOM_NAMES)
    if geom_supplied and not geom_complete:
        missing = [n for n in _ETA_GEOM_NAMES if geom[n] is None]
        raise ValueError(
            "The eta-stage geometry parameters (freeboard_m, pixel_pitch_m, "
            "focal_length_m) are all-or-nothing: orthorectification and the "
            "eta inversion need the full set. You supplied "
            f"{geom_supplied} but are missing {missing}. Either pass all "
            "three (plus theta_i_mean_deg and fs) to run the eta stage, or "
            "none to stop at the slope fields. (theta_i_mean_deg and fs may "
            "each be passed on their own to enable the empirical gain without "
            "running eta.)"
        )

    # Eta needs the full geometry PLUS theta_i and fs.
    eta_ready = geom_complete and (theta_i_mean_deg is not None) and (fs is not None)
    if geom_complete and not eta_ready:
        missing_extra = [n for n, v in
                         [("theta_i_mean_deg", theta_i_mean_deg), ("fs", fs)]
                         if v is None]
        raise ValueError(
            "The eta-stage geometry is complete but the eta inversion also "
            f"needs {missing_extra}. Pass them to run the eta stage."
        )
    eta_ran = eta_ready

    inversion = str(inversion).lower()
    _WIDE_INVERSIONS = ("empirical_wide", "seapol_lut", "hybrid")
    if inversion not in ("fresnel", "skyaware") + _WIDE_INVERSIONS:
        raise ValueError(
            "inversion must be one of 'fresnel', 'skyaware', 'empirical_wide', "
            f"'seapol_lut', 'hybrid'; got {inversion!r}")
    if inversion in _WIDE_INVERSIONS:
        # Wide-FOV modes are the Fresnel reduction path with the DoLP->AOI table
        # supplied by a wide-camera calibration instead of the ideal Fresnel
        # curve. The LUT is already baked into the calibration, so these modes
        # do NOT need seapol at run time (seapol was only needed when the
        # calibration's seapol/hybrid table was built).
        if wide_calibration is None:
            raise ValueError(
                f"inversion={inversion!r} requires wide_calibration "
                "(build it with pss.widefov.calibrate_widefov on the wide "
                "camera's mean frame).")
    if inversion == "skyaware":
        # Fail fast (and clearly) if the optional seapol dep is missing.
        require_seapol()
        # In sky-aware mode, gain_mode selects the slope-ANCHOR amplitude
        # target (none/lab/empirical), not a DoFP DoLP gain. Default to the
        # in-scene empirical anchor (the mainline single-camera recipe). This
        # also keeps the Fresnel gain-decision block below inert (gain_mode is
        # no longer None), so its auto-median logic does not run.
        gain_mode = gain_mode or "empirical"

    # ------------------------------------------------------------------
    # Gain-mode decision (empirical DoLP gain).
    #
    # The empirical gain needs (a) theta_i_mean_deg, and (b) a flat-on-average
    # reference DoLP. There are two ways to get the reference, in priority
    # order:
    #   1. An explicit `gain_reference_frame` (a mean/median background frame).
    #      Usable at ANY record length.
    #   2. No explicit frame, but the record is long enough (>= min_gain_seconds)
    #      that a temporal median formed FROM the stack is a trustworthy flat
    #      reference. We compute nanmedian over the stack and use it.
    # If neither holds (no frame, and short/over-unknown record), no gain is
    # applied -- we never self-reference a single frame (see pss.gain).
    #
    # The >= min_gain_seconds test needs the record duration, which needs fs.
    # On this array path fs may be None (it is an acquisition param), in which
    # case the auto-median trigger is unavailable; an explicit frame still
    # works.
    # ------------------------------------------------------------------
    record_seconds = (T / fs) if fs is not None else None
    auto_median_used = False
    gain_note = ""
    if gain_mode is None:
        if theta_i_mean_deg is None:
            gain_mode = "none"
            if gain_reference_frame is not None:
                gain_note = (" (gain_reference_frame given but theta_i_mean_deg "
                             "absent -> empirical gain not applied; pass "
                             "theta_i_mean_deg to enable it)")
        elif gain_reference_frame is not None:
            # Explicit reference frame + theta_i -> empirical, any length.
            gain_mode = "empirical"
            gain_note = " (empirical; ref=supplied frame)"
        elif record_seconds is not None and record_seconds >= min_gain_seconds:
            # No explicit frame, but a long-enough record: form the temporal
            # median across the stack and use it as the reference.
            gain_reference_frame = np.nanmedian(frames, axis=0)
            gain_mode = "empirical"
            auto_median_used = True
            gain_note = (f" (empirical; ref=auto nanmedian of {T} frames, "
                         f"record {record_seconds:.1f}s >= "
                         f"{min_gain_seconds:.0f}s)")
        else:
            gain_mode = "none"
            if record_seconds is not None:
                gain_note = (f" (no gain: record {record_seconds:.1f}s < "
                             f"{min_gain_seconds:.0f}s and no reference frame; "
                             f"a stack-median is not a trustworthy flat "
                             f"reference below the threshold)")
            else:
                gain_note = (" (no gain: no reference frame and fs unknown, so "
                             "the record-length trigger is unavailable; pass a "
                             "gain_reference_frame or fs to enable empirical "
                             "gain)")

    if verbose:
        print("run_epss:")
        print(f"  frames     : {T} frame(s) of {frames.shape[1]}x{frames.shape[2]}")
        if inversion == "skyaware":
            print(f"  inversion  : skyaware (seapol forward model), "
                  f"anchor_mode={gain_mode!r} (resolution={resolution!r}, "
                  f"n_water={n_water})")
        else:
            inv_note = (f", inversion={inversion!r} (wide-FOV LUT)"
                        if inversion in _WIDE_INVERSIONS else "")
            print(f"  pss reduce : resolution={resolution!r}, method={method!r}, "
                  f"gain_mode={gain_mode!r}{gain_note}{inv_note} "
                  f"(assumed L0 layout, n_water={n_water})")
        if eta_ran:
            print(f"  eta stage  : ENABLED (fs={fs} Hz, theta_i={theta_i_mean_deg} "
                  f"deg, freeboard={freeboard_m} m, pitch={pixel_pitch_m*1e6:.3f} um, "
                  f"focal={focal_length_m*1e3:.1f} mm)")
        else:
            print(f"  eta stage  : disabled (acquisition params not supplied) "
                  f"-> returning slope fields only")

    # ------------------------------------------------------------------
    # Reduction: raw frames -> per-frame slope stack. The Fresnel lookup (a
    # 200k-point PCHIP fit) is a per-record invariant, hoisted here.
    # ------------------------------------------------------------------
    if inversion in _WIDE_INVERSIONS:
        # Select the wide-calibration table for the requested mode.
        _lut_attr = {"empirical_wide": "lut_empirical",
                     "seapol_lut": "lut_seapol",
                     "hybrid": "lut_hybrid"}[inversion]
        lut = getattr(wide_calibration, _lut_attr, None)
        if lut is None:
            raise ValueError(
                f"inversion={inversion!r} needs wide_calibration.{_lut_attr}, "
                "which is None (the seapol/hybrid tables are only built when "
                "sun geometry AND seapol are available at calibration time).")
    else:
        lut = build_lookup_table(n_water=n_water)
    skyaware_env_used = None
    anchor_used = None

    if inversion == "skyaware":
        # Sky-aware single-camera path: extract Stokes for the whole stack,
        # invert each frame through seapol's forward model, then apply the
        # stack-level empirical slope anchor (it needs the FOV-mean tilt
        # variance across the record, so it cannot be a per-frame step).
        if verbose:
            print(f"  reducing {T} frame(s) via seapol sky-aware inversion ...")

        def _stokes(fr):
            return (by_superpixel(fr) if resolution == "native"
                    else compute_stokes(fr, method=method))

        stok = [_stokes(f) for f in frames]
        S0 = np.stack([a[0] for a in stok])
        s1 = np.stack([a[1] for a in stok])
        s2 = np.stack([a[2] for a in stok])
        del stok

        tv = theta_v_deg if theta_v_deg is not None else theta_i_mean_deg
        if tv is None:
            raise ValueError(
                "inversion='skyaware' needs theta_v_deg (or theta_i_mean_deg): "
                "the camera incidence angle.")
        zen = saz = None
        if skyaware_env is None:
            miss = [n for n, v in
                    (("acquisition_time_utc", acquisition_time_utc),
                     ("site_lat", site_lat), ("site_lon", site_lon),
                     ("camera_azimuth_compass_deg", camera_azimuth_compass_deg))
                    if v is None]
            if miss:
                raise ValueError(
                    "inversion='skyaware' without a precomputed skyaware_env "
                    f"needs the acquisition sun geometry; missing {miss}. (Or "
                    "pass skyaware_env=SkyAwareEnv(...) directly.)")
            zen, azc = solar_position(acquisition_time_utc, site_lat, site_lon)
            saz = scene_azimuth_deg(camera_azimuth_compass_deg, azc)

        slope_x, slope_y, skyaware_env_used, anchor_used = skyaware_slope_stack(
            S0, s1, s2, theta_v_deg=tv, n_water=n_water, env=skyaware_env,
            sun_zenith_deg=zen, sun_azimuth_scene_deg=saz, water_case=water_case,
            gain_mode=gain_mode, lab_gain=lab_gain,
            theta_i_mean_deg=theta_i_mean_deg, lookup_table=lut, verbose=verbose)
        slope_results = []
        if verbose:
            print(f"  anchor     : f={anchor_used.f:.3f} (mode={anchor_used.mode})")
    else:
        # Fresnel DoLP->AoI reduction (the default), frame by frame.
        if verbose:
            print(f"  reducing {T} frame(s) with pss ...")
        dolp_ref = None
        if gain_reference_frame is not None and gain_mode == "empirical":
            ref_res = compute_slope_field(
                np.asarray(gain_reference_frame), resolution=resolution,
                method=method, gain_mode="none", n_water=n_water,
                lookup_table=lut,
                focal_length_m=focal_length_m, pixel_pitch_m=pixel_pitch_m)
            dolp_ref = float(np.nanmedian(
                np.sqrt(ref_res.s1 ** 2 + ref_res.s2 ** 2)))

        def _reduce(frame, use_ref_frame=False):
            return compute_slope_field(
                frame,
                resolution=resolution,
                method=method,
                gain_mode=gain_mode,
                lab_gain=lab_gain,
                theta_i_mean_deg=theta_i_mean_deg,   # may be None (eta off)
                n_water=n_water,
                lookup_table=lut,
                gain_reference_frame=gain_reference_frame if use_ref_frame else None,
                dolp_obs_median=None if use_ref_frame else dolp_ref,
                focal_length_m=focal_length_m,   # used only by resolution="pistellato"
                pixel_pitch_m=pixel_pitch_m,
            )

        res0 = _reduce(frames[0], use_ref_frame=True)
        Ny, Nx = res0.Sx.shape
        slope_x = np.empty((T, Ny, Nx), dtype=float)
        slope_y = np.empty((T, Ny, Nx), dtype=float)
        slope_x[0], slope_y[0] = res0.Sx, res0.Sy
        slope_results = [res0] if keep_slope_results else []
        for ti in range(1, T):
            r = _reduce(frames[ti])
            slope_x[ti], slope_y[ti] = r.Sx, r.Sy
            if keep_slope_results:
                slope_results.append(r)

    # Eta stage off: return slope fields only.
    if not eta_ran:
        return EpssResult(
            slope_x=slope_x, slope_y=slope_y, slope_results=slope_results,
            eta_ran=False, long_wave_ran=False,
            inversion=inversion, skyaware_env=skyaware_env_used,
            anchor=anchor_used,
            gain_mode=gain_mode, gain_auto_median=auto_median_used,
            notes="slope fields only; acquisition parameters not supplied.",
        )

    # ------------------------------------------------------------------
    # Orthorectify the slope stack onto a uniform ground grid (static).
    # ------------------------------------------------------------------
    if verbose:
        print(f"  orthorectifying stack (static geometry) ...")
    pitch_field = pixel_pitch_m * (2.0 if resolution in ("native", "pistellato")
                                   else 1.0)
    ortho = orthorectify_static(
        slope_x, slope_y,
        freeboard_m=freeboard_m,
        theta_i_mean_deg=theta_i_mean_deg,
        focal_length_m=focal_length_m,
        pixel_pitch_m=pitch_field,
        camera_azimuth_deg=camera_azimuth_deg,
        verbose=verbose,
    )

    sx_r = np.nan_to_num(ortho.slope_x, nan=0.0)
    sy_r = np.nan_to_num(ortho.slope_y, nan=0.0)
    dx_m = ortho.dx_m

    # ------------------------------------------------------------------
    # Length gate + eta reconstruction.
    # ------------------------------------------------------------------
    record_duration_s = T / fs
    gate_ok, gate_threshold_s, f_min = long_wave_gate(
        record_duration_s, freqs_cwt, min_periods)
    long_wave = gate_ok if force_long_wave is None else bool(force_long_wave)

    if verbose:
        gate_state = "enabled" if long_wave else "skipped (record too short)"
        print(f"  length gate: record {record_duration_s:.2f} s vs threshold "
              f"{gate_threshold_s:.2f} s -> long-wave {gate_state}")
        print(f"  reconstructing eta ...")

    recon_kwargs = dict(
        dx=dx_m, fs=fs, water_depth_m=water_depth_m, downsample=downsample,
        spatial_alpha=spatial_alpha, spatial_pad_frac=spatial_pad_frac,
        temporal_window=temporal_window, temporal_alpha=temporal_alpha,
        aperture_diameter_m=aperture_diameter_m,
        long_wave=long_wave, short_wave=short_wave, verbose=verbose,
    )
    if freqs_cwt is not None:
        recon_kwargs["freqs_cwt"] = freqs_cwt

    eta_xyt, eta_long, eta_short, confidence, diag = reconstruct_eta_field(
        sx_r, sy_r, **recon_kwargs
    )

    return EpssResult(
        slope_x=slope_x, slope_y=slope_y, slope_results=slope_results,
        eta_xyt=eta_xyt, eta_long=eta_long, eta_short=eta_short,
        confidence=confidence, diag=diag, ortho=ortho,
        eta_ran=True, long_wave_ran=long_wave, dx_m=dx_m,
        inversion=inversion, skyaware_env=skyaware_env_used, anchor=anchor_used,
        gain_mode=gain_mode, gain_auto_median=auto_median_used,
        notes=(f"eta reconstructed; long-wave "
               f"{'ran' if long_wave else 'skipped (record too short)'}."),
    )


def run_epss_from_slopes(
    slope_x,
    slope_y,
    dx_m: float,
    fs: float,
    *,
    # eta-stage options (forwarded to reconstruct_eta_field)
    water_depth_m: float = 100.0,
    downsample: int = 4,
    freqs_cwt: np.ndarray | None = None,
    min_periods: float = DEFAULT_MIN_PERIODS,
    force_long_wave: bool | None = None,
    spatial_alpha: float = 0.1,
    spatial_pad_frac: float = 0.10,
    temporal_window: str = "tukey",
    temporal_alpha: float = 0.25,
    aperture_diameter_m: float | None = None,
    short_wave: bool = False,
    verbose: bool = True,
) -> EpssResult:
    """Reconstruct eta(x, y, t) from ALREADY-ORTHORECTIFIED slope fields.

    This is the entry point for the moving-platform workflow: the user has
    computed slope fields from images taken on a moving platform and then
    orthorectified them THEMSELVES onto a uniform ground grid (so the spatial
    sampling is already uniform with a single known `dx_m`). There is nothing
    left for `pss` to reduce and nothing for the static orthorectifier to do,
    so this function skips straight to the elevation inversion.

    Contrast with `run_epss`, which starts from raw DoFP frames and (when the
    full acquisition geometry is supplied) performs the static
    orthorectification itself. Use `run_epss` for fixed-platform raw frames;
    use this function when you already hold uniform-grid slopes.

    Slope convention
    ----------------
    `slope_x`, `slope_y` are the dimensionless surface slopes Sx = tan(tilt)
    in the cross-look and along-look directions -- exactly what
    `pss.compute_slope_field` emits (SlopeResult.Sx / .Sy) and what
    `reconstruct_eta_field` consumes. NOT tilt angles in degrees or radians.

    Args:
        slope_x, slope_y : orthorectified slope fields, dimensionless
            (tan of tilt). Either a single 2-D frame (Ny, Nx) or a stack
            (T, Ny, Nx). A single frame is always too short for the long-wave
            path (eta_long = 0).
        dx_m : the uniform ground pixel size (m) of the orthorectified grid.
            This is the true projected spacing of one slope sample on the
            water surface; the user set it when they orthorectified.
        fs : frame rate (Hz). Sets the record's time base and, with
            `min_periods`/`freqs_cwt`, the long-wave length gate.

        water_depth_m, downsample, freqs_cwt, min_periods, force_long_wave,
        spatial_alpha, spatial_pad_frac, temporal_window, temporal_alpha,
        aperture_diameter_m : forwarded to `reconstruct_eta_field` (see step-2
            docs). aperture_diameter_m sets the diameter (m) of the centered
            circular aperture over which the spatial-mean slope is averaged for
            the long-wave inversion; None (default) uses the full frame.

        verbose : print progress.

    Returns:
        EpssResult, with the eta fields populated. `slope_x`/`slope_y` echo the
        (stacked) input; `slope_results` is empty (no pss reduction occurred);
        `gain_mode` is None (gain is a raw->slope concern and does not apply to
        pre-computed slopes); `ortho` is None (the caller orthorectified).
    """
    sx = np.asarray(slope_x, dtype=float)
    sy = np.asarray(slope_y, dtype=float)
    if sx.shape != sy.shape:
        raise ValueError(
            f"slope_x and slope_y must have the same shape; got {sx.shape} "
            f"and {sy.shape}."
        )
    if sx.ndim == 2:
        sx = sx[None]
        sy = sy[None]
    if sx.ndim != 3:
        raise ValueError(
            f"slopes must be (Ny, Nx) or (T, Ny, Nx); got shape {sx.shape}."
        )
    T = sx.shape[0]

    # Replace any non-finite slopes (e.g. ortho no-data border) with 0 so g2s
    # and the CWT receive finite input -- matching run_epss's handling of the
    # static-ortho output.
    sx = np.nan_to_num(sx, nan=0.0)
    sy = np.nan_to_num(sy, nan=0.0)

    # ------------------------------------------------------------------
    # Length gate: does the record span enough of the lowest CWT frequency's
    # period to attempt the long-wave inversion? Identical physics to run_epss
    # and reconstruct_eta_from_record.
    # ------------------------------------------------------------------
    record_duration_s = T / fs
    gate_ok, gate_threshold_s, f_min = long_wave_gate(
        record_duration_s, freqs_cwt, min_periods)
    long_wave = gate_ok if force_long_wave is None else bool(force_long_wave)

    if verbose:
        print("run_epss_from_slopes:")
        print(f"  slopes      : {T} frame(s) of {sx.shape[1]}x{sx.shape[2]}, "
              f"dx={dx_m*1000:.3f} mm (pre-orthorectified)")
        gate_state = "enabled" if long_wave else "skipped (record too short)"
        print(f"  length gate : record {record_duration_s:.2f} s vs threshold "
              f"{gate_threshold_s:.2f} s -> long-wave {gate_state}")
        if aperture_diameter_m is not None:
            print(f"  aperture    : circular D={aperture_diameter_m:.3f} m")
        print(f"  reconstructing eta ...")

    recon_kwargs = dict(
        dx=dx_m, fs=fs, water_depth_m=water_depth_m, downsample=downsample,
        spatial_alpha=spatial_alpha, spatial_pad_frac=spatial_pad_frac,
        temporal_window=temporal_window, temporal_alpha=temporal_alpha,
        aperture_diameter_m=aperture_diameter_m,
        long_wave=long_wave, short_wave=short_wave, verbose=verbose,
    )
    if freqs_cwt is not None:
        recon_kwargs["freqs_cwt"] = freqs_cwt

    eta_xyt, eta_long, eta_short, confidence, diag = reconstruct_eta_field(
        sx, sy, **recon_kwargs
    )

    return EpssResult(
        slope_x=sx, slope_y=sy, slope_results=[],
        eta_xyt=eta_xyt, eta_long=eta_long, eta_short=eta_short,
        confidence=confidence, diag=diag, ortho=None,
        eta_ran=True, long_wave_ran=long_wave, dx_m=dx_m,
        gain_mode=None, gain_auto_median=False,
        notes=("eta reconstructed from pre-orthorectified slopes; long-wave "
               f"{'ran' if long_wave else 'skipped (record too short)'}."),
    )
