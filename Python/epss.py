"""
epss -- top-level entry point for the E-PSS framework.

One function, `run_epss`, ties the whole chain together for in-memory data:

    raw DoFP frames  --(pss)-->  slope fields  --(optional ortho + eta)-->  eta(x,y,t)

The behavior scales with what you give it:

  * Pass only an array of raw frames -> every frame is reduced with `pss`
    and you get the stack of slope fields back. Nothing else runs.

  * Additionally pass ALL FOUR acquisition parameters
    (fs, theta_i, freeboard, pixel_pitch) -> the slope stack is also
    orthorectified onto a uniform ground grid (static-platform geometry)
    and the surface elevation eta(x, y, t) is reconstructed, including the
    long-wave (mean-wave) inference. The long-wave path remains gated on
    record length exactly as in eta_pipeline (a short record returns
    eta_long = 0).

The four eta-enabling parameters are all-or-nothing: supplying some but not
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

from pss import compute_slope_field

from eta_field_recon.orthorectify import orthorectify_static
from eta_field_recon.recon import reconstruct_eta_field
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
    slope_results: list          # per-frame pss SlopeResult objects

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

        verbose : print progress and the assumptions made.

    Returns:
        EpssResult. `slope_x` / `slope_y` are always present; the eta fields
        are present only when the acquisition parameters enabled that stage.
    """
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
        print(f"  pss reduce : resolution={resolution!r}, method={method!r}, "
              f"gain_mode={gain_mode!r}{gain_note} "
              f"(assumed L0 layout, n_water={n_water})")
        if eta_ran:
            print(f"  eta stage  : ENABLED (fs={fs} Hz, theta_i={theta_i_mean_deg} "
                  f"deg, freeboard={freeboard_m} m, pitch={pixel_pitch_m*1e6:.3f} um, "
                  f"focal={focal_length_m*1e3:.1f} mm)")
        else:
            print(f"  eta stage  : disabled (acquisition params not supplied) "
                  f"-> returning slope fields only")

    # ------------------------------------------------------------------
    # pss reduction, frame by frame.
    # ------------------------------------------------------------------
    if verbose:
        print(f"  reducing {T} frame(s) with pss ...")

    def _reduce(frame):
        return compute_slope_field(
            frame,
            resolution=resolution,
            method=method,
            gain_mode=gain_mode,
            lab_gain=lab_gain,
            theta_i_mean_deg=theta_i_mean_deg,   # may be None when eta off; ok
            n_water=n_water,
            gain_reference_frame=gain_reference_frame,
        )

    res0 = _reduce(frames[0])
    Ny, Nx = res0.Sx.shape
    slope_x = np.empty((T, Ny, Nx), dtype=float)
    slope_y = np.empty((T, Ny, Nx), dtype=float)
    slope_x[0], slope_y[0] = res0.Sx, res0.Sy
    slope_results = [res0]
    for ti in range(1, T):
        r = _reduce(frames[ti])
        slope_x[ti], slope_y[ti] = r.Sx, r.Sy
        slope_results.append(r)

    # If the eta stage is off, we are done.
    if not eta_ran:
        return EpssResult(
            slope_x=slope_x, slope_y=slope_y, slope_results=slope_results,
            eta_ran=False, long_wave_ran=False,
            gain_mode=gain_mode, gain_auto_median=auto_median_used,
            notes="slope fields only; acquisition parameters not supplied.",
        )

    # ------------------------------------------------------------------
    # Orthorectify the slope stack onto a uniform ground grid (static).
    # ------------------------------------------------------------------
    if verbose:
        print(f"  orthorectifying stack (static geometry) ...")
    pitch_field = pixel_pitch_m * (2.0 if resolution == "native" else 1.0)
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
    f_min = float(freqs_cwt.min()) if freqs_cwt is not None else 0.05
    gate_threshold_s = min_periods / f_min
    record_duration_s = T / fs
    if force_long_wave is None:
        long_wave = record_duration_s >= gate_threshold_s
    else:
        long_wave = bool(force_long_wave)

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
    f_min = float(freqs_cwt.min()) if freqs_cwt is not None else 0.05
    gate_threshold_s = min_periods / f_min
    record_duration_s = T / fs
    if force_long_wave is None:
        long_wave = record_duration_s >= gate_threshold_s
    else:
        long_wave = bool(force_long_wave)

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
