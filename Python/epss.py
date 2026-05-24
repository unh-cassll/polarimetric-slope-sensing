"""
epss -- top-level entry point for the E-PSS framework.

One function, `run_epss`, ties the whole chain together for in-memory data:

    raw DoFP frames  --(pss)-->  slope fields  --(optional ortho + eta)-->  eta(x,y,t)

The behaviour scales with what you give it:

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
    notes: str = ""              # human-readable summary of what ran


_ACQ_NAMES = ("fs", "theta_i_mean_deg", "freeboard_m", "pixel_pitch_m",
              "focal_length_m")


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
    # pss reduction options (defaults match pss; all overridable)
    resolution: str = "native",
    method: str = "conv_demodulation",
    gain_mode: str | None = None,
    n_water: float = 1.34,
    lab_gain: tuple[float, float] = (1.2185, 1.2197),
    gain_reference_frame=None,
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
        camera_azimuth_deg : look azimuth (deg); recorded, axis-labelling only.
        downsample, freqs_cwt, min_periods, force_long_wave,
        spatial_alpha, spatial_pad_frac, temporal_window, temporal_alpha :
            forwarded to the eta reconstruction (see reconstruct_eta_field /
            reconstruct_eta_from_record).

        resolution, method, gain_mode, n_water, lab_gain,
        gain_reference_frame : forwarded to pss.compute_slope_field for the
            per-frame reduction. gain_mode defaults to "empirical" when a
            gain_reference_frame is given, else "none".

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
    # All-or-nothing gate on the acquisition parameters.
    # ------------------------------------------------------------------
    acq = dict(fs=fs, theta_i_mean_deg=theta_i_mean_deg,
               freeboard_m=freeboard_m, pixel_pitch_m=pixel_pitch_m,
               focal_length_m=focal_length_m)
    supplied = [n for n in _ACQ_NAMES if acq[n] is not None]
    eta_ran = len(supplied) == len(_ACQ_NAMES)
    if supplied and not eta_ran:
        missing = [n for n in _ACQ_NAMES if acq[n] is None]
        raise ValueError(
            "The acquisition parameters (fs, theta_i_mean_deg, freeboard_m, "
            "pixel_pitch_m, focal_length_m) are all-or-nothing: "
            "orthorectification and the eta inversion need the full set. You "
            f"supplied {supplied} but are missing {missing}. Either pass all "
            "five to run the eta stage, or none to stop at the slope fields."
        )

    # gain mode default depends on whether a reference frame was given AND
    # whether theta_i is available (empirical gain requires theta_i). If a
    # reference frame is supplied without theta_i, empirical can't run, so we
    # fall back to no gain rather than crashing deep in pss.
    gain_note = ""
    if gain_mode is None:
        if gain_reference_frame is not None and theta_i_mean_deg is not None:
            gain_mode = "empirical"
        elif gain_reference_frame is not None:
            gain_mode = "none"
            gain_note = (" (gain_reference_frame given but theta_i_mean_deg "
                         "absent -> empirical gain not applied; pass "
                         "theta_i_mean_deg to enable it)")
        else:
            gain_mode = "none"

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
        long_wave=long_wave, verbose=verbose,
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
        notes=(f"eta reconstructed; long-wave "
               f"{'ran' if long_wave else 'skipped (record too short)'}."),
    )
