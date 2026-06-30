"""
eta_pipeline -- field-data driver tying pss -> eta_field_recon together.

This is the end-to-end entry point that did not exist before: it reads a
multi-frame DoFP record from NetCDF, reduces every frame to a slope field
with `pss.compute_slope_field`, stacks the results into the (T, Ny, Nx)
arrays that `reconstruct_eta_field` expects, and reconstructs eta(x, y, t).

The long-wave (mean-wave) inversion is *gated on record length*. A record
that spans only a second or two cannot resolve a long swell: the lowest CWT
frequency has a period of 1 / freqs_cwt.min() (20 s at the 0.05 Hz default),
so a 2 s clip does not contain even a tenth of one oscillation and any
"eta_long" it produced would be CWT cone-of-influence artifact rather than
signal. The gate below therefore requires the record to span at least
`min_periods` periods of the lowest CWT frequency before the long-wave path
is run; otherwise it is skipped (long_wave=False) and eta_xyt == eta_short.

The natural defaults (min_periods, the 0.05 Hz floor) put the practical
threshold near the ~10 s figure that motivated this, but the gate is stated
in physical units (periods of the lowest resolved frequency) rather than a
bare wall-clock number, so it tracks freqs_cwt if you change the band.

Pixel scale note
----------------
`reconstruct_eta_field` needs `dx`, the GROUND pixel size in meters -- the
projected spacing of one output sample on the water, NOT the sensor pixel
pitch. The two differ by the imaging geometry (focal length, range to the
surface) and, at pss's default "native" resolution, by a further factor of
two (one Stokes vector per 2x2 super-pixel). With `orthorectify=True` the
driver derives the ground dx from the file's optics (freeboard, theta_i,
focal length, pixel pitch). With `orthorectify=False` you must pass
`ground_dx_m` explicitly:

    ground_dx_m = (range_to_surface_m / focal_length_m) * sensor_pitch_m

doubled if the reduction is at resolution="native". Passing the wrong dx
rescales every wavelength and the dispersion-relation inversion with it, so
it is never guessed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pss import (
    apply_layout_from_meta,
    compute_slope_field,
    read_netcdf_frame,
)

from .recon import reconstruct_eta_field, long_wave_gate
from .orthorectify import orthorectify_static, OrthoResult


# Default gate parameters. The threshold is "record must span >= min_periods
# periods of the lowest CWT frequency". With the reconstruct_eta_field default
# band (freqs_cwt.min() = 0.05 Hz -> 20 s period) and min_periods = 0.5 this
# puts the practical floor at 10 s, matching the figure that motivated the
# gate, while remaining physically defined rather than a bare wall-clock value.
DEFAULT_MIN_PERIODS = 0.5


@dataclass
class PipelineResult:
    """Everything the field-data driver produces, in one object.

    eta_xyt / eta_long / eta_short / confidence / diag are exactly the five
    returns of `reconstruct_eta_field` (eta_long is all-zero when the
    long-wave path was gated off). The remaining fields record what the
    driver itself did, so a caller can tell e.g. whether the mean-wave
    series is real or a placeholder.
    """
    eta_xyt: np.ndarray
    eta_long: np.ndarray
    eta_short: np.ndarray
    confidence: np.ndarray
    diag: dict[str, Any]

    # driver-level bookkeeping
    long_wave_ran: bool          # True if the long-wave inversion was run
    n_frames: int                # frames actually reduced and stacked
    fs_hz: float                 # frame rate used
    record_duration_s: float     # n_frames / fs
    gate_threshold_s: float      # minimum duration that would enable long_wave
    f_min_hz: float              # lowest CWT frequency the gate is keyed to
    gate_reason: str             # human-readable explanation of the decision
    gain_mode_applied: str | None = None  # gain actually applied to the frames
                                          # (the request can downgrade to "none")
    gain_notes: str = ""         # per-frame gain note from the reduction
    orthorectified: bool = False # whether static orthorectification was applied
    ortho: Any = None            # OrthoResult when orthorectified, else None
    # Optional passthrough of the reduced (and, if orthorectify=True,
    # orthorectified) slope stack that was handed to reconstruct_eta_field.
    # Populated only when reconstruct_eta_from_record(..., return_slopes=True).
    # Lets a caller reduce a large record ONCE and then cheaply re-run the
    # CWT-only eta step with different settings (e.g. several aperture
    # diameters) without re-reading and re-reducing the whole stack.
    slope_x: np.ndarray | None = None
    slope_y: np.ndarray | None = None
    slope_dx_m: float | None = None


def _frame_count(path: str | Path) -> int:
    """Number of frames along the record's time axis (1 if no time dim)."""
    from netCDF4 import Dataset  # local import: only the driver needs it

    with Dataset(str(path)) as ds:
        if "time" in ds.dimensions:
            return len(ds.dimensions["time"])
        # A plain 2-D (y, x) frame is a single-frame record.
        return 1


def reconstruct_eta_from_record(
    path: str | Path,
    *,
    ground_dx_m: float | None = None,
    orthorectify: bool = False,
    water_depth_m: float | None = None,
    fs_hz: float | None = None,
    downsample: int = 4,
    freqs_cwt: np.ndarray | None = None,
    min_periods: float = DEFAULT_MIN_PERIODS,
    force_long_wave: bool | None = None,
    # pss reduction options (passed through to compute_slope_field)
    resolution: str = "native",
    method: str = "conv_demodulation",
    gain_mode: str | None = None,
    gain_reference_path: str | Path | None = None,
    reduce_downsample: int = 1,
    # reconstruct_eta_field windowing passthrough
    spatial_alpha: float = 0.1,
    spatial_pad_frac: float = 0.10,
    temporal_window: str = "tukey",
    temporal_alpha: float = 0.25,
    aperture_diameter_m: float | None = None,
    short_wave: bool = False,
    return_slopes: bool = False,
    verbose: bool = True,
) -> PipelineResult:
    """Reconstruct eta(x, y, t) from a multi-frame NetCDF DoFP record.

    Reads the record, reduces every frame to a slope field with `pss`, and
    hands the stack to `reconstruct_eta_field`. The long-wave (mean-wave)
    inversion is enabled only if the record is long enough to resolve the
    lowest CWT frequency (see module docstring and the `min_periods` /
    `force_long_wave` arguments).

    Args:
        path : NetCDF record. Either a (time, x, y) stack or a single 2-D
            frame. A single frame is always too short for the long-wave path.
        ground_dx_m : GROUND pixel size in meters of one reconstructed
            sample (see the module docstring -- this is NOT the sensor pixel
            pitch). Required when orthorectify=False. When orthorectify=True
            it is computed from the acquisition optics and may be left None
            (any value passed is overridden and a note is printed).
        orthorectify : if True, project each slope field onto a uniform
            ground grid before stacking, using the file's static geometry
            (freeboard, theta_i_mean, lens_focal_length, pixel_pitch,
            camera_azimuth). This both corrects the oblique-view trapezoid
            and supplies the true uniform ground dx. STATIC PLATFORM ONLY:
            the geometry is assumed constant across the record (no per-frame
            motion). Default False (no reprojection; caller supplies dx).
        water_depth_m : water depth for the dispersion relation. If None,
            taken from the file's `water_depth` variable when present, else
            falls back to reconstruct_eta_field's own default.
        fs_hz : frame rate. If None, taken from the file's `framerate`.
        downsample : spatial subsample factor for the output grid.
        freqs_cwt : frequency grid (Hz) for the long-wave record-length gate
            (keys off its minimum). If None, linspace(0.05, 2.0, 80) is used.
        min_periods : record must span at least this many periods of the
            lowest CWT frequency for the long-wave path to run. Default 0.5.
        force_long_wave : override the gate. True forces the long-wave path
            on (even for a short record -- expect edge artifact); False forces
            it off; None (default) uses the physics gate.
        resolution, method, gain_mode, gain_reference_path : forwarded to
            `pss.compute_slope_field`. gain_mode defaults to the file's
            recommended mode. If a gain_reference_path is given (the
            temporal-median background frame), it is read once and reused as
            the empirical-gain reference for every frame.
        reduce_downsample : spatial subsample factor applied to each slope
            field AT REDUCE TIME, before the stack is accumulated and (if
            requested) orthorectified. Default 1 (no early subsample). This is
            the memory lever: a full native-resolution slope stack for a long
            high-resolution record can be tens of GB, and orthorectification
            transiently doubles that. Subsampling here, as float32, shrinks
            the peak footprint by reduce_downsample**2. Crucially, the sensor
            pixel pitch handed to the orthorectifier is scaled by the same
            factor, so the derived ground dx -- and therefore every physical
            length and the spectrum's frequency axis -- stays correct (a
            subsample WITHOUT this scaling would silently shrink the footprint
            by reduce_downsample and corrupt the result). This is independent
            of `downsample` below, which subsamples the (already small) stack
            again inside reconstruct_eta_field; the effective output grid is
            coarsened by reduce_downsample * downsample. For a spatial-mean
            long-wave spectrum the fidelity cost of an early subsample is
            negligible (the slope is spatially averaged over the aperture
            anyway); for full-resolution eta_short, keep reduce_downsample=1.
        spatial_alpha, spatial_pad_frac, temporal_window, temporal_alpha :
            forwarded to `reconstruct_eta_field`.
        aperture_diameter_m : diameter (m) of the centered circular aperture
            over which the spatial-mean slope is formed for the long-wave
            inversion; forwarded to `reconstruct_eta_field`. Default None
            (full frame).
        short_wave : if True, run the per-frame g2s integration to produce the
            resolved short-wave field eta_short (and the combined eta_xyt);
            forwarded to `reconstruct_eta_field`. Default False -- the cheap
            path ships only the slope fields (via return_slopes) and eta_long,
            skipping the expensive g2s loop. Set True when you need the
            spatially-resolved elevation field.
        return_slopes : if True, attach the reduced (and, when
            orthorectify=True, orthorectified) slope stack and its ground dx
            to the result as `slope_x`, `slope_y`, `slope_dx_m`. This lets a
            caller reduce a large record ONCE and then cheaply re-run only the
            CWT-based eta step (via `reconstruct_eta_field`) with different
            settings -- e.g. several aperture diameters -- without re-reading
            and re-reducing the whole stack. Default False.
        verbose : print progress.

    Returns:
        PipelineResult.
    """
    path = Path(path)
    # Normalize early: compute_slope_field lowercases `resolution`, so the
    # pitch-doubling checks below must compare on the same form.
    resolution = str(resolution).lower()

    # ------------------------------------------------------------------
    # Resolve frame count, frame rate, depth from the file (+ overrides).
    # ------------------------------------------------------------------
    n_frames = _frame_count(path)

    # Read frame 0 to pull metadata (and to learn the reduced field shape).
    frame0, meta = read_netcdf_frame(path, time_index=0)
    apply_layout_from_meta(meta)

    if fs_hz is None:
        fs_hz = meta.framerate_hz
    if fs_hz is None:
        raise ValueError(
            "frame rate unknown: the file has no usable 'framerate' and "
            "fs_hz was not supplied. Pass fs_hz=... explicitly."
        )

    if water_depth_m is None:
        wd = meta.raw_vars.get("water_depth")
        if wd is not None:
            water_depth_m = float(np.asarray(wd["value"]).flatten()[0])

    if gain_mode is None:
        gain_mode = meta.gain_mode

    # ------------------------------------------------------------------
    # Resolve the ground dx source.
    #   orthorectify=True  -> dx comes from the optics (computed below);
    #                         ground_dx_m is ignored.
    #   orthorectify=False -> ground_dx_m is required (cannot be inferred).
    # Also pull the optics needed for orthorectification from metadata.
    # ------------------------------------------------------------------
    def _raw_scalar(name):
        v = meta.raw_vars.get(name)
        return None if v is None else float(np.asarray(v["value"]).flatten()[0])

    ortho_geom = None
    if orthorectify:
        freeboard_m = _raw_scalar("freeboard")
        focal_mm = _raw_scalar("lens_focal_length")
        pitch_um = _raw_scalar("pixel_pitch")
        azimuth_deg = _raw_scalar("camera_azimuth")
        missing = [n for n, val in
                   [("freeboard", freeboard_m),
                    ("lens_focal_length", focal_mm),
                    ("pixel_pitch", pitch_um),
                    ("theta_i_mean", meta.theta_i_mean_deg)]
                   if val is None]
        if missing:
            raise ValueError(
                "orthorectify=True needs acquisition geometry that the file "
                f"is missing: {missing}. Either supply a file that carries "
                "these, or call with orthorectify=False and an explicit "
                "ground_dx_m."
            )
        ortho_geom = dict(
            freeboard_m=freeboard_m,
            theta_i_mean_deg=meta.theta_i_mean_deg,
            focal_length_m=focal_mm / 1000.0,
            azimuth_deg=azimuth_deg,
            # pitch_um is the SENSOR pitch; the slope field at native
            # resolution samples one Stokes vector per 2x2 super-pixel, so
            # its ground pitch corresponds to 2x the sensor pitch. For a
            # "full"-resolution reduction the field is at the sensor grid.
            pitch_m=(pitch_um / 1e6) * (2.0 if resolution == "native" else 1.0),
        )
        if ground_dx_m is not None and verbose:
            print("  note: orthorectify=True; supplied ground_dx_m is "
                  "overridden by the optics-derived value.")
        ground_dx_m = None   # will be filled from the ortho result
    else:
        if ground_dx_m is None:
            raise ValueError(
                "ground_dx_m is required when orthorectify=False (it cannot "
                "be inferred from the sensor pitch alone -- it depends on the "
                "imaging geometry). Either pass ground_dx_m=... or set "
                "orthorectify=True to derive it from the file's optics."
            )

    # Optional empirical-gain reference frame, read once.
    gain_reference_frame = None
    if gain_reference_path is not None:
        gain_reference_frame, _ = read_netcdf_frame(gain_reference_path,
                                                    time_index=0)

    # ------------------------------------------------------------------
    # Length gate (physics-based): does the record span enough of the
    # lowest CWT frequency's period to attempt the long-wave inversion?
    # ------------------------------------------------------------------
    record_duration_s = n_frames / fs_hz
    gate_ok, gate_threshold_s, f_min = long_wave_gate(
        record_duration_s, freqs_cwt, min_periods)

    if force_long_wave is None:
        long_wave = gate_ok
        if long_wave:
            gate_reason = (
                f"record {record_duration_s:.2f} s >= threshold "
                f"{gate_threshold_s:.2f} s ({min_periods} periods of "
                f"{f_min:.3f} Hz); long-wave inversion enabled"
            )
        else:
            gate_reason = (
                f"record {record_duration_s:.2f} s < threshold "
                f"{gate_threshold_s:.2f} s ({min_periods} periods of "
                f"{f_min:.3f} Hz); long-wave inversion skipped, "
                f"eta_long = 0, eta_xyt = eta_short"
            )
    else:
        long_wave = bool(force_long_wave)
        gate_reason = (
            f"force_long_wave={force_long_wave}: physics gate bypassed "
            f"(record {record_duration_s:.2f} s, threshold "
            f"{gate_threshold_s:.2f} s)"
        )

    if verbose:
        print("reconstruct_eta_from_record:")
        print(f"  file        : {path.name}")
        print(f"  frames      : {n_frames} @ {fs_hz:g} Hz "
              f"-> {record_duration_s:.2f} s")
        dx_str = ("from optics (orthorectify)" if orthorectify
                  else f"{ground_dx_m*1000:.3f} mm")
        print(f"  ground dx   : {dx_str}, "
              f"depth {water_depth_m if water_depth_m is not None else '(default)'} m")
        print(f"  pss reduce  : resolution={resolution!r}, method={method!r}, "
              f"gain_mode={gain_mode!r}"
              f"{' (median ref)' if gain_reference_frame is not None else ''}")
        print(f"  length gate : {gate_reason}")

    # ------------------------------------------------------------------
    # Reduce every frame to a slope field and stack.
    # ------------------------------------------------------------------
    # Per-record invariants, hoisted out of the frame loop: the Fresnel
    # lookup (a 200k-point PCHIP fit) and the empirical-gain reference DoLP
    # (a full Stokes reduction of the reference frame).
    from pss import build_lookup_table
    _lut = build_lookup_table(n_water=meta.n_water)
    _dolp_ref = None
    if gain_reference_frame is not None and str(gain_mode).lower() == "empirical":
        _ref_res = compute_slope_field(
            np.asarray(gain_reference_frame), resolution=resolution,
            method=method, gain_mode="none", n_water=meta.n_water,
            lookup_table=_lut)
        _dolp_ref = float(np.nanmedian(
            np.sqrt(_ref_res.s1 ** 2 + _ref_res.s2 ** 2)))

    def _reduce(frame: np.ndarray, use_ref_frame: bool = False) -> Any:
        return compute_slope_field(
            frame,
            resolution=resolution,
            method=method,
            gain_mode=gain_mode,
            theta_i_mean_deg=meta.theta_i_mean_deg,
            n_water=meta.n_water,
            lookup_table=_lut,
            # Frame 0 reduces against the reference frame itself (canonical
            # notes/bookkeeping); the rest reuse the precomputed scalar.
            gain_reference_frame=gain_reference_frame if use_ref_frame else None,
            dolp_obs_median=None if use_ref_frame else _dolp_ref,
        )

    if verbose:
        print(f"  reducing {n_frames} frame(s) with pss ...")

    rds = int(reduce_downsample)
    if rds < 1:
        raise ValueError(f"reduce_downsample must be >= 1; got {reduce_downsample!r}")

    def _reduce_sub(frame, use_ref_frame=False):
        """Reduce one frame and subsample it (float32) before stacking."""
        res = _reduce(frame, use_ref_frame=use_ref_frame)
        sx = np.asarray(res.Sx, dtype=np.float32)[::rds, ::rds]
        sy = np.asarray(res.Sy, dtype=np.float32)[::rds, ::rds]
        return res, sx, sy

    res0, sx0, sy0 = _reduce_sub(frame0, use_ref_frame=True)
    # Record the gain that was actually applied (the requested mode can be
    # silently downgraded, e.g. empirical with no reference -> none).
    gain_mode_applied = res0.gain_mode
    gain_notes = res0.gain_notes
    if (str(gain_mode).lower() == "empirical"
            and gain_mode_applied != "empirical"):
        warnings.warn(
            f"empirical gain was requested (or implied by the file metadata) "
            f"but could not be applied: {gain_notes or 'no reference'}. "
            f"Slopes are uncalibrated; pass gain_reference_path= to supply a "
            f"temporal-median reference.", UserWarning, stacklevel=2)
    Ny, Nx = sx0.shape
    # float32 stack: halves memory vs float64 and is ample precision for a
    # spatially-averaged long-wave slope spectrum.
    slope_x = np.empty((n_frames, Ny, Nx), dtype=np.float32)
    slope_y = np.empty((n_frames, Ny, Nx), dtype=np.float32)
    slope_x[0] = sx0
    slope_y[0] = sy0

    for ti in range(1, n_frames):
        frame_i, _ = read_netcdf_frame(path, time_index=ti)
        _, sx_i, sy_i = _reduce_sub(frame_i)
        slope_x[ti] = sx_i
        slope_y[ti] = sy_i

    # The sensor pixel pitch corresponds to the NATIVE grid; after an early
    # subsample by rds the effective ground pitch is rds x larger. Scale it so
    # orthorectify_static derives the correct ground dx (a subsample without
    # this scaling would shrink the footprint by rds and corrupt all lengths).
    ortho_pitch_m = ortho_geom["pitch_m"] * rds if orthorectify else None

    # ------------------------------------------------------------------
    # Static orthorectification (optional): project the slope stack onto a
    # uniform ground grid using the fixed acquisition geometry, and adopt
    # the resulting true ground dx. Static platform only.
    # ------------------------------------------------------------------
    ortho_result = None
    if orthorectify:
        if verbose:
            print(f"  orthorectifying stack (static geometry) ...")
        ortho_result = orthorectify_static(
            slope_x, slope_y,
            freeboard_m=ortho_geom["freeboard_m"],
            theta_i_mean_deg=ortho_geom["theta_i_mean_deg"],
            focal_length_m=ortho_geom["focal_length_m"],
            pixel_pitch_m=ortho_pitch_m,
            camera_azimuth_deg=ortho_geom["azimuth_deg"],
            verbose=verbose,
        )
        # griddata leaves NaN outside the footprint; reconstruct_eta_field and
        # g2s need finite slopes. Replace the small no-data border with 0.
        slope_x = np.nan_to_num(ortho_result.slope_x, nan=0.0, copy=False)
        slope_y = np.nan_to_num(ortho_result.slope_y, nan=0.0, copy=False)
        ground_dx_m = ortho_result.dx_m
    elif rds > 1:
        # No orthorectification: the caller-supplied ground_dx_m describes the
        # native grid, so an early subsample by rds makes each retained sample
        # span rds x the ground distance.
        ground_dx_m = ground_dx_m * rds

    if verbose and rds > 1:
        if orthorectify:
            # On the ortho path the scaling happened UPSTREAM: pixel pitch was
            # multiplied by rds before orthorectify_static, which then derived
            # the correct ground dx directly. ground_dx_m already reflects it.
            print(f"  reduce_downsample={rds}: slope stack {slope_x.shape} "
                  f"(float32); sensor pitch was scaled x{rds} for ortho, "
                  f"ground dx={ground_dx_m*1000:.3f} mm")
        else:
            print(f"  reduce_downsample={rds}: slope stack {slope_x.shape} "
                  f"(float32), caller dx scaled x{rds} -> "
                  f"{ground_dx_m*1000:.3f} mm")

    # Guard: reconstruct_eta_field will further subsample by `downsample`, then
    # the per-frame g2s integration needs the (downsampled) grid to exceed its
    # finite-difference stencil. The effective output grid is the post-reduce
    # stack shrunk by `downsample`; if that drops below a safe minimum, fail
    # here with an actionable message rather than deep inside g2s.
    # Applies only when the g2s short-wave integration will actually run;
    # the long-wave path is a spatial mean and works on any grid.
    G2S_MIN_NODES = 16
    eff_ny = slope_x.shape[1] // max(downsample, 1)
    eff_nx = slope_x.shape[2] // max(downsample, 1)
    if short_wave and min(eff_ny, eff_nx) < G2S_MIN_NODES:
        raise ValueError(
            f"reduce_downsample={rds} x downsample={downsample} shrinks the "
            f"output grid to {eff_ny}x{eff_nx}, below the ~{G2S_MIN_NODES}-node "
            f"minimum the g2s surface integration needs. Lower reduce_downsample "
            f"or downsample (their product sets the coarsening), or use a "
            f"higher-resolution record."
        )

    # ------------------------------------------------------------------
    # Reconstruct. Pass the gate decision through as long_wave=.
    # ------------------------------------------------------------------
    recon_kwargs = dict(
        dx=ground_dx_m, fs=fs_hz, downsample=downsample,
        spatial_alpha=spatial_alpha, spatial_pad_frac=spatial_pad_frac,
        temporal_window=temporal_window, temporal_alpha=temporal_alpha,
        aperture_diameter_m=aperture_diameter_m,
        long_wave=long_wave, short_wave=short_wave, verbose=verbose,
    )
    if water_depth_m is not None:
        recon_kwargs["water_depth_m"] = water_depth_m

    eta_xyt, eta_long, eta_short, confidence, diag = reconstruct_eta_field(
        slope_x, slope_y, **recon_kwargs
    )

    return PipelineResult(
        eta_xyt=eta_xyt,
        eta_long=eta_long,
        eta_short=eta_short,
        confidence=confidence,
        diag=diag,
        long_wave_ran=long_wave,
        n_frames=n_frames,
        fs_hz=fs_hz,
        record_duration_s=record_duration_s,
        gate_threshold_s=gate_threshold_s,
        f_min_hz=f_min,
        gate_reason=gate_reason,
        gain_mode_applied=gain_mode_applied,
        gain_notes=gain_notes,
        orthorectified=orthorectify,
        ortho=ortho_result,
        slope_x=slope_x if return_slopes else None,
        slope_y=slope_y if return_slopes else None,
        slope_dx_m=ground_dx_m if return_slopes else None,
    )
