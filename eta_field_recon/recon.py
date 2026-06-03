"""
Slope-field -> water surface elevation eta(x, y, t) reconstruction.

Designed for tight 2-D arrays of slope measurements: e.g. a 512x512
polarimetric or stereo imager at 6 mm pixels (~3 m frame), 10 Hz frame
rate.  Reconstructs eta(x, y, t) over the entire frame at a chosen
downsampled resolution (default 64x64), covering both the "long" wave
band (lambda >> frame, recovered from temporal info on the spatial-mean
slope) and the "short" wave band (lambda <= frame, recovered from
per-frame spatial integration via Harker & O'Leary).

Architecture:
  - eta_short(x, y, t)   per-frame Harker-O'Leary integration of slope.
                         Recovers the wave SHAPE; spatial mean is zero
                         per frame by construction.
                         pyGrad2Surf.g2s does the heavy lifting.

  - eta_long(t)          temporal CWT of the spatial-mean slope -> per-
                         (f, t) Krogstad direction estimator with signed
                         projection -> dispersion-relation integration ->
                         inverse CWT.  Recovers the "integration constant"
                         that varies in time but not in space (within
                         the frame).

  - eta(x, y, t) = eta_short(x, y, t) + eta_long(t)

Windowing:
  - Spatial: reflection-pad the slopes, Tukey-window the padded array,
    integrate with g2s on the padded grid, crop back, Tukey-taper output.
    Defaults are light (Tukey alpha=0.1, pad 10%) because g2s is robust
    to edge artifacts on its own; the option is exposed for noisy-edge
    cases.
  - Temporal: Tukey window applied to the spatial-mean slope BEFORE the
    CWT, suppressing the wavelet's cone-of-influence edge artifacts that
    would otherwise smear any slow drift across the whole record.
    Default Tukey alpha=0.25 (taper outer 12.5% on each end, unit gain
    over the middle 75%).

Returns a confidence mask (T, Ny, Nx) on [0, 1] = spatial_window x
temporal_window so downstream code can weight/crop appropriately.

Dependencies: numpy, scipy, xarray, ewdm, pyGrad2Surf.

GOTCHA: pyGrad2Surf.g2s modifies its input arrays IN PLACE (lines 71-74
of _g2sSylvester apply Householder reflections to F=Zy and G=Zx).  This
module passes copies to be safe; do the same if calling g2s elsewhere.
"""
import numpy as np
from pyGrad2Surf.g2s import g2s
from scipy.signal.windows import tukey, hann
from ewdm.wavelets import Morlet

from .wavelet_core import (_cwt, _inverse_cwt, lindisp_with_current,
                           krogstad_eta_coeffs, skirt_correction)


def _make_2d_window(Ny, Nx, alpha):
    """2-D separable Tukey window with taper fraction alpha (0=rect, 1=Hann)."""
    return np.outer(tukey(Ny, alpha=alpha), tukey(Nx, alpha=alpha))


def _make_temporal_window(T, kind, alpha):
    """1-D temporal window. kind is 'tukey', 'hann', or 'rect'."""
    if kind == 'hann':
        return hann(T, sym=True)
    if kind == 'tukey':
        return tukey(T, alpha=alpha)
    if kind in ('rect', None):
        return np.ones(T)
    raise ValueError(f"unknown temporal window kind {kind!r}")


def _circular_aperture_mask(Ny, Nx, dx, diameter_m):
    """Centered circular aperture mask of a given physical diameter.

    Returns a (Ny, Nx) boolean mask that is True for grid cells whose center
    lies within `diameter_m / 2` of the grid center. Used to restrict the
    spatial-mean slope (the long-wave inversion's input) to a centered
    circular footprint rather than the full rectangular frame.

    The diameter is in METRES and is evaluated on the grid whose spacing is
    `dx` (the caller passes the downsampled spacing dx_ds, so the aperture is
    specified in real-world units independent of the downsample factor).

    A diameter that does not fit inside the frame is allowed -- the mask is
    simply the largest centered disc that does fit, clipped by the frame
    edges. If `diameter_m` is None, callers should skip masking entirely
    (full-frame mean); this helper still returns an all-True mask in that
    case for convenience.
    """
    if diameter_m is None:
        return np.ones((Ny, Nx), dtype=bool)
    if diameter_m <= 0:
        raise ValueError(
            f"aperture_diameter_m must be positive (or None for full frame); "
            f"got {diameter_m!r}"
        )
    yc = (np.arange(Ny) - (Ny - 1) / 2.0) * dx
    xc = (np.arange(Nx) - (Nx - 1) / 2.0) * dx
    YY, XX = np.meshgrid(yc, xc, indexing='ij')
    r = np.sqrt(XX ** 2 + YY ** 2)
    return r <= (diameter_m / 2.0)


def _aperture_spatial_mean(field_stack, mask):
    """Spatial mean of a (T, Ny, Nx) stack over a boolean aperture mask.

    Averages only the cells where `mask` is True, per frame, returning a
    (T,) series. Equivalent to field_stack.mean(axis=(1, 2)) when the mask is
    all-True.
    """
    if mask.all():
        return field_stack.mean(axis=(1, 2))
    m = mask[None, :, :]
    n = float(mask.sum())
    return (field_stack * m).sum(axis=(1, 2)) / n


def reconstruct_eta_field(slope_x_field, slope_y_field, dx, fs,
                           freqs_cwt=None,
                           water_depth_m=100.0,
                           downsample=4,
                           mother=None,
                           inverse_per_scale=False,
                           skirt_correct=False,
                           # spatial windowing
                           spatial_alpha=0.1,
                           spatial_pad_frac=0.10,
                           # temporal windowing
                           temporal_window='tukey',
                           temporal_alpha=0.25,
                           long_wave=True,
                           short_wave=True,
                           aperture_diameter_m=None,
                           verbose=True):
    """
    Reconstruct eta(x, y, t) from a stack of slope images.

    Args:
        slope_x_field, slope_y_field : (T, Ny, Nx) slope time series (rad)
        dx            : pixel size (m); assumed square.
        fs            : frame rate (Hz).
        freqs_cwt     : output CWT frequency grid (Hz) for the long path.
                        Default linspace(0.05, 2.0, 80) Hz.
        water_depth_m : water depth (m) for the dispersion relation.
        downsample    : spatial subsample factor; output is (Ny/ds, Nx/ds).
                        Default 4 (e.g. 256->64).
        mother        : EWDM mother wavelet; default Morlet(6.0).
        inverse_per_scale : if True, the long-wave inverse CWT uses the
                        per-frequency reconstruction-gain calibration instead
                        of the single 1.4383 constant (see
                        wavelet_core._inverse_cwt).  Default False (unchanged).
        skirt_correct : if True, apply the krogstad 1/k(omega) skirt
                        correction (see wavelet_core.skirt_correction) so a
                        unit monochromatic surface component reconstructs at
                        unit amplitude.  Paired with inverse_per_scale.
                        Default False (unchanged).

        spatial_alpha    : Tukey alpha for the 2-D slope window.
                           Default 0.1 (light taper).
        spatial_pad_frac : reflection-padding fraction (per axis) before
                           integration.  Default 0.10.  Set 0 to disable.

        temporal_window : 'tukey', 'hann', or 'rect'.  Default 'tukey'.
        temporal_alpha  : Tukey alpha when kind='tukey'.  Default 0.25.

        long_wave     : if True (default) run the long-wave (spatial-mean
                        slope) CWT inversion and add eta_long(t) to the
                        field.  If False, skip it entirely: eta_long is
                        returned as zeros, eta_xyt == eta_short, and the
                        CWT cost is not paid.  Useful for records too short
                        to resolve any long wave (see eta_pipeline.py, which
                        sets this from a physics-based record-length gate).

        short_wave    : if True (default) run the per-frame Harker-O'Leary g2s
                        integration of slope -> eta_short(x, y, t), the
                        spatially-resolved short-wave field, and combine it
                        with eta_long into eta_xyt. If False, skip the g2s loop
                        entirely (the expensive per-frame step): eta_short and
                        eta_xyt are returned as None and only eta_long(t) is
                        produced. The user-facing entry points (run_epss,
                        run_epss_from_slopes, reconstruct_eta_from_record)
                        default this to False so the cheap path ships just the
                        slope fields and eta_long; callers that need the
                        resolved field (e.g. the field-spectrum tool) opt in.

        aperture_diameter_m : diameter (m) of a centered circular aperture
                        over which the slope is averaged to form the spatial-
                        mean slope series that drives the long-wave inversion.
                        Default None = full frame (average over the entire
                        downsampled grid, the original behavior). A finite
                        value restricts the average to the largest centered
                        disc of that diameter that fits in the frame, which is
                        useful when only a central sub-region of the footprint
                        is trustworthy (e.g. vignetting or grazing-edge
                        artifacts near the frame boundary). Only affects the
                        long-wave (eta_long) path; eta_short is unchanged.

        verbose       : print progress messages.

    Returns:
        eta_xyt   : (T, Ny_d, Nx_d) reconstructed elevation field, or None if
                    short_wave=False.
        eta_long  : (T,) long-wave (DC) time series (zeros if long_wave=False).
        eta_short : (T, Ny_d, Nx_d) zero-mean-per-frame short-wave field, or
                    None if short_wave=False.
        confidence: (T, Ny_d, Nx_d) on [0, 1]: spatial_W x temporal_W.
        diag      : dict of intermediates (CWT coefficients, windows,
                    cropped coordinate vectors, etc.)
    """
    if mother is None:
        mother = Morlet(6.0)
    if freqs_cwt is None:
        freqs_cwt = np.linspace(0.05, 2.0, 80)

    T, Ny, Nx = slope_x_field.shape
    ds = downsample
    sx_ds = slope_x_field[:, ::ds, ::ds]
    sy_ds = slope_y_field[:, ::ds, ::ds]
    Ny_d, Nx_d = sx_ds.shape[1], sx_ds.shape[2]
    dx_ds = dx * ds

    x_ds = (np.arange(Nx_d) - Nx_d/2.0) * dx_ds
    y_ds = (np.arange(Ny_d) - Ny_d/2.0) * dx_ds

    # Centered circular aperture over which the spatial-mean slope is formed
    # for the long-wave inversion. None -> full frame (all-True mask). Built
    # once on the downsampled grid; reused in both long_wave branches.
    aperture_mask = _circular_aperture_mask(Ny_d, Nx_d, dx_ds, aperture_diameter_m)

    if verbose:
        print(f"  reconstruct_eta_field:")
        print(f"    input : {T} frames of {Ny}x{Nx}, dx={dx*1000:.1f} mm")
        print(f"    output: {T} frames of {Ny_d}x{Nx_d}, dx={dx_ds*1000:.1f} mm "
              f"({Ny_d*dx_ds:.2f} x {Nx_d*dx_ds:.2f} m)")
        print(f"    spatial : Tukey alpha={spatial_alpha}, "
              f"reflection pad frac={spatial_pad_frac}")
        print(f"    temporal: {temporal_window!r}"
              f"{'/alpha='+str(temporal_alpha) if temporal_window=='tukey' else ''}")
        if aperture_diameter_m is None:
            print(f"    aperture: full frame "
                  f"({int(aperture_mask.sum())} cells)")
        else:
            print(f"    aperture: circular D={aperture_diameter_m:.3f} m "
                  f"({int(aperture_mask.sum())}/{Ny_d*Nx_d} cells)")

    # ------------------------------------------------------------------
    # eta_short(x, y, t): per-frame Harker-O'Leary integration with
    # reflection padding + Tukey taper.
    # ------------------------------------------------------------------
    pad_y = int(round(Ny_d * spatial_pad_frac))
    pad_x = int(round(Nx_d * spatial_pad_frac))
    Ny_p = Ny_d + 2*pad_y
    Nx_p = Nx_d + 2*pad_x

    x_p = (np.arange(Nx_p) - Nx_p/2.0) * dx_ds
    y_p = (np.arange(Ny_p) - Ny_p/2.0) * dx_ds

    spatial_W_padded = _make_2d_window(Ny_p, Nx_p, alpha=spatial_alpha)
    spatial_W = spatial_W_padded[pad_y:pad_y+Ny_d, pad_x:pad_x+Nx_d]

    if short_wave:
        eta_short = np.zeros_like(sx_ds, dtype=float)
        if verbose:
            print(f"    integrating slope -> eta_short per frame "
                  f"(padded {Ny_p}x{Nx_p}) ...")
        for ti in range(T):
            sx_p = np.pad(sx_ds[ti], ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
            sy_p = np.pad(sy_ds[ti], ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
            sx_pw = sx_p * spatial_W_padded
            sy_pw = sy_p * spatial_W_padded
            # CRITICAL: g2s mutates its inputs; pass copies.
            eta_p = g2s(x_p.copy(), y_p.copy(), sx_pw.copy(), sy_pw.copy())
            eta_c = eta_p[pad_y:pad_y+Ny_d, pad_x:pad_x+Nx_d]
            eta_short[ti] = eta_c * spatial_W
            eta_short[ti] -= eta_short[ti].mean()
    else:
        eta_short = None
        if verbose:
            print(f"    short_wave=False: skipping per-frame g2s integration "
                  f"(eta_short and eta_xyt are None).")

    # ------------------------------------------------------------------
    # eta_long(t): temporal CWT of spatial-mean slope, Krogstad signed
    # projection per (f, t), dispersion-relation integration, inverse-CWT.
    # Skipped entirely when long_wave=False (e.g. records too short to
    # resolve any long wave): eta_long is then zeros and no CWT is run.
    # ------------------------------------------------------------------
    temporal_W = _make_temporal_window(T, temporal_window, temporal_alpha)

    if long_wave:
        if verbose:
            print(f"    computing eta_long(t) from spatial-mean slopes ...")
        sx_mean = _aperture_spatial_mean(sx_ds, aperture_mask)
        sy_mean = _aperture_spatial_mean(sy_ds, aperture_mask)

        # Detrend so the temporal window doesn't multiply a constant DC offset
        sx_mean = sx_mean - sx_mean.mean()
        sy_mean = sy_mean - sy_mean.mean()

        sx_mean_w = sx_mean * temporal_W
        sy_mean_w = sy_mean * temporal_W

        Wsx = _cwt(sx_mean_w, freqs_cwt, fs, mother).values
        Wsy = _cwt(sy_mean_w, freqs_cwt, fs, mother).values

        _, k_disp = lindisp_with_current(2*np.pi*freqs_cwt, water_depth_m, 0.0)

        # Krogstad signed projection of the slope CWT coefficients onto the
        # elevation coefficients, via the dispersion-relation wavenumber.
        # See eta_field_recon.wavelet_core.krogstad_eta_coeffs for the full
        # derivation and the sign-guard rationale.
        skirt_gain = None
        if skirt_correct:
            skirt_gain = skirt_correction(
                freqs_cwt, fs, k_disp, T, mother,
                per_scale=inverse_per_scale, temporal_alpha=temporal_alpha)
        W_eta, cos_th, sin_th = krogstad_eta_coeffs(
            Wsx, Wsy, k_disp, skirt_gain=skirt_gain)

        eta_long = _inverse_cwt(W_eta, freqs_cwt, fs, mother,
                                per_scale=inverse_per_scale)
    else:
        if verbose:
            print(f"    long_wave=False: skipping eta_long(t) inversion "
                  f"(eta_long set to zero).")
        # Still expose the raw spatial-mean slopes for inspection; everything
        # downstream of the CWT is left as None so callers can tell the long
        # path was not run.
        sx_mean = _aperture_spatial_mean(sx_ds, aperture_mask)
        sy_mean = _aperture_spatial_mean(sy_ds, aperture_mask)
        sx_mean = sx_mean - sx_mean.mean()
        sy_mean = sy_mean - sy_mean.mean()
        sx_mean_w = sy_mean_w = None
        Wsx = Wsy = W_eta = cos_th = sin_th = k_disp = None
        eta_long = np.zeros(T, dtype=float)

    # ------------------------------------------------------------------
    # Combine and confidence mask
    # ------------------------------------------------------------------
    if short_wave:
        eta_xyt = eta_short + eta_long[:, None, None]
    else:
        # No resolved short-wave field; the full 2-D elevation is not formed.
        eta_xyt = None
    confidence = (spatial_W[None, :, :] * temporal_W[:, None, None]).astype(float)

    diag = dict(
        long_wave=long_wave,
        sx_mean=sx_mean, sy_mean=sy_mean,
        sx_mean_w=sx_mean_w, sy_mean_w=sy_mean_w,
        Wsx=Wsx, Wsy=Wsy, W_eta=W_eta,
        cos_th=cos_th, sin_th=sin_th, k_disp=k_disp,
        x_ds=x_ds, y_ds=y_ds, dx_ds=dx_ds,
        spatial_W=spatial_W, spatial_W_padded=spatial_W_padded,
        temporal_W=temporal_W,
        aperture_mask=aperture_mask, aperture_diameter_m=aperture_diameter_m,
        pad_y=pad_y, pad_x=pad_x,
    )
    return eta_xyt, eta_long, eta_short, confidence, diag
