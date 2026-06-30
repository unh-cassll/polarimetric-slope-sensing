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
from scipy.signal import detrend
from scipy.signal.windows import tukey, hann
import warnings

from .wavelet_core import (_cwt, _inverse_cwt, lindisp_with_current,
                           aperture_transfer_gain)


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
    mask = r <= (diameter_m / 2.0)
    if not mask.any():
        # A diameter below the cell spacing can miss every cell center; an
        # empty mask would silently produce an all-NaN spatial mean.
        raise ValueError(
            f"aperture_diameter_m={diameter_m} selects no grid cells at "
            f"dx={dx} on a {Ny}x{Nx} grid; use a diameter >= the cell spacing")
    return mask


def long_wave_gate(record_duration_s, freqs_cwt=None, min_periods=0.5):
    """Physics gate for the long-wave inversion.

    The record must span at least `min_periods` periods of the lowest CWT
    frequency. Returns (enabled, threshold_s, f_min_hz); callers format their
    own messages. Single source of the gate logic shared by run_epss,
    run_epss_from_slopes, and reconstruct_eta_from_record.
    """
    f_min = float(np.min(freqs_cwt)) if freqs_cwt is not None else 0.05
    threshold_s = float(min_periods) / f_min
    return record_duration_s >= threshold_s, threshold_s, f_min


def invert_mean_slope_series(sx_mean, sy_mean, fs, water_depth_m=100.0,
                             long_wave_method='fourier', hp_fmin=0.08,
                             temporal_alpha=0.25):
    """Long-wave elevation eta_long(t) from a spatial-mean slope series.

    The single-point form of the long-wave path in reconstruct_eta_field: the
    directionally-complete direct amplitude sqrt(|Sx|^2+|Sy|^2)/k carries the
    magnitude (no aperture jinc -- the input is already a spatial mean), and the
    per-frequency signed projection ('fourier') or a Morlet CWT projection
    ('wavelet') carries the phase. Used by _data.mean_wave_timeseries.

    Args:
        sx_mean, sy_mean : (T,) cross-look / along-look spatial-mean slope.
        fs               : sample rate (Hz).
        water_depth_m    : depth for the dispersion relation.
        long_wave_method : 'fourier' (default) or 'wavelet'.
        hp_fmin          : logistic high-pass corner (Hz). Default 0.08.
        temporal_alpha   : Tukey taper alpha. Default 0.25.

    Returns:
        eta_long : (T,) elevation series (m), up-positive.
    """
    sx = np.asarray(sx_mean, dtype=float)
    sy = np.asarray(sy_mean, dtype=float)
    A, Sx, Sy, T = _direct_complete_amplitude(sx, sy, fs, water_depth_m, None,
                                              jinc=False, hp_fmin=hp_fmin,
                                              temporal_alpha=temporal_alpha)
    if long_wave_method == 'wavelet':
        fcwt = np.linspace(0.05, 2.0, 80)
        win = tukey(T, temporal_alpha)
        Wsx = _cwt(detrend(sx) * win, fcwt, fs).values
        Wsy = _cwt(detrend(sy) * win, fcwt, fs).values
        _, kc = lindisp_with_current(2 * np.pi * fcwt, water_depth_m, 0.0)
        kc = np.asarray(kc, float)
        m = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + 1e-30
        rel = np.sign(np.real(Wsy * np.conj(Wsx)))
        rel = np.where(rel == 0, 1.0, rel)
        with np.errstate(divide='ignore', invalid='ignore'):
            Weta = 1j * ((np.abs(Wsx) / m) * Wsx + (np.abs(Wsy) / m) * rel * Wsy) / kc[:, None]
        Weta = np.where(np.isfinite(Weta), Weta, 0.0)
        bp = 1.0 / (1.0 + np.exp(-(np.log2(fcwt) - np.log2(hp_fmin)) / 0.25))
        eta_krog = np.real(_inverse_cwt(Weta * bp[:, None], fcwt, fs, per_scale=True))
        phase = np.angle(np.fft.rfft(eta_krog - eta_krog.mean()))
    else:
        m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
        rel = np.sign(np.real(Sy * np.conj(Sx)))
        rel = np.where(rel == 0, 1.0, rel)
        phase = np.angle(1j * ((np.abs(Sx) / m) * Sx + (np.abs(Sy) / m) * rel * Sy))
    eta = np.fft.irfft(A * np.exp(1j * phase), n=T)
    return eta - eta.mean()


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


def _direct_complete_amplitude(sx_mean, sy_mean, fs, depth, diameter_m, jinc=True,
                               hp_fmin=0.08, hp_width_oct=0.25, temporal_alpha=0.25):
    """rfft-grid directionally-complete long-wave amplitude A(f)=sqrt(|Sx|^2+|Sy|^2)/k
    (Phillips 1977), jinc aperture-corrected and logistic high-passed. Returns
    (A, Sx, Sy, T); Sx, Sy are the windowed disc-mean slope rffts. Shared by the
    fourier and wavelet slope projections below."""
    sE = detrend(np.asarray(sx_mean, float))
    sN = detrend(np.asarray(sy_mean, float))
    T = sE.size
    win = tukey(T, temporal_alpha)
    wn = np.sqrt(np.mean(win ** 2))
    f = np.fft.rfftfreq(T, 1.0 / fs)
    _, k = lindisp_with_current(2 * np.pi * f, depth, 0.0)
    k = np.asarray(k, float)
    Sx = np.fft.rfft(sE * win) / wn
    Sy = np.fft.rfft(sN * win) / wn
    m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(np.isfinite(m / k), m / k, 0.0)
    if jinc and diameter_m is not None:
        with warnings.catch_warnings():                  # null bands expected
            warnings.simplefilter('ignore', UserWarning)
            g = aperture_transfer_gain(f, k, diameter_m, shape='circular', min_transfer=0.3)
        A = A * np.where(np.isfinite(g), g, 0.0)
    with np.errstate(divide='ignore'):
        lr = (np.log2(np.maximum(f, 1e-12)) - np.log2(hp_fmin)) / hp_width_oct
    A = A * np.clip(1.0 / (1.0 + np.exp(-lr)), 0.0, 1.0)
    return A, Sx, Sy, T


def _aperture_disc(slope_x_field, slope_y_field, dx, aperture_diameter_m):
    """Disc-mean slope series and the disc's physical diameter (full frame if None)."""
    Ny, Nx = slope_x_field.shape[1:]
    mask = _circular_aperture_mask(Ny, Nx, dx, aperture_diameter_m)
    sxm = _aperture_spatial_mean(slope_x_field, mask)
    sym = _aperture_spatial_mean(slope_y_field, mask)
    diam = (aperture_diameter_m if aperture_diameter_m is not None
            else float(np.sqrt(Ny * Nx)) * dx)
    return sxm, sym, diam


def fourier_slope_projection(slope_x_field, slope_y_field, dx, fs, depth,
                             aperture_diameter_m=None, jinc=True, hp_fmin=0.08,
                             hp_width_oct=0.25, temporal_alpha=0.25):
    """Long-wave eta(t) by per-frequency signed slope projection.

    Disc-mean slope rffts. Per frequency: direction cos=|Sx|/m, sin=(|Sy|/m)*
    sign(Re(Sy conj Sx)) (180-deg ambiguity from the channels' relative phase); the
    projection carries only the phase, the directionally-complete direct amplitude the
    magnitude: eta = irfft(A * exp(i*angle(+1j*(cos*Sx + sin*Sy)))). The Fourier-
    amplitude form of the directional estimator of Krogstad, Magnusson & Donelan
    (2006). slope_*_field are (T, Ny, Nx); aperture_diameter_m None = full frame."""
    sxm, sym, diam = _aperture_disc(slope_x_field, slope_y_field, dx, aperture_diameter_m)
    A, Sx, Sy, T = _direct_complete_amplitude(sxm, sym, fs, depth, diam, jinc,
                                              hp_fmin, hp_width_oct, temporal_alpha)
    m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
    rel = np.sign(np.real(Sy * np.conj(Sx)))
    rel = np.where(rel == 0, 1.0, rel)
    carrier = 1j * ((np.abs(Sx) / m) * Sx + (np.abs(Sy) / m) * rel * Sy)
    eta = np.fft.irfft(A * np.exp(1j * np.angle(carrier)), n=T)
    return eta - eta.mean()


def wavelet_slope_projection(slope_x_field, slope_y_field, dx, fs, depth,
                             aperture_diameter_m=None, jinc=True, hp_fmin=0.08,
                             hp_width_oct=0.25, temporal_alpha=0.25, mother=None):
    """Long-wave eta(t): wavelet (CWT) signed slope projection for the phase, with the
    same directionally-complete direct amplitude as fourier_slope_projection.

    Disc-mean slopes -> Morlet CWT. Per (f, t): cos=|Wsx|/m, sin=(|Wsy|/m)*
    sign(Re(Wsy conj Wsx)); Weta = +1j*(cos*Wsx + sin*Wsy)/k(f), logistic high-passed;
    eta_krog = Re(iCWT). The amplitude is then imposed from the direct slope spectrum so
    the wavelet carries only the phase. The directional estimator of Krogstad, Magnusson
    & Donelan (2006), reduced to the bare per-(f,t) projection (no skirt correction, no
    aperture blend). slope_*_field are (T, Ny, Nx); aperture_diameter_m None = full frame."""
    sxm, sym, diam = _aperture_disc(slope_x_field, slope_y_field, dx, aperture_diameter_m)
    A, _, _, T = _direct_complete_amplitude(sxm, sym, fs, depth, diam, jinc,
                                            hp_fmin, hp_width_oct, temporal_alpha)
    fcwt = np.linspace(0.05, 2.0, 80)
    win = tukey(T, temporal_alpha)
    Wsx = _cwt(detrend(sxm) * win, fcwt, fs, mother).values
    Wsy = _cwt(detrend(sym) * win, fcwt, fs, mother).values
    _, kc = lindisp_with_current(2 * np.pi * fcwt, depth, 0.0)
    kc = np.asarray(kc, float)
    m = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + 1e-30
    rel = np.sign(np.real(Wsy * np.conj(Wsx)))
    rel = np.where(rel == 0, 1.0, rel)
    with np.errstate(divide='ignore', invalid='ignore'):
        Weta = 1j * ((np.abs(Wsx) / m) * Wsx + (np.abs(Wsy) / m) * rel * Wsy) / kc[:, None]
    Weta = np.where(np.isfinite(Weta), Weta, 0.0)
    bp = 1.0 / (1.0 + np.exp(-(np.log2(fcwt) - np.log2(hp_fmin)) / hp_width_oct))
    eta_krog = np.real(_inverse_cwt(Weta * bp[:, None], fcwt, fs, mother, per_scale=True))
    phase = np.angle(np.fft.rfft(eta_krog - eta_krog.mean()))
    eta = np.fft.irfft(A * np.exp(1j * phase), n=T)
    return eta - eta.mean()


def reconstruct_eta_field(slope_x_field, slope_y_field, dx, fs,
                           water_depth_m=100.0,
                           downsample=4,
                           long_wave_method='fourier',
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
        water_depth_m : water depth (m) for the dispersion relation.
        downsample    : spatial subsample factor; output is (Ny/ds, Nx/ds).
                        Default 4 (e.g. 256->64).
        long_wave_method : 'fourier' (default, fourier_slope_projection) or
                        'wavelet' (wavelet_slope_projection) for the long-wave
                        eta(t); both share the directionally-complete amplitude
                        and differ only in the phase source.

        spatial_alpha    : Tukey alpha for the 2-D slope window.
                           Default 0.1 (light taper).
        spatial_pad_frac : reflection-padding fraction (per axis) before
                           integration.  Default 0.10.  Set 0 to disable.

        temporal_window : 'tukey', 'hann', or 'rect'.  Default 'tukey'.
        temporal_alpha  : Tukey alpha when kind='tukey'.  Default 0.25.

        long_wave     : if True (default) run the long-wave (spatial-mean
                        slope) projection and add eta_long(t) to the field.
                        If False, skip it entirely: eta_long is returned as
                        zeros and eta_xyt == eta_short. Useful for records too
                        short to resolve any long wave (see eta_pipeline.py,
                        which sets this from a physics-based record-length gate).

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
                        downsampled grid). A finite
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
    # eta_long(t): directionally-complete slope projection of the disc-mean
    # slopes (fourier or wavelet phase). Skipped (zeros) when long_wave=False.
    # ------------------------------------------------------------------
    temporal_W = _make_temporal_window(T, temporal_window, temporal_alpha)
    sx_mean = _aperture_spatial_mean(sx_ds, aperture_mask)
    sy_mean = _aperture_spatial_mean(sy_ds, aperture_mask)

    if long_wave:
        if verbose:
            print(f"    computing eta_long(t) from disc-mean slopes "
                  f"({long_wave_method}) ...")
        proj = (wavelet_slope_projection if long_wave_method == 'wavelet'
                else fourier_slope_projection)
        eta_long = proj(sx_ds, sy_ds, dx_ds, fs, water_depth_m,
                        aperture_diameter_m=aperture_diameter_m,
                        temporal_alpha=temporal_alpha)
    else:
        if verbose:
            print(f"    long_wave=False: skipping eta_long(t) (set to zero).")
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
        long_wave=long_wave, long_wave_method=long_wave_method,
        sx_mean=sx_mean, sy_mean=sy_mean,
        x_ds=x_ds, y_ds=y_ds, dx_ds=dx_ds,
        spatial_W=spatial_W, spatial_W_padded=spatial_W_padded,
        temporal_W=temporal_W,
        aperture_mask=aperture_mask, aperture_diameter_m=aperture_diameter_m,
        pad_y=pad_y, pad_x=pad_x,
    )
    return eta_xyt, eta_long, eta_short, confidence, diag
