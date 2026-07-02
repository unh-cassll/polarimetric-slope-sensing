# eta_field_recon

Reconstruct water surface elevation η(x, y, t) from a stack of 2-D
slope images.

Designed for tight slope-imager arrays such as polarimetric or stereo
systems: e.g. 512×512 measurements at 6 mm pixels (~3 m frame), 10 Hz
frame rate.  Recovers waves with frequencies from ~0.1 Hz (long swell,
wavelength much larger than the frame) to >1 Hz (short wind sea,
wavelength shorter than the frame) in a single pipeline.

## Files

- `wavelet_core.py` — `lindisp_with_current`, `_cwt`, `_inverse_cwt`.
  Self-contained core utilities, no other project dependencies.
- `recon.py` — `reconstruct_eta_field(...)`.  The core reconstruction
  (per-frame Harker-O'Leary short-wave path + CWT long-wave path).  Takes
  in-memory slope stacks.
- `eta_pipeline.py` — `reconstruct_eta_from_record(...)`.  Field-data driver:
  reads a NetCDF record, reduces each frame with `pss`, optionally
  orthorectifies, and reconstructs η.  Carries the record-length gate that
  enables/skips the long-wave inversion.
- `orthorectify.py` — `orthorectify_static(...)`.  Projects an obliquely-
  viewed slope field onto a uniform ground grid (static-platform geometry)
  and returns the true ground `dx`.
- `demo_eta_field.py` — self-contained runnable demo that synthesizes a
  realistic wave field, runs the reconstruction, and produces a
  diagnostic plot.

See also the top-level `epss.run_epss(...)` (one level up), the in-memory
front door that runs the whole raw-frames → slopes → η chain.

## Quick use

```python
from eta_field_recon import reconstruct_eta_field

# slope_x_field, slope_y_field: (T, Ny, Nx) float arrays, units of radians
# dx: pixel size in meters
# fs: frame rate in Hz

eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
    slope_x_field, slope_y_field,
    dx=0.006, fs=10.0,
    water_depth_m=100.0,
    downsample=8,         # 512 -> 64 output
)
```

Returns:

- `eta_xyt`   `(T, Ny_d, Nx_d)` reconstructed elevation field (m), or
              `None` when `short_wave=False`.
- `eta_long`  `(T,)` long-wave (spatial-mean) time series (m).
- `eta_short` `(T, Ny_d, Nx_d)` zero-mean-per-frame short-wave field (m),
              or `None` when `short_wave=False`.
- `conf`      `(T, Ny_d, Nx_d)` on [0, 1]: spatial window × temporal
              window.  Use for masking/weighting downstream.
- `diag`      dict of intermediates (disc-mean slope series, window
              vectors, aperture mask, coordinate vectors, pad sizes).

## Method

Two paths, summed:

1. **Short wave path**: each frame's slope `(sx, sy)` is integrated
   spatially via Harker-O'Leary least squares (`pyGrad2Surf.g2s`),
   producing `η_short(x, y, t)` with zero spatial mean per frame.
   The frame is reflection-padded and Tukey-tapered before integration,
   then cropped and tapered again.

2. **Long wave path**: the disc-mean slopes `(<sx>(t), <sy>(t))` are
   windowed in time and projected onto a Krogstad signed-direction
   estimate — per rfft frequency (`fourier_slope_projection`, the
   default) or per CWT `(f, t)` cell (`wavelet_slope_projection`) — with
   the amplitude imposed from the direct slope spectrum via the
   dispersion relation, giving `η_long(t)`.

The full field is `η(x, y, t) = η_short(x, y, t) + η_long(t)`.

## Why this works

For frequencies whose wavelength is much smaller than the frame
(`f > sqrt(g / (2π·L))` ≈ 0.7 Hz at L = 3 m), the in-frame spatial
gradient carries enough information to recover wave shape: H&O
integration is the right tool.  Such waves have effectively zero
spatial mean over the frame, so the long path doesn't see them.

For frequencies whose wavelength is much larger than the frame
(swell at 0.1–0.5 Hz), the field looks nearly uniform across the
frame; H&O integration loses the DC.  But the spatial-mean slope
preserves the long-wave temporal signature, and the dispersion relation
plus Krogstad direction lets us invert it to get the elevation time
series.

The two paths are orthogonal: short has zero spatial mean, long has no
spatial structure inside the frame.  No frequency-domain blending is
needed.

## Performance

On a 256×256 input at downsample=4 (64×64 output), 1024 frames:
- Reconstruction: ~10 s total.
- Output array size: ~33 MB.

## Dependencies

- `numpy`, `scipy`, `xarray` (standard)
- `ewdm` (Empirical Wave Directional Method): `pip install ewdm`
- `pyGrad2Surf`: `pip install pyGrad2Surf`

## Known gotchas

- **`pyGrad2Surf.g2s` modifies its inputs in place** (in its
  `_g2sSylvester`).  `recon.py` passes `.copy()` of the slopes;
  do the same if you call `g2s` elsewhere.

- The wavelet inverse uses an empirical calibration factor of 1.4383
  (≈ √2) to compensate for an under-normalization in the EWDM CWT
  round-trip.  This brings the amplitude to within ~1–3% across signal
  types and frequency grids; documented in `wavelet_core.py`.

- The Krogstad slope-only direction estimator has a 180° ambiguity in
  general, but for the spatial-mean slope here the sign of
  `Re(W_sy · conj(W_sx))` resolves it correctly.

## Tuning windowing

Defaults: spatial Tukey α=0.1 with 10% reflection padding, temporal
Tukey α=0.25.  Empirically the temporal window provides almost all of
the benefit when there's any slow drift in the spatial-mean slope.  The
spatial window is exposed for noisy-edge scenarios but rarely improves
on rectangular for clean data, because g2s is already robust.

To disable: `spatial_alpha=0, spatial_pad_frac=0, temporal_window='rect'`.
