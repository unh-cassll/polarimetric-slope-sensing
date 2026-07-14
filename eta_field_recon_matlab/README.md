# eta_field_recon_matlab

MATLAB port of the `eta_field_recon` Python package: reconstruction of the
water surface elevation field eta(x, y, t) from a time series of 2-D slope
images.

```matlab
addpath('.../eta_field_recon_matlab');

% slope_x_field, slope_y_field: (Ny, Nx, T) double, units rad
[eta_xyt, eta_long, eta_short, confidence, diagn] = reconstruct_eta_field( ...
    slope_x_field, slope_y_field, 0.006, 10.0, ...
    water_depth_m=15.0, downsample=8, ...
    grad2surf_dir='/path/to/grad2Surf', dopbox_dir='/path/to/DOPbox');
```

The short-wave step requires local copies of the Harker & O'Leary Grad2Surf
and DOPbox MATLAB packages (MATLAB File Exchange: "Surface Reconstruction
from Gradient Fields: grad2Surf" and "Discrete Orthogonal Polynomial
Toolbox: DOPbox"; also at harkeroleary.org). Point at them with the
`grad2surf_dir`/`dopbox_dir` options as above, or `addpath` both packages
yourself and omit the options.

## Scope

Ported (numerically parity-checked against the Python source):

- `reconstruct_eta_field` — main entry point: per-frame Harker-O'Leary g2s
  integration (short-wave shape) + directionally-complete Fourier slope
  projection of the disc-mean slope (long-wave eta_long(t));
  eta = eta_short + eta_long.
- `fourier_slope_projection`, `invert_mean_slope_series` (fourier method),
  `long_wave_gate`
- `lindisp_with_current`, `aperture_transfer_function`, `aperture_transfer_gain`

Not ported:

- `wavelet_slope_projection` and the `long_wave_method='wavelet'` phase
  source: these sit on the EWDM Morlet CWT and its per-scale calibration.
  The fourier method (the Python default) is directionally identical and
  differs only in phase source; requesting 'wavelet' raises an error
  pointing back to the Python implementation.
- `orthorectify` / `eta_pipeline` (camera-geometry and driver layers).

## Differences from the Python package

- **Array convention**: slope stacks and field outputs are (Ny, Nx, T) with
  frames along dim 3 (MATLAB-native), not the Python (T, Ny, Nx). For arrays
  saved from numpy via scipy.io.savemat, convert with `permute(a, [2 3 1])`.
  Time series (`eta_long`, `sx_mean`, ...) are (T, 1) columns.
- **g2s is external**: the short-wave step calls the original Grad2Surf/DOPbox
  MATLAB packages (which pyGrad2Surf transcribed); supply their locations via
  `grad2surf_dir`/`dopbox_dir` or put them on the path yourself. `lyap`
  (Control System Toolbox) is required by `g2sSylvester`.
- `aperture_diameter_m=[]` replaces Python `None` (full-frame mean).
- Padding size uses MATLAB `round` (half away from zero) where Python uses
  banker's rounding; they differ only when `N_d * spatial_pad_frac` lands
  exactly on .5.

## Parity check

`parity/` holds an end-to-end demonstration that both implementations produce
the same elevation arrays from the same input slope fields:

```bash
# 1. synthesize slope fields, run the Python reconstruction, save both:
python parity/make_parity_reference.py /path/to/parity_case.mat
# 2. run the MATLAB port on the same input and compare:
matlab -batch "cd eta_field_recon_matlab/parity; run_parity_check('/path/to/parity_case.mat', '/path/to/grad2Surf', '/path/to/DOPbox')"
```

The MATLAB checker reports max |difference| and the difference relative to
the field's standard deviation for `eta_xyt`, `eta_short`, `eta_long`, and
`confidence`, writes a comparison figure, and fails (nonzero exit) if any
relative difference exceeds 1e-6. Residual differences at the 1e-10 level
and below reflect solver implementations (Sylvester/lyap, spline
interpolation, least-squares), not algorithmic divergence.
