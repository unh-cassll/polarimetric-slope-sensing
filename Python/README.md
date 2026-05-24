# pss + eta_field_recon — Polarimetric Slope Sensing and η(x, y, t) Reconstruction

Two sibling Python packages for ocean surface-wave remote sensing,
implementing the techniques described in the two-paper E-PSS series by
Laxague et al. (2026):

- **`pss`** — Polarimetric Slope Sensing with E-PSS empirical gain.
  Python port of the MATLAB demo driver from
  [`unh-cassll/polarimetric-slope-sensing`](https://github.com/unh-cassll/polarimetric-slope-sensing).
  Given a single raw frame from a division-of-focal-plane (DoFP) polarimetric
  camera, recovers Stokes parameters, DoLP, polarization orientation, angle of
  incidence at each pixel, and along-look/cross-look surface slopes.
  *Reference: Laxague et al. (2026), IEEE TGRS.*

- **`eta_field_recon`** — Surface-elevation η(x, y, t) reconstruction.
  Given a time-series stack of 2-D slope fields (the output of `pss`,
  or of any other slope-imaging modality), recovers the elevation field
  over the entire imager footprint by combining per-frame Harker-O'Leary
  spatial integration (short-wave shape) with a wavelet-based temporal
  inversion of the spatial-mean slope (long-wave time series).
  *Reference: Laxague et al. (2026), IEEE Journal of Oceanic Engineering.*

The two packages share a repository and a distribution (one
`pip install pss` brings both in), but are imported independently and
operate on different scales: `pss` is per-frame, `eta_field_recon` is
per-record. The natural workflow is

```
frame stack -> pss.compute_slope_field (per frame) -> slope_x_field, slope_y_field
            -> eta_field_recon.reconstruct_eta_field (over the record) -> eta(x,y,t)
```

Each can also be used standalone.

## Top-level API — `run_epss` and `reconstruct_eta_from_record`

Two convenience entry points tie the whole chain together so you don't have
to wire `pss` and `eta_field_recon` by hand.

**`epss.run_epss`** — the in-memory front door. Hand it an array of raw DoFP
frames and it reduces every frame to a slope field; additionally hand it the
acquisition geometry and it also orthorectifies and reconstructs η(x, y, t):

```python
import numpy as np
from epss import run_epss

# frames: (T, H, W) raw DoFP stack (or a single (H, W) frame)

# Frames only -> slope fields, nothing else:
res = run_epss(frames)
res.slope_x, res.slope_y          # (T, Ny, Nx) slope stack (rad)

# Frames + ALL FIVE acquisition params -> ortho + η reconstruction:
res = run_epss(
    frames,
    fs=30.0,                      # frame rate (Hz)
    theta_i_mean_deg=30.0,        # mean incidence angle (deg)
    freeboard_m=23.0,             # camera height above the surface (m)
    pixel_pitch_m=3.45e-6,        # sensor pixel pitch (m)
    focal_length_m=0.075,         # lens focal length (m)
    water_depth_m=15.0,
    downsample=8,
)
res.eta_xyt, res.eta_long, res.eta_short, res.confidence   # elevation fields
res.ortho.dx_m                    # the true ground dx, derived from the optics
```

The five acquisition parameters are **all-or-nothing**: supply all five to
run the η stage, none to stop at the slope fields. A partial set raises a
clear error. The long-wave (mean-wave) inversion is itself gated on record
length (a too-short record returns `eta_long = 0`); see below.

**`eta_field_recon.reconstruct_eta_from_record`** — the same chain, but
starting from a NetCDF record on disk rather than an in-memory array. It
reads the acquisition geometry from the file, so with `orthorectify=True`
the ground `dx` is derived automatically:

```python
from eta_field_recon import reconstruct_eta_from_record

res = reconstruct_eta_from_record(
    "my_record.nc",
    orthorectify=True,                       # derive dx from the file's optics
    gain_reference_path="my_median.nc",      # empirical-gain reference
    downsample=8,
)
res.eta_xyt, res.eta_long, res.long_wave_ran, res.ortho.dx_m
```

### Static orthorectification and the platform-motion caveat

Orthorectification here is the **static-platform** case only: it projects the
obliquely-viewed slope field onto a uniform ground grid using fixed geometry
(freeboard, θᵢ, focal length, pixel pitch), correcting the oblique-view
trapezoid and yielding the one true ground `dx` the η inversion needs. It
assumes the camera pose is constant over the record. The moving-platform case
(per-frame attitude-driven rectification) needs a per-frame motion source
(IMU/INS) and is **not** implemented; see the TODO in `eta_pipeline.py`.

Note that on a static platform the camera's constant viewing tilt is already
removed by `pss` (which subtracts the per-frame spatial mean of the slopes),
so orthorectification is purely the geometric pixel resample — it does not
re-rotate the slope vectors.

### Record-length gate on the long-wave inversion

The long-wave path (the mean-wave elevation time series `eta_long(t)`)
requires the record to span enough time to resolve the lowest wave frequency.
The gate is physics-based: the record must cover at least `min_periods`
periods of the lowest CWT frequency (default 0.5 periods of 0.05 Hz, i.e.
~10 s). Shorter records skip the long-wave inversion entirely (`eta_long = 0`,
`eta_xyt = eta_short`) and save the CWT cost. Override with `force_long_wave`.

## pss — Quick start

```bash
git clone <repo-url> pss && cd pss
pip install -e .
python examples/load_and_reduce.py            # reduces the bundled example frame
```

…produces a 2×2 panel showing the gain-corrected Stokes components and the
along-look / cross-look slope fields, reproducing the canonical example:

![Example output: gain-corrected Stokes parameters and slope fields from a
2056×2464 ASIT2019 frame](examples/example_output.jpg)

The bundled example reduces a single raw frame (`asit2019_frame0001.nc`)
using the E-PSS workflow: the empirical gain is calibrated against the
temporal-median background frame (`asit2019_median.nc`), never against the
frame itself. Run it with the dedicated script:

```bash
python examples/load_and_reduce_with_median_gain.py
```

Reduction summary (printed to stdout, default native reduction,
median-referenced empirical gain):

    resolution      : native (half-res, one Stokes vector per super-pixel)
    gain mode       : empirical (g = 1.5587, ref = temporal-median frame)
    median DoLP     : 0.3309
    median AOI      : 26.099 deg
    mss             : 0.022729  (dimensionless, var(Sx) + var(Sy))

The empirical gain is calibrated against a **temporal average**, not the
individual frame. Its `median(DoLP) → DoLP_ideal(θᵢ)` step assumes the
reference's median surface tilt equals the mean viewing angle — i.e. a flat
surface *on average*. That holds for a temporal median (waves average out to
flat) but is false for any single instantaneous frame, so the frame keeps its
true DoLP departure from the background instead of being forced onto the
Fresnel ideal. If no reference is supplied, **no gain is applied** (the
correction is never self-referenced).

By default the reduction is **native** resolution: one Stokes vector per 2×2
super-pixel, returned at half the frame size (e.g. 1028×1232 from a
2056×2464 sensor). This is honest with respect to information content — the
output grid equals the measurement grid, with no interpolation implying
detail the sensor never resolved. Pass `resolution="full"` to interpolate
back to the full frame size with a chosen `method` (see below).

The s1 and s2 panels show the gain-corrected normalized Stokes components;
the diverging-colormap panels show the orthogonal decomposition of surface
tilt into the camera's along-look (toward the wave-coming-from direction
in this geometry) and cross-look axes. Individual wave faces, crests, and
gravity-capillary structure down to ~1 mm scale are clearly resolved.

For a step-by-step visual walkthrough of the whole chain — raw frame →
orientation sub-images → Stokes parameters → DoLP/orientation → angle of
incidence → slope fields, plus a Fresnel-curve visualization of the empirical
gain — open the notebook **[`PSS_walkthrough.ipynb`](PSS_walkthrough.ipynb)**
(all figures grayscale).

## eta_field_recon — Quick start

```bash
python eta_field_recon/demo_eta_field.py
```

…synthesizes a 4-component wave field (long swell + wind sea + capillary
waves) at 256×256 × 1024 frames, runs the reconstruction, and produces a
diagnostic plot. Takes ~2 minutes (most of it is synthesis).

![Example output: synthetic eta truth vs reconstruction, plus PSD, time
series, and confidence overlays](eta_field_recon/demo_eta_field.jpg)

Demo reconstruction quality on this 4-mode synthetic field:

    centre time series RMSE       : 0.040 m (truth std = 0.18 m)
    centre time series correlation: +0.975
    confidence-weighted RMSE      : 0.028 m
    confidence-weighted correlation: +0.988
    full-field RMSE               : 0.045 m
    full-field correlation        : +0.963

See `eta_field_recon/README.md` for the quick-start API, and
`eta_field_recon/HANDOFF.md` for method derivation, gotchas, and ideas
for extensions.

## Input data (for the pss layer)

`pss` reads raw DoFP frames from annotated NetCDF-4 files (CF-1.10 +
ACDD-1.3). Each file carries the acquisition geometry (θᵢ, n_water,
super-pixel layout, frame rate, etc.) as variables/attributes, so a single
file fully specifies how to reduce itself — `read_netcdf_frame` reads those
back and `compute_slope_field` consumes them. The reader handles both a
plain 2-D `(y, x)` frame and the stack schema (`raw_frame` with a leading
`time` dimension), selecting a frame with `time_index=` and reorienting as
needed.

The example data live in a Zenodo archive (CC-BY-4.0, DOI
[10.5281/zenodo.20361229](https://doi.org/10.5281/zenodo.20361229)) and are
**downloaded on first use**, not committed to the repository. `examples/_data.py`
resolves each file locally if cached, else fetches it from Zenodo and verifies
its md5 against the published checksum. The record holds four files:

| file                              | what it is                                  | size    |
|-----------------------------------|---------------------------------------------|---------|
| `asit_2019_raw_pol_frame0001.nc`  | first frame of the record                   | 5.6 MB  |
| `asit_2019_raw_pol_median.nc`     | temporal-median frame (empirical-gain ref)  | 4.9 MB  |
| `asit_2019_raw_pol_stack_3s.nc`   | first 3 s of the record                     | 505 MB  |
| `asit_2019_raw_pol_stack.nc`      | full 60 s @ 30 fps                          | 10.1 GB |

```python
from examples import _data
frame  = _data.frame_path()       # downloads + caches on first call
median = _data.median_path()
stack3 = _data.stack_3s_path()    # 505 MB
full   = _data.stack_full_path()  # 10.1 GB
```

The one data file that **is** committed is a small derived artifact,
`examples/asit2019_mean_slope_60s.nc` (a few KB): the spatial-mean slope time
series `sx_mean(t)`, `sy_mean(t)` for the full 60 s record, produced once by
`tools/precompute_mean_wave.py`. It lets `_data.mean_wave_timeseries()`
reconstruct the mean-wave elevation `eta_long(t)` live and offline, without the
10 GB download:

```python
t, eta_long = _data.mean_wave_timeseries()   # offline; uses the committed series
```

Because the raw data are not committed, the test suite resolves them
**local-only** (no network) and **skips** the data-dependent tests when the
files have not been cached — so the suite stays green offline (e.g. in CI),
and runs the full numerical regressions once the data are present.

## Repository layout

```
pss/
├── epss.py                               run_epss() top-level entry point
├── pss/                                  Polarimetric Slope Sensing package
│   ├── __init__.py
│   ├── stokes.py                         3 Stokes methods (Ratliff M4 default)
│   ├── gain.py                           none / lab / empirical DoLP gain
│   ├── fresnel.py                        DoLP <-> θᵢ lookup (Fresnel inversion)
│   ├── slope.py                          end-to-end pipeline
│   ├── io.py                             NetCDF reader (2-D and time-stack schemas)
│   └── _cli.py                           console-script bridge
├── eta_field_recon/                      η(x, y, t) reconstruction package
│   ├── __init__.py
│   ├── recon.py                          reconstruct_eta_field() (long_wave switch)
│   ├── eta_pipeline.py                   reconstruct_eta_from_record() field-data driver
│   ├── orthorectify.py                   orthorectify_static() ground-grid projection
│   ├── wavelet_core.py                   CWT + Fresnel-corrected dispersion
│   ├── demo_eta_field.py                 synthetic-field validation demo
│   ├── demo_eta_field.jpg                rendered demo output
│   ├── README.md                         package-specific quick reference
│   └── HANDOFF.md                        full method derivation and dev notes
├── examples/
│   ├── load_and_reduce.py                pss demo (single frame; optional --median)
│   ├── load_and_reduce_with_median_gain.py  E-PSS median-referenced gain demo
│   ├── _data.py                          Zenodo resolver (download + md5) + mean_wave_timeseries()
│   ├── asit2019_mean_slope_60s.nc        committed: spatial-mean slope series (few KB)
│   └── example_output.jpg                rendered output of the pss demo
├── tools/
│   └── precompute_mean_wave.py           one-off: build the committed mean-slope series
├── tests/                                pytest suite
│   ├── conftest.py
│   ├── test_stokes.py
│   ├── test_gain.py
│   ├── test_fresnel.py
│   ├── test_pipeline.py                  synthetic + NetCDF end-to-end
│   ├── test_median_gain.py               stack reader + median-referenced gain
│   └── test_eta_field_recon.py
├── smoke_test.py                         synthetic-frame validation
├── PSS_walkthrough.ipynb                 visual walkthrough notebook (grayscale)
├── pyproject.toml
├── requirements.txt                      (kept for non-pip workflows)
├── LICENSE                               GPL-3.0-or-later
└── README.md
```

The raw NetCDF files are **not** committed — they download from Zenodo on
first use and cache in `examples/` (md5-verified), so `examples/*.nc` is
`.gitignore`'d. The one exception, force-included in git, is the small
committed artifact `asit2019_mean_slope_60s.nc` (the spatial-mean slope
series; see the Input-data section).

## Install

Python 3.10+. The Python package lives in the `Python/` subfolder of the
repository.

**Straight from GitHub (no clone needed):**

```bash
pip install "git+https://github.com/unh-cassll/polarimetric-slope-sensing.git#subdirectory=Python"
```

The `#subdirectory=Python` fragment tells pip the installable project is in
that subfolder rather than the repo root (the repo root is the MATLAB
project). The import name is `pss` even though the distribution is named
`epss`:

```python
import pss
from pss import compute_slope_field
```

**For development (editable), after cloning:**

```bash
git clone https://github.com/unh-cassll/polarimetric-slope-sensing.git
cd polarimetric-slope-sensing/Python
pip install -e .                       # editable install + core dependencies
pip install -e ".[test]"               # add pytest + xarray for the test suite
pip install -e ".[all]"                # everything
```

(On a PEP-668 "externally managed" Python, add `--break-system-packages` to
the `pip install` lines, or use a virtual environment.)

Core dependencies are `numpy`, `scipy`, `matplotlib`, and `netCDF4`. If you
prefer not to install the package, dependencies can be pulled from
`requirements.txt` and the scripts can be run in-place
(`python examples/load_and_reduce.py ...`).

After `pip install -e .`, three console scripts are available on `$PATH`:

| command                   | source                                      |
|---------------------------|---------------------------------------------|
| `pss-load-reduce`         | `examples/load_and_reduce.py`               |
| `pss-load-reduce-median`  | `examples/load_and_reduce_with_median_gain.py` |
| `pss-eta-demo`            | `eta_field_recon/demo_eta_field.py`         |

## Running the tests

```bash
pip install -e ".[test]"
pytest                                  # 68 tests, runs in ~30 seconds
```

The test suite covers all three Stokes methods, all three gain modes
(including the median-referenced empirical gain), the Fresnel-inversion
lookup, the NetCDF reader (both 2-D and time-stack schemas), exact
reproduction of the documented `pss` reduction numbers, and (for
`eta_field_recon`) the dispersion-relation accuracy, shape/return-type
contracts, the orthogonal-decomposition invariant, and single-mode
synthetic-field correlation with truth.

## The three gain modes

The MATLAB original applies hard-coded lab-calibrated scalars (S1×1.2185,
S2×1.2197) to correct for the difference between the polarimeter's measured
DoLP and the ideal Fresnel response. `pss` replaces those with a
`gain_mode` argument:

| `gain_mode`   | Behavior                                                                       |
|---------------|--------------------------------------------------------------------------------|
| `"none"`      | No correction — raw polarimeter response                                       |
| `"lab"`       | Fixed lab-calibrated gain (defaults to the MATLAB values; overridable)         |
| `"empirical"` | Scales a temporal-median reference DoLP to the Fresnel ideal at θᵢ (no reference → no gain) |

The empirical gain is computed as

    g = DoLP_ideal(θᵢ) / median(DoLP_reference)

where `DoLP_ideal(θᵢ)` is the Fresnel prediction for unpolarized sky and no
upwelling radiance at the camera's mean angle of incidence. Both normalized
Stokes components are scaled by `g`. Field values reported in the E-PSS
paper for ASIT2019 are in the range 1.2–1.7; the implementation clips at
`[0.5, 3.0]` by default to refuse obviously bogus values.

`median(DoLP_reference)` is **always** taken from a temporal average, never
from the frame being corrected. The `median(DoLP) → DoLP_ideal(θᵢ)` step
assumes the reference's median surface tilt equals the mean viewing angle (a
flat surface on average) — true for a temporal median, false for a single
instantaneous frame. So self-referencing is not supported. Supply the
reference one of two ways:

- `gain_reference_frame=` to `compute_slope_field` — a per-pixel
  temporal-median background frame (the canonical E-PSS workflow); or
- `dolp_obs_median=` (a precomputed reference DoLP scalar) to `apply_gain`,
  for the single-frame case where a temporal median can't be formed.

If both are given, the reference frame wins. **If neither is supplied, no
gain is applied** (the result is downgraded to `gain_mode="none"` with an
explanatory note) — the gain is never silently self-referenced.

## Library use

The simplest end-to-end call reads a frame and its metadata from NetCDF, so
the reduction parameters come straight from the file:

```python
from pss import read_netcdf_frame, apply_layout_from_meta, compute_slope_field

frame, meta = read_netcdf_frame("examples/asit2019_frame0001.nc")
apply_layout_from_meta(meta)   # honor the file's super-pixel layout

result = compute_slope_field(
    frame,
    # resolution defaults to "native" (half-res, honest sampling). Pass
    # resolution="full" to interpolate to the full frame with `method`.
    gain_mode=meta.gain_mode,      # "none" / "lab" / "empirical"
    theta_i_mean_deg=meta.theta_i_mean_deg,
    n_water=meta.n_water,
)

print(result.mss, result.gain_notes)
# result.s1, .s2, .dolp, .aoi_deg, .Sx, .Sy, .Ax_deg, .Ay_deg are 2-D arrays
# (at native resolution these are H/2 x W/2).
```

For a full-resolution interpolated reduction, choose a `method`:

```python
result = compute_slope_field(
    frame, resolution="full",
    method="bilinear",   # or "kernel_averaging" / "conv_demodulation" (Ratliff M4)
    gain_mode="empirical",
    theta_i_mean_deg=meta.theta_i_mean_deg, n_water=meta.n_water,
)
```

For the E-PSS workflow, derive the empirical gain from the temporal-median
background frame and apply it to an individual frame:

```python
frame,  meta = read_netcdf_frame("examples/asit2019_frame0001.nc")
median, _    = read_netcdf_frame("examples/asit2019_median.nc")
apply_layout_from_meta(meta)

result = compute_slope_field(
    frame,
    # resolution defaults to "native"; add resolution="full", method=... to interpolate.
    gain_mode="empirical",
    theta_i_mean_deg=meta.theta_i_mean_deg,
    n_water=meta.n_water,
    gain_reference_frame=median,   # gain calibrated against the median, not the frame
)
```

## CLI

**`examples/load_and_reduce.py`** — reduces a single frame. Reads metadata
from the file and runs with the file's suggested defaults; CLI flags
override any metadata-supplied parameter. With no arguments it reduces the
bundled example frame.

```bash
# Bundled example frame (no --median -> no empirical gain applied)
python examples/load_and_reduce.py

# A specific file, picking a frame along the time axis
python examples/load_and_reduce.py my_record.nc --time-index 12

# Calibrate the empirical gain against a median frame; save the figure
python examples/load_and_reduce.py --median examples/asit2019_median.nc \
    --save out.png --no-show

# Different Stokes method / lab gain
python examples/load_and_reduce.py --method kernel_averaging --gain lab
```

**`examples/load_and_reduce_with_median_gain.py`** — dedicated E-PSS demo:
reduce a frame with the empirical gain calibrated against a temporal-median
background frame. With no arguments it uses the two bundled example files.

```bash
python examples/load_and_reduce_with_median_gain.py
python examples/load_and_reduce_with_median_gain.py my_frame.nc \
    --median my_median.nc --save out.png --no-show
```

## NetCDF schema (archival)

`pss` reads CF-1.10 + ACDD-1.3 NetCDF-4 files. The reader expects:

- a `raw_frame` variable — either 2-D `(y, x)` or a stack `(time, x, y)`;
  the reader selects a frame with `time_index=` and reorients to `(y, x)`
  from the declared dimension names;
- a `superpixel_layout` 2×2 array giving the angle→position mapping, so the
  super-pixel arrangement is unambiguous (read back via the metadata and
  applied with `apply_layout_from_meta`);
- scalar geometry variables (θᵢ, n_water, lens, pixel pitch, azimuth,
  freeboard, frame rate, exposure, position, wind);
- CF discovery attributes (title, keywords, creator, license, geospatial
  bounds) and `pss_processing_*` hints that tell `read_netcdf_frame` how to
  reduce the frame.

These files (and longer time series) are produced by the field group's
acquisition pipeline and distributed via a Zenodo archive (DOI TBD). The
bundled `asit2019_frame0001.nc` reduces (default native resolution,
median-referenced empirical gain against `asit2019_median.nc`) to
`g = 1.5587`, `median DoLP = 0.3309`, `median AOI = 26.099°`,
`mss = 0.022729` (dimensionless).

## Super-pixel layout — why it matters and how to verify

A DoFP sensor tiles four micropolarizers (at 0°, 45°, 90°, 135°) into each
2×2 super-pixel. The exact assignment of angle to (row, column) position
is sensor-specific and matters: getting it wrong rotates the inferred
polarization orientation φ by a multiple of 45°, and a 180° rotation
inverts the slope sign in a way that adds super-pixel-scale checkerboard
noise (visible as a ~10× jump in `mss`).

The default in `pss.stokes._OFFSETS` is:

```
(row=0, col=0) -> 90°
(row=0, col=1) -> 45°
(row=1, col=0) -> 135°
(row=1, col=1) ->  0°
```

This is the **L0** layout used by the Sony IMX250MZR. To verify on a new
sensor:

1. Reduce a frame with all four 90°-rotated layouts.
2. Plot the DoLP-weighted histogram of φ for each.
3. The 180° rotation produces dramatically higher `mss`; rule it out.
4. The other three differ by exactly 45° shifts in the φ peak.
5. The correct layout is the one whose φ peak aligns with what you expect
   for the wave field — typically the camera's horizontal or vertical
   axis, depending on mounting.

The NetCDF file stores the verified layout in the `superpixel_layout`
variable (as a 2×2 array of angles in degrees), and `read_netcdf_frame`
returns it in the metadata so downstream code can patch the package
default if needed (`apply_layout_from_meta`).

## eta_field_recon — elevation reconstruction from a slope-image time series

The `eta_field_recon` package lifts a stack of 2-D slope images to the
elevation field η(x, y, t). The natural input is the output of `pss`
applied frame-by-frame to a NetCDF frame stack, but any slope-imaging
modality works (stereo, refractive shape-from-stereo, etc.) so long as you
can hand it `(slope_x, slope_y)` per frame.

### Library use

```python
from eta_field_recon import reconstruct_eta_field

# slope_x_field, slope_y_field: (T, Ny, Nx) float arrays, slope in radians
eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
    slope_x_field, slope_y_field,
    dx=0.006,              # pixel size in metres
    fs=10.0,               # frame rate in Hz
    water_depth_m=15.0,    # for the dispersion relation
    downsample=8,          # output is (T, Ny/8, Nx/8)
)
```

Returns:
- `eta_xyt` — `(T, Ny_d, Nx_d)` elevation field (m)
- `eta_long` — `(T,)` long-wave (spatial-mean) time series (m)
- `eta_short` — `(T, Ny_d, Nx_d)` zero-mean-per-frame short-wave field (m)
- `conf` — `(T, Ny_d, Nx_d)` confidence mask on `[0, 1]`
- `diag` — dict of intermediates (CWT coefficients, windows, etc.)

### Method, briefly

Two paths, summed:

1. **Short-wave path** (per-frame Harker-O'Leary least-squares spatial
   integration via `pyGrad2Surf.g2s`). Recovers the wave SHAPE inside the
   frame; sets the spatial mean to zero per frame by construction.

2. **Long-wave path** (continuous-wavelet transform of the spatial-mean
   slope, Krogstad signed-direction estimator per `(f, t)`, dispersion-
   relation projection, inverse CWT). Recovers the elevation time series
   the spatial integration loses.

The two paths are orthogonal — short has zero spatial mean, long has no
spatial structure inside the frame — and sum cleanly. For a frame of
size L, the crossover frequency where wavelength = L is
`f_crossover = √(g / 2πL) ≈ 0.7 Hz at L = 3 m`. Above that, the short
path dominates; below it, the long path dominates.

See `eta_field_recon/README.md` for the API reference and
`eta_field_recon/HANDOFF.md` for the full method derivation, decisions,
known gotchas, and ideas for extensions (anti-aliasing, MEM/MLM
directional estimator, current correction, streaming for long records).

### Directional spectrum

The full directional wave spectrum F(f, θ) derives from `eta_xyt`
together with the slope fields, but lives in a separate downstream repo
(not yet linked). The `eta_field_recon` outputs are designed to feed
straight into it.

## What was ported from the MATLAB

The 94-line MATLAB driver (`sample_slope_field_calculations.m`) does:

1. Load a DoLP→θᵢ lookup table (`dolp_theta_vecs.mat`), keep the rising
   Fresnel branch, resample on a uniform DoLP grid with PCHIP.
2. Read a raw DoFP frame from disk.
3. Loop over three Stokes-reconstruction methods.
4. Multiply S1 by 1.2185 and S2 by 1.2197 (hard-coded lab gains).
5. Compute DoLP, orientation, AOI, slopes, and a 2×2 plot.

The Python version reproduces all of that and replaces step 4 with the
configurable `gain_mode`. The lookup table (step 1) is generated from
first-principles Fresnel theory by default
(`pss.fresnel.build_lookup_table`); the original `.mat` file is also
loadable via `pss.fresnel.load_lookup_table` (uses `scipy.io.loadmat`).

## A few implementation choices worth knowing

- **Output resolution (`resolution=`).** The default is `"native"`: the four
  orientation samples in each 2×2 super-pixel are combined directly into one
  Stokes vector, so the result is half the frame size (H/2 × W/2) with no
  interpolation. This is honest about information content — there is genuinely
  one polarization measurement per super-pixel, and the output grid says so.
  The effective pixel pitch is then **twice** the sensor pitch, which matters
  for any wavelength/spectrum computation (pass the doubled `dx`). Passing
  `resolution="full"` interpolates each orientation back to the full (H, W)
  grid using `method`; this implies 2× the real linear resolution but
  partially corrects the IFOV half-pixel offset between orientations.

- **Bilinear interpolation method** uses
  `scipy.ndimage.map_coordinates(order=1)` so the half-pixel offset between
  each orientation's sample grid and the full frame is handled correctly.

- **Conv-demodulation method (`conv_demodulation`) is the package default**
  and implements the exact **Ratliff, LaCasse & Tyo (2009) Method 4** — the
  16-pixel interpolator from Fig. 3 of that paper (radius 3√2/2, four pixels
  per orientation, distance weights A = 0.3541, B = 0.2639, C = 0.1180). The
  four 4×4 kernels are convolved with the frame and combined per Eq. (12);
  the per-parity kernel→orientation mapping is derived from the configured
  super-pixel layout (`_OFFSETS`) rather than hard-coded, so it tracks the
  layout. Earlier releases used a separable Bartlett kernel here as a
  stand-in; that has been replaced by the exact Method 4 filter. It takes no
  kernel-size argument. (`kernel_averaging` still accepts `"2x2"`/`"4x4"`.)

- **`mss` definition.** `mss` is the **dimensionless** mean-square slope,
  `var(Sx) + var(Sy)`, where `Sx`, `Sy` are the surface slopes (tangents of
  the tilt). This is the standard air-sea quantity. Note this differs from
  the MATLAB driver, which computed `var(atand(Ax))` — the variance of the
  tilt *angle* in deg² (and via a double-`atand`). To recover the tilt-angle
  variance instead, use `var(result.Ax_deg) + var(result.Ay_deg)`; see the
  comment in `pss/slope.py`.

- **Empirical-gain safety clamp.** The gain is clipped at `[0.5, 3.0]` by
  default; override via the `clip_gain` argument of `pss.gain.apply_gain`.

## Verifying against MATLAB output

For a like-for-like comparison with the original MATLAB, reduce a frame
with the fixed lab gain (the MATLAB original's behavior):

```bash
python examples/load_and_reduce.py my_frame.nc --method bilinear --gain lab
```

…produces panels visually indistinguishable from MATLAB's figure 1 output,
with `mss` values within a few percent (small differences are expected
from the bilinear half-pixel handling and, under the default, the Ratliff
Method 4 reconstruction).

## Citation

If you use `pss` or `eta_field_recon` in published work, please cite the
relevant paper(s) in the E-PSS series:

> Laxague, N. J. M., Z. G. Duvarcı, L. Hogan, J. Liu, C. Bouillon, and
> C. J. Zappa (2026). *E-PSS: the Extended Polarimetric Slope Sensing
> technique for measuring ocean surface waves.* IEEE Transactions on
> Geoscience and Remote Sensing.
> *(use this for the polarimetric / slope-field part of the pipeline)*

> Laxague, N. J. M., et al. (2026). *Reconstruction of the surface
> elevation field η(x, y, t) from slope-imager data using a long-short
> wavelet decomposition.* IEEE Journal of Oceanic Engineering. (in
> preparation / forthcoming)
> *(use this for the elevation-reconstruction part of the pipeline)*

…and reference the MATLAB original at
[`unh-cassll/polarimetric-slope-sensing`](https://github.com/unh-cassll/polarimetric-slope-sensing).

## License

GPL-3.0-or-later. See `LICENSE`. This Python implementation is released
under the same license as the MATLAB original
([`unh-cassll/polarimetric-slope-sensing`](https://github.com/unh-cassll/polarimetric-slope-sensing)),
which is GPL-3.0.
