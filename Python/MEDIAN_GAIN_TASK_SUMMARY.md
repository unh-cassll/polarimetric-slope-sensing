# Task summary — median-derived empirical gain + NetCDF-only migration

Two pieces of work, in order:

## A. Single-frame reduction with a median-derived empirical gain

Reduce the individual frame `asit2019_frame0001.nc`, computing the E-PSS
empirical DoLP gain from the separate temporal-median frame
`asit2019_median.nc` rather than from the frame being corrected.

| quantity                          | value         |
|-----------------------------------|---------------|
| empirical gain g                  | **1.5593**    |
| DoLP median of the median frame   | 0.2826        |
| Fresnel-ideal DoLP @ 30°          | 0.4406        |
| frame0001 median DoLP (post-gain) | 0.3289        |
| frame0001 median AOI              | 26.023°       |
| mss                               | 68.2639 deg²  |

The gain is `0.4406 / 0.2826 = 1.5593`, derived once from the stable median
background. frame0001's post-gain median DoLP is **0.3289, not the ideal
0.4406** — and that's the point: median calibration lets each frame keep its
real DoLP departure from the background (the wave signal) instead of being
self-normalized onto the ideal. For contrast, the self-referenced gain gives
g = 2.0890 and an inflated mss of 112.4112 deg².

### Code changes

- **`pss/gain.py` — `apply_gain()`** gained three prioritized ways to source
  the observed-DoLP median: `dolp_obs_median` (scalar) > `s1_ref`/`s2_ref`
  (reference frame) > `s1`/`s2` (self, the original default). `gain_notes`
  records the source (`ref=self` / `ref=external frame` / `ref=precomputed`).
- **`pss/slope.py` — `compute_slope_field()`** gained `gain_reference_frame=`.
- **`pss/io.py` — `read_netcdf_frame()`** handles the stack schema
  (`raw_frame` dims `(time, x, y)`, `superpixel_layout` dims
  `(super_col, super_row)`): selects a frame with `time_index=` and reorients
  to `(y, x)` from the declared dimension names. Legacy 2-D `(y, x)` untouched.
- **`examples/load_and_reduce_with_median_gain.py`** — dedicated E-PSS demo;
  console script `pss-load-reduce-median`.
- **Tests:** 4 new unit tests in `test_gain.py` for the reference precedence;
  5 regression tests in `test_median_gain.py` pinning the numbers above.

## B. TIFF removal — NetCDF-only inputs

Per the decision that all data now comes as NetCDF (with longer records from
a TBD Zenodo archive), the entire TIFF path was removed.

- **Deleted:** `demo.py`, `examples/load_and_reduce_tiff.py`,
  `tools/write_netcdf_from_tiff.py`, `tools/NETCDF_SCHEMA_HANDOFF.md` (and the
  now-empty `tools/`), the committed `.tif` / `.cdl`, and the old generated
  `.nc`.
- **Dependencies:** dropped `tifffile` and `imageio` (the `images` extra)
  from `pyproject.toml` and `requirements.txt`.
- **Console scripts:** removed `pss-demo`, `pss-write-nc`,
  `pss-load-reduce-tiff`; the remaining three are `pss-load-reduce`,
  `pss-load-reduce-median`, `pss-eta-demo`.
- **`examples/load_and_reduce.py`** rewritten as NetCDF-only (defaults to the
  bundled frame, optional `--median`, `--time-index`); both example scripts
  run with no arguments.
- **`examples/_data.py`** added: centralizes the example filenames and holds
  `ZENODO_URL = None` plus a `resolve()` that returns the committed local
  file now and will fetch from Zenodo once the DOI/URL is assigned.
- **Tests:** `conftest.py` and `test_pipeline.py` repointed at the committed
  `asit2019_frame0001.nc` (self-gain numbers 2.0890 / 0.4406 / 30.000 /
  112.4112); the two TIFF round-trip tests were removed.
- **`.gitignore`:** the two example `.nc` files are now committed (un-ignored
  via `!`-exceptions); any *other* `examples/*.nc` stays out of git.
- **Docs:** `README.md` and `SESSION_HANDOFF.md` rewritten for the NetCDF-only
  workflow; the handoff has a new §11 migration note.

## State

`pytest` reports **68 passed**. Quick verification:

```bash
cd pss_py
pip install -e ".[test]" --break-system-packages   # PEP-668 managed env
pytest -q                                           # 68 passed

python examples/load_and_reduce.py --no-show                 # gain 2.0890, mss 112.4112
python examples/load_and_reduce_with_median_gain.py --no-show # gain 1.5593, mss 68.2639
```
