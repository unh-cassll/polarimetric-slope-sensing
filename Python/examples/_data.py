"""
Example-data resolution for the pss / E-PSS package.

The canonical data live in a Zenodo archive (CC-BY-4.0):

    Polarized Light Intensity Reflected from Ocean Surface Waves
    Laxague, N.  DOI 10.5281/zenodo.20361229
    https://zenodo.org/records/20361229

Four files make up the record:

    asit_2019_raw_pol_frame0001.nc   first frame of the stack          (5.6 MB)
    asit_2019_raw_pol_median.nc      time-median of the 60 s stack      (4.9 MB)
    asit_2019_raw_pol_stack_3s.nc    first 3 s of the stack           (505.4 MB)
    asit_2019_raw_pol_stack.nc       full 60 s @ 30 fps                 (10.1 GB)

The data files are NOT committed to the repository; they are downloaded from
Zenodo on first use and cached in EXAMPLES_DIR, with md5 verification against
the checksums published in the record. (Anything dropped into EXAMPLES_DIR by
hand is used as-is if the checksum matches.)

What IS committed is a small derived artifact, asit2019_mean_slope_60s.nc,
holding the spatial-mean slope time series sx_mean(t), sy_mean(t) for the full
60 s record (1-D, a few KB). It is produced once by tools/precompute_mean_wave.py
and lets mean_wave_timeseries() reconstruct the long-wave elevation eta_long(t)
live, offline, without the 10 GB download. See mean_wave_timeseries() below.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

# Directory holding (cached) example NetCDF files and the committed artifact.
EXAMPLES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Zenodo archive.
# ---------------------------------------------------------------------------
# Versioned record id (10.5281/zenodo.20361229, v1). We deliberately pin the
# *version* record, not the concept record (20361228), so the bytes we fetch
# never change underneath us. Bump this when pinning a newer data version.
ZENODO_RECORD_ID = "20361229"
ZENODO_FILE_URL = (
    "https://zenodo.org/records/" + ZENODO_RECORD_ID + "/files/{name}?download=1"
)

# Archive filenames + md5 checksums (from the published record).
FRAME_FILENAME = "asit_2019_raw_pol_frame0001.nc"
MEDIAN_FILENAME = "asit_2019_raw_pol_median.nc"
STACK_3S_FILENAME = "asit_2019_raw_pol_stack_3s.nc"
STACK_FULL_FILENAME = "asit_2019_raw_pol_stack.nc"

_MD5 = {
    FRAME_FILENAME: "93630b51e5f029f63ddf161858e8ee79",
    MEDIAN_FILENAME: "50a1e40594d92f2e20d1cdb4725e0e55",
    STACK_3S_FILENAME: "00dad245c07a9fd1fce105f3b88ce12f",
    STACK_FULL_FILENAME: "9974f2b354f7517652d003cb5aef13fa",
}

# Committed derived artifact (produced by tools/precompute_mean_wave.py).
MEAN_SLOPE_FILENAME = "asit2019_mean_slope_60s.nc"

# Committed independent-validation artifact: Riegl LD90-3 water-surface
# elevation (produced by tools/make_lidar_elevation_nc.py).
LIDAR_ELEVATION_FILENAME = "asit2019_lidar_elevation_10min.nc"


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_from_zenodo(name: str, dest: Path) -> None:
    """Fetch `name` from the Zenodo record into `dest`, verifying md5.

    Streams to a temporary file, checks the md5 against the published
    checksum, and only then moves it into place, so an interrupted download
    never leaves a corrupt file at `dest`.
    """
    if name not in _MD5:
        raise KeyError(
            f"{name!r} is not a known archive file; expected one of "
            f"{sorted(_MD5)}."
        )
    url = ZENODO_FILE_URL.format(name=name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {name} from Zenodo (record {ZENODO_RECORD_ID}) ...")
    urllib.request.urlretrieve(url, tmp)

    got = _md5_of(tmp)
    if got != _MD5[name]:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"md5 mismatch for {name}: expected {_MD5[name]}, got {got}. "
            f"The download may be corrupt; try again."
        )
    tmp.replace(dest)
    print(f"  saved + verified -> {dest}")


def resolve(name: str, *, allow_download: bool = True) -> Path:
    """Return a local path to archive file `name`, downloading if needed.

    If the file is already in EXAMPLES_DIR it is used as-is (and, if a
    checksum is known, verified). Otherwise it is fetched from Zenodo when
    `allow_download` is True; if False, a missing file raises instead.
    """
    path = EXAMPLES_DIR / name
    if path.exists():
        if name in _MD5 and _md5_of(path) != _MD5[name]:
            raise ValueError(
                f"{name} exists in {EXAMPLES_DIR} but its md5 does not match "
                f"the archive. Delete it to re-download, or restore the "
                f"correct file."
            )
        return path
    if allow_download:
        _download_from_zenodo(name, path)
        return path
    raise FileNotFoundError(
        f"{name!r} not found in {EXAMPLES_DIR} and downloads are disabled."
    )


def frame_path(*, allow_download: bool = True) -> Path:
    """Local path to the single-frame example NetCDF (Zenodo)."""
    return resolve(FRAME_FILENAME, allow_download=allow_download)


def median_path(*, allow_download: bool = True) -> Path:
    """Local path to the temporal-median example NetCDF (gain reference)."""
    return resolve(MEDIAN_FILENAME, allow_download=allow_download)


def stack_3s_path(*, allow_download: bool = True) -> Path:
    """Local path to the 3-second stack (505 MB; downloaded on first use)."""
    return resolve(STACK_3S_FILENAME, allow_download=allow_download)


def stack_full_path(*, allow_download: bool = True) -> Path:
    """Local path to the full 60-second stack (10.1 GB; downloaded on use)."""
    return resolve(STACK_FULL_FILENAME, allow_download=allow_download)


def mean_slope_path() -> Path:
    """Local path to the committed spatial-mean slope artifact.

    This is a small derived file, committed to the repository (not on Zenodo).
    If it is missing, it must be regenerated with tools/precompute_mean_wave.py.
    """
    path = EXAMPLES_DIR / MEAN_SLOPE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{MEAN_SLOPE_FILENAME} not found in {EXAMPLES_DIR}. It is the "
            f"committed spatial-mean slope series; regenerate it once with:\n"
            f"    python tools/precompute_mean_wave.py --input <60s_stack.nc>\n"
            f"(see that script's header for details)."
        )
    return path


def lidar_elevation_path() -> Path:
    """Local path to the committed Riegl LD90-3 elevation artifact.

    A small committed file (not on Zenodo). If missing, regenerate it with
    tools/make_lidar_elevation_nc.py from the raw elevation array.
    """
    path = EXAMPLES_DIR / LIDAR_ELEVATION_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{LIDAR_ELEVATION_FILENAME} not found in {EXAMPLES_DIR}. It is "
            f"the committed Riegl LD90-3 elevation reference; build it once "
            f"with:\n"
            f"    python tools/make_lidar_elevation_nc.py --input elev_m.npy\n"
            f"(see that script's header for details)."
        )
    return path


def lidar_elevation():
    """Independent water-surface elevation ground truth (Riegl LD90-3).

    Loads the committed 10-minute elevation series used to validate the PSS
    long-wave reconstruction.

    Returns:
        (t, elev) : time vector (s, from acquisition start) and elevation (m),
        both 1-D. NOTE: the lidar and the PSS stack start together but view
        spatially offset points, so a propagation lag between this series and
        eta_long(t) is expected -- cross-correlate to measure it rather than
        assuming sample alignment (see the file's timing_note attribute).
    """
    import numpy as np
    from netCDF4 import Dataset

    path = lidar_elevation_path()
    with Dataset(str(path)) as ds:
        elev = np.asarray(ds.variables["elev_m"][...], dtype=float)
        t = np.asarray(ds.variables["time"][...], dtype=float)
    return t, elev


def mean_wave_timeseries(water_depth_m: float | None = None, verbose: bool = True):
    """Mean-wave (long-wave) elevation time series eta_long(t).

    Loads the committed spatial-mean slope series (sx_mean(t), sy_mean(t),
    orthorectified) from asit2019_mean_slope_60s.nc and runs the long-wave
    inversion live -- the same CWT -> Krogstad signed direction -> dispersion
    projection -> inverse-CWT used inside reconstruct_eta_field, here exercised
    on the real 60 s ASIT record without needing the 10 GB stack.

    Returns:
        (t, eta_long) : time vector (s) and mean-wave elevation (m), both 1-D.

    Requires the committed artifact to exist (see mean_slope_path()).
    """
    import numpy as np
    from netCDF4 import Dataset

    from eta_field_recon.wavelet_core import (
        lindisp_with_current, _cwt, _inverse_cwt,
    )
    from eta_field_recon.recon import _make_temporal_window

    path = mean_slope_path()
    with Dataset(str(path)) as ds:
        sx_mean = np.asarray(ds.variables["sx_mean"][...], dtype=float)
        sy_mean = np.asarray(ds.variables["sy_mean"][...], dtype=float)
        fs = float(np.asarray(ds.variables["fs"][...]))
        depth = (water_depth_m if water_depth_m is not None
                 else float(np.asarray(ds.variables["water_depth"][...])))

    T = sx_mean.size
    t = np.arange(T) / fs
    if verbose:
        print(f"mean_wave_timeseries: {T} samples @ {fs:g} Hz "
              f"({T/fs:.1f} s), depth {depth} m")

    freqs = np.linspace(0.05, 2.0, 80)   # same default band as reconstruct_eta_field
    sx = sx_mean - sx_mean.mean()
    sy = sy_mean - sy_mean.mean()
    win = _make_temporal_window(T, "tukey", 0.25)
    Wsx = _cwt(sx * win, freqs, fs, None).values
    Wsy = _cwt(sy * win, freqs, fs, None).values
    _, k = lindisp_with_current(2 * np.pi * freqs, depth, 0.0)

    eps = 1e-30
    mag = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + eps
    rel_sign = np.sign(np.real(Wsy * np.conj(Wsx)))
    rel_sign = np.where(rel_sign == 0, 1.0, rel_sign)   # indeterminate phase -> +1
    cos_th = np.abs(Wsx) / mag
    sin_th = (np.abs(Wsy) / mag) * rel_sign
    W_eta = 1j * (cos_th * Wsx + sin_th * Wsy) / k[:, None]
    W_eta = np.where(np.isfinite(W_eta), W_eta, 0.0)
    eta_long = _inverse_cwt(W_eta, freqs, fs, None)

    return t, eta_long
