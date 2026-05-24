"""Shared pytest fixtures: synthetic DoFP frame and example data.

Data-dependent fixtures resolve files through examples/_data.py with
downloads DISABLED, so the test suite never hits the network: a file that
has been fetched/cached locally is used, otherwise the dependent test skips.
This keeps the suite green offline (e.g. in CI without the data) while still
running the full numerical regressions when the data is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pss.fresnel import fresnel_dolp
from pss.stokes import _OFFSETS


def _data_module():
    """Import examples/_data.py regardless of how pytest is invoked."""
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "examples"))
    import _data  # noqa: E402
    return _data


def _resolve_local_or_skip(getter_name: str) -> Path:
    """Resolve an example file locally (no download); skip the test if absent."""
    _data = _data_module()
    getter = getattr(_data, getter_name)
    try:
        return getter(allow_download=False)
    except FileNotFoundError:
        pytest.skip(
            f"example data not present locally (via _data.{getter_name}); "
            f"fetch it once or run with the data cached. Skipping offline."
        )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def example_nc_path() -> Path:
    """The canonical single-frame example NetCDF (Zenodo; cached locally).

    Resolved without downloading; the dependent test skips if the file has
    not been fetched into examples/ yet.
    """
    return _resolve_local_or_skip("frame_path")


@pytest.fixture(scope="session")
def frame_stack_nc_path() -> Path:
    """The single-frame stack NetCDF (stack schema: raw_frame (time, x, y)).

    Same file as example_nc_path; resolved local-only, skips if absent.
    """
    return _resolve_local_or_skip("frame_path")


@pytest.fixture(scope="session")
def median_nc_path() -> Path:
    """The per-pixel temporal-median frame (empirical-gain reference).

    Resolved local-only; skips if absent.
    """
    return _resolve_local_or_skip("median_path")


@pytest.fixture(scope="session")
def synthetic_frame() -> tuple[np.ndarray, dict]:
    """A small DoFP frame synthesized from a known wave-slope field.

    Returns (frame, truth) where `truth` carries the ground-truth scalars
    used to build the frame (theta_i_mean_deg, n_water, etc.). The DoLP
    field is set by Fresnel theory and modulated by a sinusoidal slope
    pattern, then sampled at each super-pixel orientation per Malus's law.
    """
    rng = np.random.default_rng(42)
    H, W = 256, 256

    # Smooth 2D wave-slope field
    xx, yy = np.meshgrid(np.linspace(-3, 3, W), np.linspace(-3, 3, H))
    Sx_true = 0.05 * np.sin(2.0 * xx) * np.cos(1.3 * yy)
    Sy_true = 0.05 * np.cos(1.7 * xx) * np.sin(0.9 * yy)
    slope_mag = np.sqrt(Sx_true**2 + Sy_true**2)
    theta_local_rad = np.arctan(slope_mag)

    THETA_MEAN_DEG = 30.0
    N_WATER = 1.34
    theta_total_deg = np.degrees(theta_local_rad) + THETA_MEAN_DEG

    phi_rad = np.arctan2(Sx_true, Sy_true)
    DoLP_true = np.clip(fresnel_dolp(theta_total_deg, n_water=N_WATER), 0, 1)
    s1_true = DoLP_true * np.cos(2 * phi_rad)
    s2_true = DoLP_true * np.sin(2 * phi_rad)
    S0_true = 1000.0 + 50.0 * yy

    def I_at(alpha_deg, S0, dolp, phi):
        a = np.deg2rad(alpha_deg)
        return 0.5 * S0 * (1.0 + dolp * np.cos(2 * (a - phi)))

    frame = np.zeros((H, W), dtype=np.float64)
    angle_to_offset = {
        "I0":   0.0,
        "I45":  45.0,
        "I90":  90.0,
        "I135": 135.0,
    }
    for name, (r_off, c_off) in _OFFSETS.items():
        I_full = I_at(angle_to_offset[name], S0_true, DoLP_true, phi_rad)
        frame[r_off::2, c_off::2] = I_full[r_off::2, c_off::2]
    frame += rng.normal(0, 0.5, size=frame.shape)

    truth = {
        "theta_i_mean_deg": THETA_MEAN_DEG,
        "n_water":          N_WATER,
        "Sx_true":          Sx_true,
        "Sy_true":          Sy_true,
        "DoLP_true":        DoLP_true,
        "shape":            (H, W),
    }
    return frame, truth
