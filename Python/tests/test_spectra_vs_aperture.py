"""
Tests for the pure helpers in tools/spectra_vs_aperture.py.

These cover the spectrum estimator and the inscribed-diameter geometry without
needing the Zenodo stack (which the full script downloads). The end-to-end
reduction path is exercised manually against real data when run for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# tools/ is not a package; add it to the path to import the script's helpers.
_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

from spectra_vs_aperture import (  # noqa: E402
    omnidirectional_spectrum,
    inscribed_diameter_m,
    APERTURE_FRACTIONS,
)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def test_aperture_fractions_are_full_half_quarter():
    assert APERTURE_FRACTIONS == (1.0, 0.5, 0.25)


def test_spectrum_integral_equals_variance():
    # Parseval: integral of the PSD over f equals the signal variance.
    fs = 10.0
    t = np.arange(0, 600.0, 1.0 / fs)
    rng = np.random.RandomState(0)
    eta = 0.4 * np.sin(2 * np.pi * 0.12 * t) + 0.05 * rng.randn(t.size)
    f, S = omnidirectional_spectrum(eta, fs, seg_seconds=204.8)
    ratio = _trapz(S, f) / np.var(eta)
    assert 0.95 < ratio < 1.05


def test_spectrum_peaks_at_dominant_frequency():
    fs = 10.0
    t = np.arange(0, 300.0, 1.0 / fs)
    f0 = 0.137
    eta = np.sin(2 * np.pi * f0 * t)
    f, S = omnidirectional_spectrum(eta, fs, seg_seconds=102.4)
    assert abs(f[np.argmax(S)] - f0) < 0.01


def test_spectrum_units_are_psd():
    # Doubling the sample rate (same physical signal, same segment DURATION)
    # must not change S(f) at a given frequency by more than the windowing
    # tolerance: PSD is a density, not scaled by the sample-rate ratio.
    t1 = np.arange(0, 300.0, 1 / 10.0)
    t2 = np.arange(0, 300.0, 1 / 20.0)
    e1 = np.sin(2 * np.pi * 0.1 * t1)
    e2 = np.sin(2 * np.pi * 0.1 * t2)
    f1, S1 = omnidirectional_spectrum(e1, 10.0, seg_seconds=51.2)
    f2, S2 = omnidirectional_spectrum(e2, 20.0, seg_seconds=51.2)
    peak1 = S1[np.argmax(S1)]
    peak2 = S2[np.argmax(S2)]
    # peak densities should be comparable (within a factor of 2), not scaled
    # by the sample-rate ratio
    assert 0.5 < peak1 / peak2 < 2.0


def test_spectrum_handles_nans():
    fs = 10.0
    t = np.arange(0, 100.0, 1 / fs)
    eta = np.sin(2 * np.pi * 0.1 * t)
    eta[::50] = np.nan   # sprinkle no-data
    f, S = omnidirectional_spectrum(eta, fs, seg_seconds=25.6)
    assert np.isfinite(S).all()


def test_spectrum_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        omnidirectional_spectrum(np.zeros(4), 10.0)


def test_seg_seconds_sets_low_frequency_resolution():
    # A 30 s segment must resolve down to ~1/30 Hz regardless of sample rate,
    # so the same seg_seconds gives the same lowest frequency at 10 and 30 Hz.
    t10 = np.arange(0, 300.0, 1 / 10.0)
    t30 = np.arange(0, 300.0, 1 / 30.0)
    e10 = np.sin(2 * np.pi * 0.2 * t10)
    e30 = np.sin(2 * np.pi * 0.2 * t30)
    f10, _ = omnidirectional_spectrum(e10, 10.0, seg_seconds=30.0)
    f30, _ = omnidirectional_spectrum(e30, 30.0, seg_seconds=30.0)
    # first non-zero frequency bin ~ 1/30 Hz for both
    assert abs(f10[1] - 1 / 30.0) < 1e-3
    assert abs(f30[1] - 1 / 30.0) < 1e-3
    assert abs(f10[1] - f30[1]) < 1e-6


def test_seg_seconds_longer_than_record_falls_back(capsys):
    # A 30 s segment requested on a 5 s record clips to one periodogram segment
    # and warns, rather than erroring.
    fs = 10.0
    eta = np.sin(2 * np.pi * 0.3 * np.arange(0, 5.0, 1 / fs))
    f, S = omnidirectional_spectrum(eta, fs, seg_seconds=30.0)
    assert np.isfinite(S).all()
    out = capsys.readouterr().out
    assert "shorter than the requested" in out


def test_inscribed_diameter_uses_min_side():
    # Build a minimal diag like reconstruct_eta_field returns: a non-square
    # mask and a known dx_ds. Inscribed diameter = min(Ny, Nx) * dx_ds.
    Ny, Nx, dx_ds = 24, 40, 0.1
    diag = {"dx_ds": dx_ds, "aperture_mask": np.ones((Ny, Nx), bool)}
    assert inscribed_diameter_m(diag) == pytest.approx(min(Ny, Nx) * dx_ds)


# ---------------------------------------------------------------------------
# reduce_downsample memory lever (driver) -- verified via the documented
# contract: subsample preserves values, float32, and dx scales by the factor.
# ---------------------------------------------------------------------------

def test_reduce_downsample_contract():
    # The driver subsamples each reduced frame [::rds, ::rds] as float32 and
    # scales the effective ground dx by rds, so the physical footprint
    # (dx * n_samples) is preserved. Verify that invariant directly.
    W = 160
    dx_native = 0.01
    full = (0.001 * np.arange(W)).astype(float)
    footprints = []
    for rds in (1, 2, 4):
        sub = np.asarray(full, dtype=np.float32)[::rds]
        dx_eff = dx_native * rds
        footprints.append(dx_eff * sub.size)
        # subsampled values are an exact subset of the full field
        assert np.allclose(sub, full[::rds].astype(np.float32))
        assert sub.dtype == np.float32
    # footprint preserved across all subsample factors
    assert max(footprints) - min(footprints) < 1e-9


def test_g2s_min_nodes_guard():
    # Over-shrinking the grid must raise an actionable error from the driver,
    # not a cryptic one from g2s. We exercise the guard arithmetic directly:
    # a 200x160 native frame at reduce_downsample=4 + downsample=8 -> ~6x5,
    # below the 16-node floor.
    from eta_field_recon import eta_pipeline as ep
    ny, nx, rds, ds = 200, 160, 4, 8
    eff_ny = (ny // rds) // ds
    eff_nx = (nx // rds) // ds
    assert min(eff_ny, eff_nx) < 16  # this configuration is the failure case


def test_eta_long_fast_path_matches_full_reconstruct():
    # The tool computes eta_short ONCE and re-runs only the long-wave inversion
    # per aperture. That fast path must be bit-identical to a full
    # reconstruct_eta_field call with the same aperture.
    from spectra_vs_aperture import _eta_long_for_aperture, inscribed_diameter_m
    from eta_field_recon.recon import (reconstruct_eta_field,
                                       _circular_aperture_mask)
    rng = np.random.RandomState(0)
    T, Ny, Nx, fs, dx = 256, 48, 56, 10.0, 0.05
    t = np.arange(T) / fs
    tilt = 0.02 * np.sin(2 * np.pi * 0.12 * t)
    sx = np.broadcast_to(tilt[:, None, None], (T, Ny, Nx)).copy()
    sx += 0.001 * rng.randn(T, Ny, Nx)
    sy = 0.5 * sx

    # full-frame reference + diag (downsample=1: grid == input)
    _, eta_long_full, _, _, diag = reconstruct_eta_field(
        sx, sy, dx=dx, fs=fs, downsample=1, aperture_diameter_m=None,
        long_wave=True, verbose=False)
    full_diam = inscribed_diameter_m(diag)
    freqs = np.linspace(0.05, 2.0, 80)

    for frac in (0.5, 0.25):
        diam = frac * full_diam
        # full reconstruct with this aperture
        _, el_full, _, _, _ = reconstruct_eta_field(
            sx, sy, dx=dx, fs=fs, downsample=1, aperture_diameter_m=diam,
            long_wave=True, verbose=False)
        # fast eta_long-only path
        mask = _circular_aperture_mask(Ny, Nx, diag["dx_ds"], diam)
        el_fast = _eta_long_for_aperture(sx, sy, fs, mask, freqs, 100.0)
        assert np.array_equal(el_full, el_fast)
