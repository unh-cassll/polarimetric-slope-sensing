"""Wide->narrow AOI recovery on synthetic data.

Mirrors the Piermont dual-camera result (empirical wide LUT 2.2 deg vs ideal
Fresnel 10.3 deg) in a controlled setting: a common, sky-suppressed DoLP-AOI
relationship seen by both cameras. The wide camera's EMPIRICAL LUT recovers the
narrow imager's incidence; the sky-blind ideal Fresnel LUT is biased low because
the suppressed DoLP maps to a smaller incidence on the ideal curve.
"""

from __future__ import annotations

import numpy as np

from pss.fresnel import build_lookup_table, dolp_to_aoi, fresnel_dolp, lut_from_curve


# A real sky + turbid water suppresses the measured DoLP below ideal Fresnel.
SKY_SUPPRESSION = 0.6


def _observed_dolp(aoi_deg, n_water=1.34):
    return SKY_SUPPRESSION * fresnel_dolp(aoi_deg, n_water=n_water)


def test_empirical_wide_lut_beats_ideal_fresnel():
    # Wide camera: measures the suppressed DoLP across a broad AOI sweep.
    aoi_wide = np.linspace(3.0, 50.0, 300)
    dolp_wide = _observed_dolp(aoi_wide)
    lut_empirical = lut_from_curve(aoi_wide, dolp_wide)
    lut_fresnel = build_lookup_table(n_water=1.34)

    # Narrow camera: a handful of view angles, same sky (same suppression).
    aoi_true = np.array([20.0, 28.0, 35.0, 42.0, 48.0])
    dolp_narrow = _observed_dolp(aoi_true)

    rec_emp = dolp_to_aoi(dolp_narrow, *lut_empirical)
    rec_fre = dolp_to_aoi(dolp_narrow, *lut_fresnel)
    mae_emp = float(np.nanmean(np.abs(rec_emp - aoi_true)))
    mae_fre = float(np.nanmean(np.abs(rec_fre - aoi_true)))

    # Empirical recovers near-exactly; ideal Fresnel is biased and far worse.
    assert mae_emp < 1.0
    assert mae_emp < mae_fre
    assert mae_fre > 5.0


def test_empirical_lut_is_unbiased_in_same_physics_case():
    """When the wide camera sees the unsuppressed ideal curve, the empirical and
    ideal LUTs agree and both recover the true incidence."""
    aoi_wide = np.linspace(3.0, 50.0, 300)
    dolp_wide = fresnel_dolp(aoi_wide, n_water=1.34)
    lut_empirical = lut_from_curve(aoi_wide, dolp_wide)

    aoi_true = np.array([22.0, 31.0, 44.0])
    dolp_narrow = fresnel_dolp(aoi_true, n_water=1.34)
    rec = dolp_to_aoi(dolp_narrow, *lut_empirical)
    assert float(np.nanmean(np.abs(rec - aoi_true))) < 1.0


# ---------------------------------------------------------------------------
# Data-backed regression on the committed Piermont 2025 dual-camera artifacts.
# Skips offline if the small committed .nc files are not present.
# ---------------------------------------------------------------------------
import numpy as np

from pss import read_netcdf_frame, by_superpixel
from pss.widefov import (calibrate_widefov, strip_profile, rolling_nanmedian)

_LAB_GAIN = 1.0 / 0.81


def _narrow_points(stack_path):
    from netCDF4 import Dataset
    with Dataset(str(stack_path)) as ds:
        theta_i = np.asarray(ds.variables["theta_i_per_frame"][...], float)
    pts = []
    for t in range(theta_i.size):
        frame, _ = read_netcdf_frame(str(stack_path), time_index=t)
        _, s1, s2 = by_superpixel(frame)
        prof = rolling_nanmedian(
            strip_profile(np.sqrt(s1 ** 2 + s2 ** 2) * _LAB_GAIN), 9)
        pts.append((float(theta_i[t]),
                    float(np.nanmedian(prof[np.isfinite(prof)]))))
    return pts


def test_piermont_empirical_wide_beats_ideal_fresnel(
        piermont_wide_path, piermont_narrow_stack_path):
    from pss.fresnel import dolp_to_aoi
    wide, wm = read_netcdf_frame(str(piermont_wide_path))
    cal = calibrate_widefov(
        wide, focal_length_m=0.005, pixel_pitch_m=3.45e-6,
        incidence_mean_deg=wm.theta_i_mean_deg, dolp_gain=_LAB_GAIN,
        row_sign=wm.raw_attrs.get("row_sign", -1.0), verbose=False)

    pts = _narrow_points(piermont_narrow_stack_path)
    true = np.array([p[0] for p in pts])
    dmeas = np.array([p[1] for p in pts])
    rec_emp = np.array([dolp_to_aoi(np.array([d]), *cal.lut_empirical)[0]
                        for d in dmeas])
    rec_fre = np.array([dolp_to_aoi(np.array([d]), *cal.lut_fresnel)[0]
                        for d in dmeas])
    mae_emp = float(np.nanmean(np.abs(rec_emp - true)))
    mae_fre = float(np.nanmean(np.abs(rec_fre - true)))
    # Headline Piermont result: empirical wide LUT ~2 deg vs ideal ~10 deg.
    assert mae_emp < 4.0
    assert mae_fre > 6.0
    assert mae_emp < mae_fre
