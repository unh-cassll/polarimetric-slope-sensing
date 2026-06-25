#!/usr/bin/env python3
"""Dual-camera wide+narrow polarimetric slope sensing on the Piermont 2025 data.

A WIDE 5 mm camera sees water across ~11-75 deg incidence in one frame, so it
measures the whole DoLP-vs-AOI relationship. A NARROW 75 mm imager sees only a
few degrees and cannot self-calibrate that relationship. This demo uses the wide
camera's measured curve to invert the narrow imager's DoLP into angle of
incidence. The figure has two stacked panels (the single-condition analog of
the paper's overcast/cloudless columns):

    (a) measured DoLP vs angle of incidence -- the wide-FOV curve, the
        multi-angle narrow-imager scan points, the seapol forward model, and the
        ideal-Fresnel curve.
    (b) observed incidence vs the incidence INFERRED from the narrow DoLP
        (inferred on the horizontal axis), for five DoLP->AOI strategies:

            no gain      ideal-Fresnel lookup on the raw narrow DoLP
            lab gain     ditto, with the fixed lab-calibrated DoLP gain
            emp. gain    ditto, with the single-camera empirical DoLP gain
            dual cam     the wide camera's MEASURED DoLP(theta) lookup  <- winner
            pure seapol  the seapol forward-model lookup at the real sun geometry

The first three share the ideal-Fresnel lookup and differ only in the narrow
DoLP gain; the last two keep the lab gain but swap the lookup table.

The wide frame is Stokes-reduced with the Pistellato projective correction
(meaningful on a 5 mm lens). seapol is optional: the 'pure seapol' series and
the panel-(a) model curve are skipped with a note if seapol is not installed.

Reproduces the headline Piermont result (dual cam ~2 deg MAE vs ideal Fresnel
~10 deg). Run from the repo root:

    python _examples/widefov_dualcam_demo.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _data                                                # noqa: E402
from pss import read_netcdf_frame, by_superpixel, dolp_to_aoi, fresnel_dolp
from pss.skyaware import solar_position
from pss.widefov import (calibrate_widefov, per_row_incidence, plane_normal_R,
                         strip_profile, rolling_nanmedian, valid_incidence_mask,
                         _HAS_SEAPOL)
from pss.pistellato import build_K

HEADING_DEG = 350.0           # campaign camera heading (compass, off the dock)
LAB_GAIN = 1.0 / 0.81         # DoLP calibration gain used in the MATLAB analysis
PIX = 3.45e-6
N_WATER = 1.34


def _sun_geometry(meta):
    """(zenith, compass-azimuth) from the file's UTC acquisition time + site."""
    attrs = meta.raw_attrs
    t = dt.datetime.fromisoformat(attrs["acquisition_time_utc"])
    return solar_position(t, attrs["site_lat"], attrs["site_lon"])


def _narrow_points(stack_path):
    """Per-run (true incidence, RAW median DoLP) for the narrow imager, plus the
    camera's known mean incidence (the empirical-gain reference angle). The DoLP
    is returned un-gained; the gain treatments are applied by the caller."""
    from netCDF4 import Dataset
    with Dataset(str(stack_path)) as ds:
        theta_i = np.asarray(ds.variables["theta_i_per_frame"][...], float)
        theta_i_mean = float(np.asarray(ds.variables["theta_i_mean"][...]))
        T = theta_i.size
    pts = []
    for t in range(T):
        frame, _ = read_netcdf_frame(str(stack_path), time_index=t)
        _, s1, s2 = by_superpixel(frame)
        dolp = np.sqrt(s1 ** 2 + s2 ** 2)            # raw; gains applied by caller
        prof = rolling_nanmedian(strip_profile(dolp), 9)
        med = float(np.nanmedian(prof[np.isfinite(prof)]))
        pts.append((float(theta_i[t]), med))
    return pts, theta_i_mean


def main():
    # ---- wide camera: build the DoLP->AOI calibration ----
    wide_path = _data.piermont_wide_path()
    wide_frame, wmeta = read_netcdf_frame(str(wide_path))
    sun_z, sun_a = _sun_geometry(wmeta)
    cal = calibrate_widefov(
        wide_frame,
        focal_length_m=float(wmeta.raw_vars["lens_focal_length"]["value"]),
        pixel_pitch_m=float(wmeta.raw_vars["pixel_pitch"]["value"]),
        incidence_mean_deg=wmeta.theta_i_mean_deg,
        n_water=N_WATER, row_sign=wmeta.raw_attrs.get("row_sign", -1.0),
        dolp_gain=LAB_GAIN,
        sun_zenith_deg=sun_z, sun_azimuth_deg=sun_a, heading_deg=HEADING_DEG,
        verbose=True)

    # ---- narrow camera: recover AOI through each DoLP->AOI strategy ----
    pts, theta_ref = _narrow_points(_data.piermont_narrow_stack_path())
    true = np.array([p[0] for p in pts])
    dmeas_raw = np.array([p[1] for p in pts])          # raw narrow DoLP per run

    # DoLP gains for the narrow imager:
    #   no gain   - raw measurement, no correction
    #   lab gain  - fixed lab-calibrated gain (the campaign value)
    #   emp. gain - force the median DoLP onto the ideal-Fresnel value at the
    #               camera's known mean incidence (single-camera empirical recipe)
    g_emp = fresnel_dolp(theta_ref, n_water=N_WATER) / float(np.median(dmeas_raw))
    narrow_gains = {"no gain": 1.0, "lab gain": LAB_GAIN, "emp. gain": g_emp}

    # Inference strategies, in figure order. The first three share the ideal-
    # Fresnel lookup and differ only in the narrow DoLP gain; "dual cam" swaps in
    # the wide camera's MEASURED DoLP(theta) lookup (lab-gained DoLP); "pure
    # seapol" is the seapol forward-model lookup (also lab-gained DoLP).
    dmeas_lab = dmeas_raw * LAB_GAIN
    series = [(name, dmeas_raw * g, cal.lut_fresnel)
              for name, g in narrow_gains.items()]
    series.append(("dual cam", dmeas_lab, cal.lut_empirical))
    if cal.lut_seapol is not None:
        series.append(("pure seapol", dmeas_lab, cal.lut_seapol))

    print(f"\nsun: zenith {sun_z:.1f} deg, azimuth {sun_a:.0f} deg (compass); "
          f"heading {HEADING_DEG:.0f} deg")
    print(f"narrow scan: {len(pts)} runs, true AOI "
          f"{true.min():.0f}-{true.max():.0f} deg; emp.-gain ref "
          f"{theta_ref:.1f} deg, g_emp {g_emp:.3f}\n")
    recovered = {}
    print(f"{'DoLP->AOI source':<26}  MAE [deg]")
    print("-" * 40)
    for name, dolp_in, lut in series:
        rec = np.array([dolp_to_aoi(np.array([d]), *lut)[0] for d in dolp_in])
        recovered[name] = rec
        mae = float(np.nanmean(np.abs(rec - true)))
        print(f"{name:<26}  {mae:6.2f}")
    if not _HAS_SEAPOL:
        print("\n(seapol not installed -> 'pure seapol' skipped; "
              "install with: pip install 'epss[skyaware]')")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _figstyle import figure_style
    except Exception as e:
        print(f"(matplotlib unavailable: {e!r}; skipping figure)")
        return
    colors, fullwidth, _, fsize = figure_style()
    # Strategies take the figure_style color cycle in order.
    cmap = {name: colors[i % len(colors)]
            for i, (name, _, _) in enumerate(series)}

    AX_MAX = 90.0
    ticks = np.arange(0, AX_MAX + 1, 15)
    fig, (axT, axB) = plt.subplots(2, 1, sharex=True,
                                   figsize=(0.62 * fullwidth, fullwidth))

    # (a) measured DoLP vs angle of incidence: wide curve, narrow scan points,
    #     the seapol forward model, and the ideal-Fresnel curve.
    the, de = cal.theta_deg, cal.dolp_measured              # wide curve (lab-gained)
    o = np.argsort(the)
    mm = np.isfinite(the[o]) & np.isfinite(de[o])
    axT.plot(the[o][mm], de[o][mm], "-", lw=3, color=colors[4],
             label="5 mm lens (wide)")
    axT.plot(true, dmeas_lab, "*", ms=11, color=colors[5],
             label="75 mm lens (narrow)")
    if cal.lut_seapol is not None:
        ds_, ths_ = cal.lut_seapol
        os_ = np.argsort(ths_)
        axT.plot(ths_[os_], ds_[os_], "--", color="black", lw=1.5,
                 label="seapol model")
    thg = np.linspace(0, AX_MAX, 400)
    axT.plot(thg, fresnel_dolp(thg, n_water=N_WATER), "-", color=(0.5, 0.5, 0.5),
             label="ideal Fresnel")
    axT.set(ylabel="DoLP", ylim=(0, 1.1))
    axT.legend(loc="lower right", fontsize=fsize - 2)

    # (b) observed vs INFERRED incidence (inferred on x), one series per strategy.
    axB.plot([0, AX_MAX], [0, AX_MAX], "k:", lw=1)
    for name, _, _ in series:
        inferred = recovered[name]
        mae = float(np.nanmean(np.abs(inferred - true)))
        axB.plot(inferred, true, "s-", ms=5, lw=1.5, color=cmap[name],
                 label=f"{name} ({mae:.1f}°)")
    axB.set(xlabel=r"inferred $\theta_i$ [deg]", ylabel=r"observed $\theta_i$ [deg]",
            xlim=(0, AX_MAX), ylim=(0, AX_MAX), xticks=ticks, yticks=ticks)
    axB.legend(loc="lower right", fontsize=fsize - 2, title="MAE [deg]")

    for a, lab in zip((axT, axB), ("(a)", "(b)")):
        a.text(0.03, 0.95, lab, transform=a.transAxes, ha="left", va="top",
               fontsize=fsize)
    fig.suptitle("Dual-camera polarimetric slope sensing (Piermont 2025)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out_widefov_dualcam.png")
    fig.savefig(out, dpi=200)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
