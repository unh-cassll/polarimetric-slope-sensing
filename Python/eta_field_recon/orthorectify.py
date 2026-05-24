"""
orthorectify -- static (fixed-platform) orthorectification of slope fields.

Projects an obliquely-viewed slope field from the image plane onto a uniform
grid on the flat mean sea surface, using the fixed acquisition geometry that
the NetCDF file already carries:

    freeboard          camera height above the mean surface  H   (m)
    theta_i_mean       mean incidence angle of the optical axis  (deg from vertical)
    lens_focal_length  f   (mm)
    pixel_pitch        sensor pixel pitch  (um)
    camera_azimuth     look azimuth (deg); used only to label axes here

This is the STATIC case only: the platform does not move, so the camera pose
is constant and the projection is a single fixed mapping computed once -- no
per-frame motion and no IMU needed. The moving-platform case (per-frame
attitude-driven rectification) is deliberately out of scope; see the TODO in
eta_pipeline.py.

Why this matters for the eta inversion
--------------------------------------
An oblique view samples the surface non-uniformly: pixels imaging the far
(more grazing) part of the footprint cover more ground than near pixels, so
the footprint is a trapezoid and the effective ground dx VARIES across the
frame. reconstruct_eta_field assumes a single constant dx on a uniform grid.
Orthorectifying resamples the trapezoid onto a uniform grid and yields the
one true dx, which is exactly the value reconstruct_eta_field needs (and which
the pipeline previously required the caller to supply by hand).

What this does NOT do: rotate the (Sx, Sy) slope vectors for camera tilt.
On a static platform that tilt is a CONSTANT offset, and pss.compute_slope_field
already subtracts the per-frame spatial mean of Sx and Sy (slope.py), which
removes any constant tilt as a side effect. Adding a tilt rotation here would
double-correct. So orthorectification is purely the geometric pixel resample.

Single-tilt-plane assumption
----------------------------
The tilt is taken to lie in one image plane (the camera looks down at
theta_i in the plane of the "along-look" image axis; the orthogonal axis is
"cross-look"). This is the simplest correct model for a camera tilted in a
single plane (the typical nadir-ish oblique mount) and needs no full 3-D
attitude. The cross-look axis is treated as level. If a future setup has a
compound tilt, this is where a full rotation matrix would go.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OrthoPlan:
    """Reusable orthorectification plan for a fixed (static) geometry.

    Holds everything that depends only on the acquisition geometry and the
    input field shape -- the ground coordinates, the target grid, and the
    precomputed Delaunay triangulation + barycentric weights. Building this is
    the expensive step (the ~1e6-point triangulation); applying it to a frame
    is a cheap gather-and-dot. Build ONCE with build_ortho_plan(), then pass it
    to orthorectify_static(..., plan=plan) for every frame of the same camera
    setup -- this is what makes a long record fast.
    """
    Ny: int
    Nx: int
    dx_m: float
    x_ground: np.ndarray
    y_ground: np.ndarray
    Nyg: int
    Nxg: int
    finite: np.ndarray          # (Ny, Nx) bool: input pixels with valid ground
    inside: np.ndarray          # (n_targets,) bool: targets inside the hull
    weights: np.ndarray         # (M, 3) barycentric weights
    vertices: np.ndarray        # (M, 3) input-point indices per target
    n_targets: int
    valid: np.ndarray           # (Nyg, Nxg) bool: grid cells with data
    diag: dict[str, Any]


@dataclass
class OrthoResult:
    """Output of orthorectify_static for one frame (or a stack)."""
    slope_x: np.ndarray   # resampled cross-look slope on the uniform grid
    slope_y: np.ndarray   # resampled along-look slope on the uniform grid
    dx_m: float           # the single true ground pixel size (m), uniform
    x_ground: np.ndarray  # 1-D uniform cross-look ground coordinate (m)
    y_ground: np.ndarray  # 1-D uniform along-look ground coordinate (m)
    valid: np.ndarray     # bool mask: True where the uniform grid had data
    diag: dict[str, Any]  # geometry intermediates for inspection


def _ground_coordinates(Ny, Nx, H, theta_i_deg, f_m, pitch_m):
    """Per-pixel ground (X cross-look, Y along-look) for a single-plane tilt.

    Camera at height H, optical axis at incidence theta_i from vertical, tilt
    in the along-look (image-row / y) plane. Returns X, Y arrays of shape
    (Ny, Nx) giving where each pixel's ray meets the z=0 mean surface.
    """
    th = np.deg2rad(theta_i_deg)
    ix = (np.arange(Nx) - Nx / 2.0)
    iy = (np.arange(Ny) - Ny / 2.0)
    # angle of each pixel's ray from the optical axis
    ax = np.arctan(ix * pitch_m / f_m)   # cross-look
    ay = np.arctan(iy * pitch_m / f_m)   # along-tilt
    AX, AY = np.meshgrid(ax, ay)

    # camera-frame ray direction (optical axis = +z_cam)
    dx_c = np.tan(AX)
    dy_c = np.tan(AY)
    dz_c = np.ones_like(AX)

    # rotate about the cross-look (x) axis by theta_i: optical axis now points
    # down at incidence th from vertical. World z is up; the camera is at +H.
    dy_w = dy_c * np.cos(th) - dz_c * np.sin(th)
    dz_w = -(dy_c * np.sin(th) + dz_c * np.cos(th))   # downward => negative
    dx_w = dx_c

    # intersect z = 0 from (0, 0, H): H + t * dz_w = 0  ->  t = -H / dz_w
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -H / dz_w
    t = np.where(dz_w < 0, t, np.nan)   # rays not pointing down never hit
    X = t * dx_w
    Y = t * dy_w
    return X, Y


def orthorectify_static(
    slope_x,
    slope_y,
    *,
    freeboard_m: float,
    theta_i_mean_deg: float,
    focal_length_m: float,
    pixel_pitch_m: float,
    camera_azimuth_deg: float | None = None,
    target_dx_m: float | None = None,
    fill_value: float = np.nan,
    plan: "OrthoPlan | None" = None,
    verbose: bool = False,
) -> OrthoResult:
    """Project an obliquely-viewed slope field onto a uniform ground grid.

    Accepts either a single 2-D frame (Ny, Nx) or a stack (T, Ny, Nx); a
    stack is rectified frame-by-frame onto a common grid (the geometry is
    identical for every frame in the static case, so the grid is shared).

    Args:
        slope_x, slope_y : (Ny, Nx) or (T, Ny, Nx) slope fields (rad).
        freeboard_m      : camera height above the mean surface (m).
        theta_i_mean_deg : mean incidence angle of the optical axis (deg).
        focal_length_m   : lens focal length (m).
        pixel_pitch_m    : ground-sample pixel pitch of the INPUT field (m).
                           NOTE: this is the pitch of the slope-field samples,
                           which at pss resolution="native" is TWICE the
                           sensor pixel pitch (one Stokes vector per 2x2
                           super-pixel). The caller must pass the pitch that
                           matches the slope field it is handing in.
        camera_azimuth_deg : look azimuth (deg). Recorded in diag and used to
                           label the ground axes; does not change the resample
                           in the single-tilt-plane model.
        target_dx_m      : uniform output ground pixel size (m). If None, uses
                           the median of the native ground sampling (a natural,
                           information-preserving choice).
        fill_value       : value for uniform-grid cells with no data.
        verbose          : print the inferred geometry.

    Returns:
        OrthoResult. `dx_m` is the single true ground dx to feed downstream.
    """
    sx = np.asarray(slope_x, dtype=float)
    sy = np.asarray(slope_y, dtype=float)
    stacked = sx.ndim == 3
    if not stacked:
        sx = sx[None]
        sy = sy[None]
    T, Ny, Nx = sx.shape

    # Build the (expensive) geometry/triangulation plan, or reuse one passed
    # in. Reusing a plan across calls is what makes per-frame processing fast:
    # the ~1e6-point Delaunay triangulation is then built ONCE, not per call.
    if plan is None:
        plan = build_ortho_plan(
            Ny, Nx, freeboard_m=freeboard_m, theta_i_mean_deg=theta_i_mean_deg,
            focal_length_m=focal_length_m, pixel_pitch_m=pixel_pitch_m,
            camera_azimuth_deg=camera_azimuth_deg, target_dx_m=target_dx_m,
            verbose=verbose)
    elif (plan.Ny, plan.Nx) != (Ny, Nx):
        raise ValueError(
            f"plan was built for input shape ({plan.Ny}, {plan.Nx}) but the "
            f"slope field is ({Ny}, {Nx}).")

    sx_out = np.empty((T, plan.Nyg, plan.Nxg), dtype=float)
    sy_out = np.empty((T, plan.Nyg, plan.Nxg), dtype=float)
    for ti in range(T):
        sx_out[ti] = _apply_plan(plan, sx[ti], fill_value)
        sy_out[ti] = _apply_plan(plan, sy[ti], fill_value)

    if not stacked:
        sx_out = sx_out[0]
        sy_out = sy_out[0]

    return OrthoResult(
        slope_x=sx_out, slope_y=sy_out,
        dx_m=plan.dx_m,
        x_ground=plan.x_ground, y_ground=plan.y_ground,
        valid=plan.valid, diag=plan.diag,
    )


def _apply_plan(plan: OrthoPlan, frame_2d: np.ndarray,
                fill_value: float) -> np.ndarray:
    """Resample one (Ny, Nx) field onto the plan's uniform grid (cheap)."""
    values_finite = frame_2d[plan.finite].ravel()
    out = np.full(plan.n_targets, fill_value, dtype=float)
    out[plan.inside] = np.einsum(
        "mi,mi->m", values_finite[plan.vertices], plan.weights)
    return out.reshape(plan.Nyg, plan.Nxg)


def build_ortho_plan(
    Ny: int,
    Nx: int,
    *,
    freeboard_m: float,
    theta_i_mean_deg: float,
    focal_length_m: float,
    pixel_pitch_m: float,
    camera_azimuth_deg: float | None = None,
    target_dx_m: float | None = None,
    verbose: bool = False,
) -> OrthoPlan:
    """Build a reusable orthorectification plan for one fixed geometry.

    This does the expensive, frame-independent work: ground coordinates, the
    uniform target grid, the Delaunay triangulation of the input points, and
    the barycentric weights locating every target in it. Pass the result to
    orthorectify_static(..., plan=plan) to rectify many frames cheaply.
    """
    from scipy.spatial import Delaunay

    X, Y = _ground_coordinates(Ny, Nx, freeboard_m, theta_i_mean_deg,
                               focal_length_m, pixel_pitch_m)
    finite = np.isfinite(X) & np.isfinite(Y)

    dX = np.diff(X, axis=1)
    dY = np.diff(Y, axis=0)
    native_dx = float(np.nanmedian(np.abs(np.concatenate(
        [dX[finite[:, 1:]].ravel(), dY[finite[1:, :]].ravel()]))))
    if target_dx_m is None:
        target_dx_m = native_dx

    col = Nx // 2
    ycol = Y[:, col]
    yc_finite = ycol[np.isfinite(ycol)]
    if yc_finite.size > 2:
        spacings = np.abs(np.diff(yc_finite))
        trapezoid_ratio = float(spacings.max() / spacings.min())
    else:
        trapezoid_ratio = float("nan")

    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)
    x_ground = np.arange(x_min, x_max + target_dx_m, target_dx_m)
    y_ground = np.arange(y_min, y_max + target_dx_m, target_dx_m)
    Xg, Yg = np.meshgrid(x_ground, y_ground)
    Nyg, Nxg = Xg.shape

    if verbose:
        print("build_ortho_plan:")
        print(f"  input        : {Ny}x{Nx}")
        print(f"  geometry     : H={freeboard_m} m, theta_i={theta_i_mean_deg} deg, "
              f"f={focal_length_m*1000:.1f} mm, pitch={pixel_pitch_m*1e3:.4f} mm")
        if camera_azimuth_deg is not None:
            print(f"  azimuth      : {camera_azimuth_deg} deg")
        print(f"  footprint    : X {x_min:.3f}..{x_max:.3f} m, "
              f"Y {y_min:.3f}..{y_max:.3f} m")
        print(f"  native dx    : {native_dx*1000:.3f} mm  "
              f"(trapezoid near/far ratio {trapezoid_ratio:.3f})")
        print(f"  output grid  : {Nyg}x{Nxg} @ dx={target_dx_m*1000:.3f} mm")
        print(f"  triangulating {int(finite.sum())} input points ...")

    pts = np.column_stack([X[finite].ravel(), Y[finite].ravel()])
    targets = np.column_stack([Xg.ravel(), Yg.ravel()])

    tri = Delaunay(pts)
    simplex = tri.find_simplex(targets)
    inside = simplex >= 0

    tr = tri.transform[simplex[inside]]
    delta = targets[inside] - tr[:, 2, :]
    bary = np.einsum("mij,mj->mi", tr[:, :2, :], delta)
    weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
    vertices = tri.simplices[simplex[inside]]
    n_targets = targets.shape[0]

    valid = np.zeros(n_targets, dtype=bool)
    valid[inside] = True
    valid = valid.reshape(Nyg, Nxg)

    diag = dict(
        X=X, Y=Y, finite=finite,
        native_dx=native_dx, trapezoid_ratio=trapezoid_ratio,
        camera_azimuth_deg=camera_azimuth_deg,
        Xg=Xg, Yg=Yg,
    )

    return OrthoPlan(
        Ny=Ny, Nx=Nx, dx_m=float(target_dx_m),
        x_ground=x_ground, y_ground=y_ground, Nyg=Nyg, Nxg=Nxg,
        finite=finite, inside=inside, weights=weights, vertices=vertices,
        n_targets=n_targets, valid=valid, diag=diag,
    )
