"""
Projective-camera polarizer-tilt correction (Pistellato & Bergamasco 2024).

Shape-from-polarization assumes an *orthographic* camera: every ray strikes the
sensor perpendicularly, so each DoFP micropolarizer is square-on to its ray. Real
lenses are projective -- peripheral rays are tilted, the per-pixel polarizer is
tilted relative to its ray, and the naively-demodulated Stokes vector (hence
AoLP/DoLP) is biased. The bias scales with field of view: it is several degrees of
AoLP on a wide (5 mm) lens and negligible (~0.02 deg) on a telephoto (75 mm).

This module implements the paper's per-pixel correction in a tight, vectorized
form. For an ideal polarizer whose effective 2-D transmitting axis is t = (tx, ty),
the Jones matrix is t t^T and the first row of its Mueller matrix is

    M[0, :3] = [ 0.5 (tx^2 + ty^2)^2,
                 0.5 (tx^2 - ty^2)(tx^2 + ty^2),
                 tx ty (tx^2 + ty^2) ].

The four tilted polarizers give a 4x3 linear system per pixel for (S0, S1, S2).
The design matrix depends only on the camera geometry (intrinsics K), not on the
measured intensities, so it is built once for the whole image and every pixel's
system is solved in a single batched ``np.linalg.solve``.

The geometric core (`get_camera_rays`, `compute_effective_transm_ax`) and the
batched solve (`compute_stokes_from_tilted_polarizers_fast`) are ported verbatim
from the reference implementation accompanying the paper; only the epss-facing
wrappers (`build_K`, `corrected_stokes_superpixel`) are new.

Reference:
    Pistellato & Bergamasco, "A Geometric Model for Polarization Imaging on
    Projective Cameras", IJCV 132:4688-4702 (2024).
"""

from __future__ import annotations

import numpy as np

from .stokes import _extract_channels


def get_camera_rays(W: int, H: int, K: np.ndarray) -> np.ndarray:
    """Unit ray directions leaving each pixel, in camera coordinates.

    Returns a (3, H*W) array of column vectors (norm 1), row-major over the
    (H, W) grid -- the back-projection of each pixel through the intrinsics K.
    """
    xx, yy = np.meshgrid(range(W), range(H))
    rays = np.stack((xx.flatten(), yy.flatten(), np.ones((H * W), dtype=float)),
                    axis=0)
    rays = np.linalg.inv(K) @ rays
    rays = rays / np.linalg.norm(rays, axis=0)
    return rays


def compute_effective_transm_ax(W: int, H: int, K: np.ndarray) -> list[np.ndarray]:
    """Effective transmitting axes of the tilted 0/45/90/135 polarizers.

    Returns a 4-element list (one per nominal polarizer angle, in the order
    [0, 45, 90, 135]) of (3, H*W) matrices; each column vector has norm 1. Each
    is the in-image-plane projection of that polarizer's transmitting axis once
    the per-pixel ray tilt is accounted for.
    """
    # camera rays are the z axis of the local (per-pixel) reference frame
    r_z = get_camera_rays(W, H, K)

    vertical = np.tile(np.array([[0, 1, 0]]).T, (1, W * H))
    r_x = np.cross(vertical, r_z, axisa=0, axisb=0, axisc=0)
    r_x = r_x / np.linalg.norm(r_x, axis=0)
    r_y = np.cross(r_z, r_x, axisa=0, axisb=0, axisc=0)

    pi_2 = np.pi * 0.5
    pol_angles = [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]

    all_alpha_vec = [np.array([[np.cos(a + pi_2), -np.sin(a + pi_2), 0]]).T
                     for a in pol_angles]

    z = np.tile(np.array([[0, 0, 1]]).T, (1, W * H))
    eff_t = []
    for alpha in all_alpha_vec:
        # move alpha into the per-pixel ray coordinate frame
        P_a = np.concatenate([np.dot(r_x.T, alpha),
                              np.dot(r_y.T, alpha),
                              np.dot(r_z.T, alpha)], axis=1).T
        t_h = np.cross(z, P_a, axisa=0, axisb=0, axisc=0)
        t_h = t_h / np.linalg.norm(t_h, axis=0)
        eff_t.append(t_h)
    return eff_t


def compute_stokes_from_tilted_polarizers_fast(
    I: np.ndarray, K: np.ndarray, mask_img: np.ndarray | None = None
) -> np.ndarray:
    """Vectorized projective Stokes solve from four tilted-polarizer channels.

    Parameters
    ----------
    I : ndarray, shape (H, W, 4)
        Channel stack ordered (I0, I45, I90, I135).
    K : ndarray, shape (3, 3)
        Camera intrinsics at the working (channel) resolution.
    mask_img : ndarray, shape (H, W), optional
        Pixels that are NaN or zero are zeroed in the output.

    Returns
    -------
    S : ndarray, shape (H, W, 3)
        Corrected Stokes field (S0, S1, S2).
    """
    H, W, _ = I.shape
    eff_t = compute_effective_transm_ax(W, H, K)  # 4 x (3, H*W)

    A = np.empty((H * W, 4, 3))
    for i in range(4):
        tx = eff_t[i][0, :]
        ty = -eff_t[i][1, :]          # matches t_h[1,:] *= -1 in the original
        r = tx * tx + ty * ty
        A[:, i, 0] = 0.5 * r * r
        A[:, i, 1] = 0.5 * (tx * tx - ty * ty) * r
        A[:, i, 2] = tx * ty * r

    Iflat = I.reshape(H * W, 4)
    AtA = np.einsum("nij,nik->njk", A, A)
    Atb = np.einsum("nij,ni->nj", A, Iflat)
    S = np.linalg.solve(AtA, Atb[..., None])[..., 0].reshape(H, W, 3)

    if mask_img is not None:
        m = (~np.isnan(mask_img)) & (mask_img != 0)
        S = S * m[..., None]
    return S


def build_K(
    focal_length_m: float, pixel_pitch_m: float, W: int, H: int,
    *, half_res: bool = True,
) -> np.ndarray:
    """Pinhole intrinsics at the working (reduced) resolution.

    Parameters
    ----------
    focal_length_m : float
        Lens focal length (m).
    pixel_pitch_m : float
        SENSOR pixel pitch (m).
    W, H : int
        The grid the correction runs on -- HALF the sensor dims when
        ``half_res`` (one Stokes vector per 2x2 super-pixel).
    half_res : bool
        When True (default), the pixel pitch is doubled to match the
        super-pixel grid, so the focal length in pixels uses 2*pixel_pitch_m.
    """
    pitch = pixel_pitch_m * (2.0 if half_res else 1.0)
    f_px = focal_length_m / pitch
    return np.array([[f_px, 0.0, W / 2.0],
                     [0.0, f_px, H / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def corrected_stokes_superpixel(
    frame: np.ndarray, *, focal_length_m: float, pixel_pitch_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pistellato-corrected (S0, s1, s2) at native (half) resolution.

    Same contract as `pss.stokes.by_superpixel`: one Stokes vector per 2x2
    super-pixel, returned at HALF resolution (H/2, W/2). The four orientation
    channels are split by name from `pss.stokes._extract_channels` (so the
    correction tracks a runtime-patched `_OFFSETS` and stays layout-agnostic),
    assembled in the (I0, I45, I90, I135) order the design matrix expects, and
    solved through the projective Stokes system with intrinsics K built at the
    half-resolution grid.
    """
    ch = _extract_channels(frame)   # each (H/2, W/2), keyed by orientation name
    I = np.stack([ch["I0"], ch["I45"], ch["I90"], ch["I135"]], axis=-1)
    ny, nx, _ = I.shape
    K = build_K(focal_length_m, pixel_pitch_m, nx, ny, half_res=True)
    S = compute_stokes_from_tilted_polarizers_fast(I, K)
    S0 = S[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(S0 > 0, S[..., 1] / S0, 0.0)
        s2 = np.where(S0 > 0, S[..., 2] / S0, 0.0)
    return S0, s1, s2
