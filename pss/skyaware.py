"""
Sky-aware Stokes -> slope inversion (single-camera, mainline E-PSS path).

Replaces the "AoP = plane-of-incidence azimuth" Fresnel-lookup inversion with the
full environmental forward model: for the camera/sun/water geometry, every facet
slope (s_along, s_cross) is forward-modeled to normalized Stokes (s1, s2) through
seapol's meridian-frame Mueller chain (clear/Rayleigh sky radiance at the mirror
direction, Fresnel reflection, first-order water-leaving), and the map is inverted
by a dense measurement-space lookup table (3-D nearest-neighbor with the intensity
channel resolving the sky-polarization fold).

This module is the library home of the inverter that previously lived in the
dev tree (_tools/sky_aware_inversion.py). It carries an OPTIONAL dependency on the
`seapol` package; importing `pss` (and the default Fresnel/empirical/lab paths)
never requires seapol. Selecting the sky-aware path without seapol raises a clear,
actionable error (see `require_seapol`).

Frames and conventions (validated):
- working frame: +x = surface->camera horizontal, +z up; s_along = d(eta)/dx
  positive rising toward the camera; s_cross = d(eta)/dy, +y toward
  look-azimuth + 90.
- seapol camera-meridian Stokes -> pss frame: s1 = -Q/I, s2 = -U/I, so the
  measurement table is built directly in the pss normalized-Stokes frame and the
  inverter consumes pss-frame (s1, s2) as produced by pss.stokes.
- ambiguity: smallest-|slope| branch wins per measurement cell; with S0 the 3-D
  kNN resolves the sky-polarization fold by brightness; cells outside the forward
  image -> NaN (out-of-model: noise, foam, sun glint, saturation).

Not modeled: direct sun glint (sun disk), foam, multiple scattering.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .fresnel import build_lookup_table
from .anchor import (present_slope_from_stokes, slope_anchor_gain,
                     AnchorResult)

# ---- optional seapol dependency (guarded) --------------------------------
_SEAPOL_IMPORT_ERROR = None
try:
    from seapol.render import (make_clear_sky, make_unpolarized_sky,
                               downwelling_irradiance)
    from seapol.skylight import direction_from_angles
    from seapol.polarization import apply_mueller, normalize, reflection_chain
    from seapol.water import water_leaving_stokes, WaterBody
except Exception as _e:                      # ImportError or transitive failure
    _SEAPOL_IMPORT_ERROR = _e


def require_seapol() -> None:
    """Raise an actionable ImportError if the optional seapol dep is unusable."""
    if _SEAPOL_IMPORT_ERROR is not None:
        raise ImportError(
            "The sky-aware inversion path requires the optional 'seapol' package, "
            "which is not importable:\n"
            f"    {_SEAPOL_IMPORT_ERROR!r}\n"
            "Install it with:  pip install 'epss[skyaware]'  (or install the "
            "surface-wave-light-reflection / seapol package directly).\n"
            "The default Fresnel inversion and the none/lab/empirical DoLP-gain "
            "paths do NOT need seapol."
        ) from _SEAPOL_IMPORT_ERROR


# ---- optional torch GPU backend for the 3-D kNN (device-agnostic) --------
# The S0 inversion is a per-frame 3-D k=1 nearest-neighbor query of ~N=512^2
# points against an ~M~5e4 forward-model cloud -- the dominant cost of the
# sky-aware path. scipy cKDTree (CPU) is the default/fallback; when torch with a
# CUDA or ROCm accelerator is present, the query runs as a chunked
# squared-distance matmul + argmin on-device (model resident across frames). The
# code is device-agnostic (no faiss/cuML/cupy): only the installed torch wheel
# differs between NVIDIA (CUDA) and AMD (ROCm).
try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:
    _torch = None
    _HAS_TORCH = False


def _resolve_knn_device(backend):
    """Map a backend request to a kNN device. Returns None to use the scipy
    cKDTree (CPU) path, or a torch device string ('cuda'/'cpu') for the torch
    path. `backend`:
      'auto'   -> cKDTree (the default; honors the PSS_KNN_BACKEND env override).
      'cpu'/'kdtree'/'scipy' -> cKDTree.
      'gpu'/'cuda'/'rocm'/'torch' -> torch path (errors if torch missing; uses
                  the GPU if available, else torch-CPU).

    NOTE: 'auto' deliberately resolves to cKDTree, NOT the GPU. For this
    workload (one frame = N~512^2 queries against an M~5e4 forward-model cloud)
    the inversion is a 3-D k=1 nearest-neighbor search, and a multithreaded
    cKDTree (O(N log M)) is ~30x FASTER than a brute-force GPU distance matmul
    (O(N M)) on a mid-range accelerator -- benchmarked 27 ms vs ~985 ms/frame on
    a Quadro RTX 4000. The torch path is kept as a portable, validated option
    for environments with few CPU cores or where M is small; select it
    explicitly (knn_backend='gpu' or PSS_KNN_BACKEND=gpu). The GPU win in this
    pipeline is the FFT spectral path, not the kNN."""
    import os
    if backend == "auto":
        backend = os.environ.get("PSS_KNN_BACKEND", "cpu")
    backend = (backend or "cpu").lower()
    if backend in ("cpu", "kdtree", "scipy", "auto"):
        return None
    if backend in ("gpu", "cuda", "rocm", "torch"):
        if not _HAS_TORCH:
            raise ImportError(
                f"knn_backend={backend!r} requires torch, which is not "
                "installed. Install it with:  pip install 'epss[gpu]'  (choose "
                "the CUDA or ROCm wheel matching your hardware).")
        return "cuda" if _torch.cuda.is_available() else "cpu"
    raise ValueError(f"unknown knn_backend {backend!r}")


def _knn_query_torch(model_t, mnorm, query_np, device, chunk):
    """Device-agnostic k=1 nearest-neighbor over a resident model cloud.

    model_t : (M, 3) float32 model points on `device`
    mnorm   : (M,) precomputed ||model||^2 on `device`
    query_np: (N, 3) query points (numpy)
    Returns (dist2, idx) as numpy; dist2 is the SQUARED Euclidean distance
    (compare against d_reject**2 -- avoids a sqrt and matches the cKDTree
    rejection semantics). The (chunk x M) score tile bounds device memory."""
    q_all = _torch.as_tensor(query_np, dtype=_torch.float32, device=device)
    n = q_all.shape[0]
    idx = _torch.empty(n, dtype=_torch.int64, device=device)
    dist2 = _torch.empty(n, dtype=_torch.float32, device=device)
    for i in range(0, n, chunk):
        q = q_all[i:i + chunk]
        # ||q - m||^2 = ||m||^2 - 2 q.m  (+ ||q||^2, restored after argmin)
        score = mnorm.unsqueeze(0) - 2.0 * (q @ model_t.T)
        smin, imin = score.min(dim=1)
        idx[i:i + q.shape[0]] = imin
        dist2[i:i + q.shape[0]] = smin + (q * q).sum(1)
    return dist2.cpu().numpy(), idx.cpu().numpy()


# ==========================================================================
# Forward-model LUT inverter (ported verbatim from the validated dev tree).
# ==========================================================================
class SkyAwareInverter:
    """Forward-model LUT inversion for one acquisition geometry.

    Args:
        theta_v_deg : camera incidence (deg above nadir)
        n_water     : refractive index
        sun_zenith_deg, sun_azimuth_scene_deg :
            sun position; azimuth in the SCENE frame (+x = toward camera,
            CCW). None -> unpolarized sky only.
        water       : seapol WaterBody or None (the upwelling_scale=1 water term)
        sky_depolarization, upwelling_scale : initial environment (see set_environment)
        s_max, n_s  : slope-grid half-range / size
        m_max, n_m  : measurement-grid half-range / size for (s1, s2)
        max_fill    : hole-fill radius in measurement cells (beyond -> NaN)
    """

    def __init__(self, theta_v_deg=30.0, n_water=1.34,
                 sun_zenith_deg=None, sun_azimuth_scene_deg=None,
                 water=None, sky_depolarization=0.0, upwelling_scale=1.0,
                 s_max=0.75, n_s=301, m_max=1.05, n_m=421, max_fill=3.0,
                 knn_backend="auto", knn_tile_bytes=512 * 1024 ** 2):
        require_seapol()
        self.theta_v_deg = theta_v_deg
        self.n_water = n_water
        self.m_max, self.n_m = float(m_max), int(n_m)
        self.dm = 2.0 * m_max / (n_m - 1)
        self.max_fill = float(max_fill)

        # ---- 3-D kNN backend (cKDTree CPU vs device-agnostic torch) ----
        self.knn_backend = knn_backend
        self._knn_device = _resolve_knn_device(knn_backend)
        self._knn_tile_bytes = int(knn_tile_bytes)
        self._knn_model_t = None             # resident GPU model (torch path)
        self._knn_mnorm = None

        # ---- geometry + Stokes components on the slope grid ----
        s = np.linspace(-s_max, s_max, n_s)
        SA, SC = np.meshgrid(s, s, indexing="ij")     # along (x), cross (y)
        self.SA, self.SC = SA, SC
        n_hat = normalize(np.stack([-SA, -SC, np.ones_like(SA)], axis=-1))
        d_cam = direction_from_angles(np.deg2rad(theta_v_deg), 0.0)
        d_out = np.broadcast_to(d_cam, n_hat.shape)
        M, d_in, valid = reflection_chain(d_out, n_hat, n_water)

        if sun_zenith_deg is None:
            S_sky = np.asarray(make_unpolarized_sky(1.0)(-d_in), dtype=float)
            S_pol = np.zeros_like(S_sky)
        else:
            sky_full = make_clear_sky(sun_zenith_deg, sun_azimuth_scene_deg,
                                      I_sky=1.0, turbidity=0.0)  # external seapol arg (NOT renamed)
            S_full = np.asarray(sky_full(-d_in), dtype=float)
            S_sky = S_full.copy()
            S_sky[..., 1:] = 0.0                       # unpolarized part
            S_pol = S_full - S_sky                     # polarized part
            self._sky_full_fn = sky_full
        self.R0 = apply_mueller(M, S_sky)
        self.R1 = apply_mueller(M, S_pol)
        if water is not None:
            sky_for_Ed = (make_unpolarized_sky(1.0) if sun_zenith_deg is None
                          else self._sky_full_fn)
            E_d = downwelling_irradiance(sky_for_Ed)
            self.RW = water_leaving_stokes(d_out, n_hat, water, E_d, n_water)
        else:
            self.RW = np.zeros_like(self.R0)
        self.valid = valid
        self.set_environment(sky_depolarization, upwelling_scale)

    # ------------------------------------------------------------------
    def _stokes(self, sky_depolarization, upwelling_scale):
        return self.R0 + (1.0 - sky_depolarization) * self.R1 + upwelling_scale * self.RW

    def set_environment(self, sky_depolarization, upwelling_scale=1.0, d_sep=0.12):
        """Finalize the forward map and the TWO-BRANCH inverse table for one
        environment. Branch 1 = smallest |slope| per measurement cell;
        branch 2 = smallest-|slope| alternative at slope distance > d_sep
        (fold disambiguation). Each branch stores its predicted intensity."""
        self.sky_depolarization, self.upwelling_scale = float(sky_depolarization), float(upwelling_scale)
        S = self._stokes(sky_depolarization, upwelling_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = -S[..., 1] / S[..., 0]
            s2 = -S[..., 2] / S[..., 0]
        I = S[..., 0]
        ok = self.valid & np.isfinite(s1) & np.isfinite(s2) & (I > 0)
        self._s1, self._s2, self._I, self._ok = s1, s2, I, ok

        n_m, m_max = self.n_m, self.m_max
        i1 = np.rint((s1 + m_max) / self.dm).astype(int)
        i2 = np.rint((s2 + m_max) / self.dm).astype(int)
        inside = ok & (i1 >= 0) & (i1 < n_m) & (i2 >= 0) & (i2 < n_m)
        flat = (i1 * n_m + i2)[inside]
        sa_v, sc_v, I_v = self.SA[inside], self.SC[inside], I[inside]
        mag = sa_v ** 2 + sc_v ** 2
        order = np.lexsort((mag, flat))      # grouped by cell, |s| ascending
        flat_s, sa_s, sc_s, I_s = (flat[order], sa_v[order], sc_v[order],
                                   I_v[order])
        cells, starts = np.unique(flat_s, return_index=True)
        ends = np.append(starts[1:], len(flat_s))

        shape = (n_m * n_m,)
        A1 = np.full(shape, np.nan); C1 = np.full(shape, np.nan)
        I1 = np.full(shape, np.nan)
        A2 = np.full(shape, np.nan); C2 = np.full(shape, np.nan)
        I2 = np.full(shape, np.nan)
        A1[cells] = sa_s[starts]; C1[cells] = sc_s[starts]
        I1[cells] = I_s[starts]
        d2 = d_sep ** 2
        for c, a, b in zip(cells, starts, ends):
            if b - a < 2:
                continue
            da = sa_s[a + 1:b] - sa_s[a]
            dc = sc_s[a + 1:b] - sc_s[a]
            far = np.nonzero(da * da + dc * dc > d2)[0]
            if far.size:
                j = a + 1 + far[0]
                A2[c], C2[c], I2[c] = sa_s[j], sc_s[j], I_s[j]

        A1 = A1.reshape(n_m, n_m); C1 = C1.reshape(n_m, n_m)
        I1 = I1.reshape(n_m, n_m)
        filled = np.isfinite(A1)
        dist, (jy, jx) = ndimage.distance_transform_edt(
            ~filled, return_indices=True)
        near = dist <= self.max_fill
        self.tabA = np.where(near, A1[jy, jx], np.nan)
        self.tabC = np.where(near, C1[jy, jx], np.nan)
        self.tabI = np.where(near, I1[jy, jx], np.nan)
        self.tabA2 = np.where(near, A2.reshape(n_m, n_m)[jy, jx], np.nan)
        self.tabC2 = np.where(near, C2.reshape(n_m, n_m)[jy, jx], np.nan)
        self.tabI2 = np.where(near, I2.reshape(n_m, n_m)[jy, jx], np.nan)
        self.coverage = float(np.mean(near))
        self.ambiguous = float(np.nanmean(np.isfinite(self.tabA2)[near]))
        self._tree = None                    # KNN rebuilt lazily per env
        return self

    # ------------------------------------------------------------------
    def _ensemble_median_I(self, mss0):
        SA, SC, I, okf = self.SA, self.SC, self._I, self._ok
        w = np.exp(-0.5 * (SA ** 2 / max(mss0[0], 1e-12)
                           + SC ** 2 / max(mss0[1], 1e-12)))[okf]
        Iv = I[okf]
        o = np.argsort(Iv)
        cw = np.cumsum(w[o])
        return float(np.interp(0.5 * cw[-1], cw, Iv[o]))

    def _build_knn(self, w_I, mss0, s_knn_max=0.7):
        sel = self._ok & (self.SA ** 2 + self.SC ** 2 <= s_knn_max ** 2)
        I_med = self._ensemble_median_I(mss0)
        pts = np.column_stack([self._s1[sel], self._s2[sel],
                               w_I * (self._I[sel] / I_med - 1.0)])
        self._tree = cKDTree(pts)            # CPU path + sentinel ("built")
        self._knn_sa = self.SA[sel]
        self._knn_sc = self.SC[sel]
        self._knn_w_I = w_I
        self._knn_I_med = I_med
        if self._knn_device is not None:
            m = pts.shape[0]
            self._knn_model_t = _torch.as_tensor(
                pts, dtype=_torch.float32, device=self._knn_device)
            self._knn_mnorm = (self._knn_model_t ** 2).sum(1)
            # query rows per tile so the (chunk x M) score matrix stays within
            # the device-memory budget (float32)
            self._knn_chunk = max(1024, self._knn_tile_bytes // (4 * max(m, 1)))

    # ------------------------------------------------------------------
    def invert(self, s1, s2, gain=1.0, S0=None, mss0=(0.02, 0.015),
               w_I=0.5, d_reject=0.10, dolp_floor=0.0, glint_mad=0.0):
        """Measured normalized Stokes -> (s_cross, s_along).

        Without S0: 2D table lookup, smallest-|slope| branch, NaN outside
        the forward image.
        With S0: full 3D nearest-neighbor inversion in (s1, s2, w_I *
        relative intensity) -- the continuum of sky-polarization folds is
        resolved by brightness. Matches farther than d_reject -> NaN
        (out-of-model). `gain` scales (s1, s2) before matching; S0 is used
        in relative units (median-matched at the mss0 operating point).

        Robustness gates (default off -> exact legacy behavior):
        `dolp_floor` rejects unpolarized, out-of-model pixels (observed DoLP
        sqrt(s1^2+s2^2) below the floor); `glint_mad` excludes the bright
        sun-glint/saturation tail (S0 above median + glint_mad*MAD) from BOTH
        the per-frame S0 normalizer and the output, so a few glinty pixels no
        longer shift the relative-intensity match for every other pixel."""
        if S0 is None:
            i1 = np.rint((np.asarray(s1) * gain + self.m_max) / self.dm)
            i2 = np.rint((np.asarray(s2) * gain + self.m_max) / self.dm)
            ok = ((i1 >= 0) & (i1 < self.n_m) & (i2 >= 0) & (i2 < self.n_m)
                  & np.isfinite(i1) & np.isfinite(i2))
            i1c = np.clip(np.nan_to_num(i1), 0, self.n_m - 1).astype(int)
            i2c = np.clip(np.nan_to_num(i2), 0, self.n_m - 1).astype(int)
            sa = np.where(ok, self.tabA[i1c, i2c], np.nan)
            sc = np.where(ok, self.tabC[i1c, i2c], np.nan)
            return sc, sa

        if self._tree is None or self._knn_w_I != w_I:
            self._build_knn(w_I, mss0)
        shp = np.asarray(s1).shape
        S0n = np.asarray(S0, dtype=float)
        s1g = np.asarray(s1) * gain
        s2g = np.asarray(s2) * gain
        if dolp_floor <= 0.0 and glint_mad <= 0.0:
            med = np.nanmedian(S0n)                     # legacy normalizer
            gate = None
        else:
            gate = np.isfinite(S0n)                     # robust normalizer pool
            if dolp_floor > 0.0:
                gate &= np.sqrt(s1g ** 2 + s2g ** 2) >= dolp_floor
            if glint_mad > 0.0:
                pool = S0n[gate]
                m0 = np.nanmedian(pool) if pool.size else np.nanmedian(S0n)
                mad = np.nanmedian(np.abs(pool - m0)) * 1.4826 if pool.size else 0.0
                if mad > 0.0:
                    gate &= S0n <= m0 + glint_mad * mad  # drop glint/saturation tail
            med = np.nanmedian(S0n[gate]) if gate.any() else np.nanmedian(S0n)
        I_obs = S0n / med * self._knn_I_med
        pts = np.column_stack([
            s1g.ravel(),
            s2g.ravel(),
            (w_I * (I_obs / self._knn_I_med - 1.0)).ravel()])
        bad = ~np.isfinite(pts).all(axis=1)
        if gate is not None:
            bad |= ~gate.ravel()
        pts[bad] = 0.0
        if self._knn_device is None:
            dist, idx = self._tree.query(pts, workers=-1)
            reject = bad | (dist > d_reject)
        else:
            dist2, idx = _knn_query_torch(
                self._knn_model_t, self._knn_mnorm, pts,
                self._knn_device, self._knn_chunk)
            reject = bad | (dist2 > d_reject * d_reject)
        sa = self._knn_sa[idx].copy()
        sc = self._knn_sc[idx].copy()
        sa[reject] = np.nan
        sc[reject] = np.nan
        return sc.reshape(shp), sa.reshape(shp)

    # ------------------------------------------------------------------
    def predicted_dolp_quantiles(self, mss_along, mss_cross, qs,
                                 sky_depolarization=None, upwelling_scale=None):
        t = self.sky_depolarization if sky_depolarization is None else sky_depolarization
        w = self.upwelling_scale if upwelling_scale is None else upwelling_scale
        S = self._stokes(t, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.sqrt(S[..., 1] ** 2 + S[..., 2] ** 2) / S[..., 0]
        ok = self.valid & np.isfinite(d)
        wgt = np.exp(-0.5 * (self.SA ** 2 / max(mss_along, 1e-12)
                             + self.SC ** 2 / max(mss_cross, 1e-12)))[ok]
        d = d[ok]
        order = np.argsort(d)
        cw = np.cumsum(wgt[order])
        return np.interp(np.asarray(qs, dtype=float) * cw[-1], cw, d[order])

    def predicted_median_dolp(self, mss_along, mss_cross):
        return float(self.predicted_dolp_quantiles(mss_along, mss_cross,
                                                   [0.5])[0])

    # ------------------------------------------------------------------
    def _predicted_aop_strength(self, mss_along, mss_cross, sky_depolarization,
                                upwelling_scale):
        S = self._stokes(sky_depolarization, upwelling_scale)
        psi = np.arctan2(-S[..., 2], -S[..., 1])      # 2*phi in pss frame
        ok = self.valid & np.isfinite(psi)
        w = np.exp(-0.5 * (self.SA ** 2 / max(mss_along, 1e-12)
                           + self.SC ** 2 / max(mss_cross, 1e-12)))[ok]
        z = np.exp(1j * psi[ok])
        return float(np.abs(np.sum(w * z) / np.sum(w)))

    def infer_sky(self, s1, s2, S0=None, mss0=(0.02, 0.015),
                  qs=(0.10, 0.25, 0.75, 0.90), w_aop=3.0, n_outer=2,
                  t_grid=None, w_grid=None, sample=200_000, rng=None):
        """Minimal sky-condition inference from observed DoLP/AoP statistics.
        Grid-search (sky_depolarization, upwelling_scale); gain from the median DoLP; the
        slope-spread prior mss0 is refined by inverting a subsample.
        Returns dict(sky_depolarization, upwelling_scale, gain, misfit, mss_used)."""
        rng = rng or np.random.default_rng(0)
        t_grid = np.linspace(0.0, 1.0, 11) if t_grid is None else t_grid
        w_grid = (np.linspace(0.0, 2.0, 11) if w_grid is None else w_grid)
        if np.all(self.RW == 0.0):
            w_grid = np.array([self.upwelling_scale])

        flat1 = np.asarray(s1).ravel()
        flat2 = np.asarray(s2).ravel()
        flat0 = (np.ones_like(flat1) if S0 is None
                 else np.asarray(S0, dtype=float).ravel())
        fin = np.isfinite(flat1) & np.isfinite(flat2) & np.isfinite(flat0)
        flat1, flat2, flat0 = flat1[fin], flat2[fin], flat0[fin]
        if flat1.size > sample:
            idx = rng.choice(flat1.size, sample, replace=False)
            flat1, flat2, flat0 = flat1[idx], flat2[idx], flat0[idx]
        d = np.sqrt(flat1 ** 2 + flat2 ** 2)
        obs_ratio = np.log(np.quantile(d, qs) / np.median(d))
        psi_obs = np.arctan2(flat2, flat1)
        R_obs = float(np.abs(np.mean(np.exp(1j * psi_obs))))

        best = None
        mss = tuple(mss0)
        for outer in range(max(n_outer, 1)):
            best = None
            for t in t_grid:
                for w in w_grid:
                    pred_q = self.predicted_dolp_quantiles(
                        *mss, list(qs) + [0.5], sky_depolarization=t, upwelling_scale=w)
                    ratio = np.log(pred_q[:-1] / pred_q[-1])
                    R_pred = self._predicted_aop_strength(*mss, t, w)
                    misfit = float(np.sum((ratio - obs_ratio) ** 2)
                                   + w_aop * (R_pred - R_obs) ** 2)
                    if best is None or misfit < best["misfit"]:
                        best = dict(sky_depolarization=float(t), upwelling_scale=float(w),
                                    misfit=misfit)
            self.set_environment(best["sky_depolarization"], best["upwelling_scale"])
            best["gain"] = self.calibrate_gain(flat1, flat2, mss0=mss,
                                               rng=rng)
            if outer < n_outer - 1:
                sc, sa = self.invert(flat1, flat2, gain=best["gain"],
                                     S0=None if S0 is None else flat0,
                                     mss0=mss)
                if np.isfinite(sa).sum() > 1000:
                    mss = (float(np.nanvar(sa)), float(np.nanvar(sc)))
        best["mss_used"] = mss
        return best

    # ------------------------------------------------------------------
    def calibrate_gain(self, s1, s2, n_iter=3, mss0=(0.02, 0.015),
                       sample=200_000, rng=None):
        """Fixed-point polarimetric gain: match observed median DoLP to the
        ensemble prediction at the retrieved mss."""
        rng = rng or np.random.default_rng(0)
        d = np.sqrt(np.asarray(s1) ** 2 + np.asarray(s2) ** 2).ravel()
        d = d[np.isfinite(d)]
        if d.size > sample:
            d = rng.choice(d, sample, replace=False)
        obs = float(np.median(d))
        flat1 = np.asarray(s1).ravel()
        flat2 = np.asarray(s2).ravel()
        if flat1.size > sample:
            idx = rng.choice(flat1.size, sample, replace=False)
            flat1, flat2 = flat1[idx], flat2[idx]
        mss_a, mss_c = mss0
        g = self.predicted_median_dolp(mss_a, mss_c) / obs
        for _ in range(n_iter - 1):
            sc, sa = self.invert(flat1, flat2, gain=g)
            if np.isfinite(sa).sum() < 100:
                break
            mss_a = float(np.nanvar(sa))
            mss_c = float(np.nanvar(sc))
            g = self.predicted_median_dolp(mss_a, mss_c) / obs
        return float(g)


# ==========================================================================
# Sun geometry (ported from the dev tree; no seapol needed).
# ==========================================================================
def solar_position(dt_utc: _dt.datetime, lat: float, lon: float):
    """Approximate solar zenith and (compass, from-North CW) azimuth (deg)."""
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jdn = (dt_utc.day + (153 * m + 2) // 5 + 365 * y + y // 4
           - y // 100 + y // 400 - 32045)
    frac = (dt_utc.hour - 12) / 24 + dt_utc.minute / 1440 + dt_utc.second / 86400
    n = (jdn + frac) - 2451545.0
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 4e-7 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    lmst = np.deg2rad((gmst * 15 + lon) % 360)
    ha = lmst - ra
    latr = np.deg2rad(lat)
    alt = np.arcsin(np.sin(latr) * np.sin(dec)
                    + np.cos(latr) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(-np.sin(ha),
                    np.tan(dec) * np.cos(latr) - np.sin(latr) * np.cos(ha))
    return 90.0 - np.degrees(alt), np.degrees(az) % 360.0


def scene_azimuth_deg(camera_azimuth_compass_deg: float,
                      sun_azimuth_compass_deg: float) -> float:
    """Sun azimuth in seapol's scene frame (+x toward camera, CCW)."""
    return (camera_azimuth_compass_deg + 180.0 - sun_azimuth_compass_deg) % 360.0


# ==========================================================================
# Environment resolution + inverter construction.
# ==========================================================================
@dataclass
class SkyAwareEnv:
    """Per-acquisition sky/water environment for the inverter."""
    sky_depolarization: float
    upwelling_scale: float
    gain: float
    sun_zenith_deg: float | None = None
    sun_azimuth_scene_deg: float | None = None
    water_case: int | None = 2
    mss_used: tuple[float, float] = (0.02, 0.015)
    inferred: bool = False
    misfit: float = float("nan")


def build_skyaware_inverter(*, theta_v_deg: float, n_water: float,
                            env: SkyAwareEnv,
                            knn_backend: str = "auto") -> SkyAwareInverter:
    """Construct an inverter for a resolved environment (one per record).

    knn_backend selects the 3-D nearest-neighbor backend ('auto' = GPU torch if
    present else cKDTree; see SkyAwareInverter / _resolve_knn_device)."""
    require_seapol()
    water = WaterBody(case=env.water_case) if env.water_case else None
    inv = SkyAwareInverter(theta_v_deg=theta_v_deg, n_water=n_water,
                           sun_zenith_deg=env.sun_zenith_deg,
                           sun_azimuth_scene_deg=env.sun_azimuth_scene_deg,
                           water=water,
                           sky_depolarization=env.sky_depolarization, upwelling_scale=env.upwelling_scale,
                           knn_backend=knn_backend)
    inv.set_environment(env.sky_depolarization, env.upwelling_scale)
    return inv


# loose sanity bounds on a blind-inferred environment (low-sun infer_sky can
# rail the grid and produce an unphysical gain); outside these we warn.
_GAIN_BOUNDS = (0.2, 4.0)


def resolve_environment(s1_ref, s2_ref, S0_ref, *, theta_v_deg, n_water,
                        sun_zenith_deg, sun_azimuth_scene_deg, water_case=2,
                        env: SkyAwareEnv | None = None,
                        mss0=(0.02, 0.015), verbose=True) -> SkyAwareEnv:
    """Return a usable SkyAwareEnv. If `env` is supplied, use it. Otherwise
    blind-infer (sky_depolarization, upwelling_scale, gain) from the reference frame's Stokes
    via infer_sky, validating the gain against loose bounds (a railed low-sun
    fit is flagged but still returned -- the caller may override)."""
    require_seapol()
    if env is not None:
        return env
    probe = build_skyaware_inverter(
        theta_v_deg=theta_v_deg, n_water=n_water,
        env=SkyAwareEnv(sky_depolarization=0.0, upwelling_scale=1.0, gain=1.0,
                        sun_zenith_deg=sun_zenith_deg,
                        sun_azimuth_scene_deg=sun_azimuth_scene_deg,
                        water_case=water_case))
    fit = probe.infer_sky(np.asarray(s1_ref), np.asarray(s2_ref),
                          S0=np.asarray(S0_ref), mss0=mss0)
    railed = not (_GAIN_BOUNDS[0] <= fit["gain"] <= _GAIN_BOUNDS[1])
    if verbose:
        flag = "  [WARN: gain out of bounds -- low-sun/glint?]" if railed else ""
        print(f"    skyaware infer: sky_depolarization={fit['sky_depolarization']:.2f} "
              f"upwelling_scale={fit['upwelling_scale']:.2f} gain={fit['gain']:.3f} "
              f"misfit={fit['misfit']:.4f}{flag}")
    return SkyAwareEnv(
        sky_depolarization=fit["sky_depolarization"], upwelling_scale=fit["upwelling_scale"], gain=fit["gain"],
        sun_zenith_deg=sun_zenith_deg, sun_azimuth_scene_deg=sun_azimuth_scene_deg,
        water_case=water_case, mss_used=tuple(fit["mss_used"]),
        inferred=True, misfit=fit["misfit"])


# ==========================================================================
# Stack-level orchestrator: Stokes -> anchored sky-aware slope stack.
# ==========================================================================
def skyaware_slope_stack(S0, s1, s2, *, theta_v_deg, n_water=1.34,
                         env: SkyAwareEnv | None = None,
                         sun_zenith_deg=None, sun_azimuth_scene_deg=None,
                         water_case=2, gain_mode="empirical",
                         lab_gain=(1.2185, 1.2197), theta_i_mean_deg=None,
                         dolp_obs_median=None, ref_index=None,
                         lookup_table=None, sign_along=1.0, verbose=True):
    """Invert a Stokes stack with the sky-aware forward model and apply the
    empirical slope anchor. Reused by run_epss (after extracting Stokes from
    raw frames) and by Stokes-archive verification.

    Args:
        S0, s1, s2 : Stokes stacks (T, Ny, Nx) (or single (Ny, Nx)) in the pss
            frame (s1 = S1/S0, s2 = S2/S0).
        theta_v_deg : camera incidence (deg from nadir).
        env : precomputed SkyAwareEnv; if None it is blind-inferred from the
            stack-median reference frame (sun geometry then required).
        gain_mode : anchor amplitude target -- "none" (f=1), "empirical"
            (in-scene median DoLP), or "lab" (lab_gain). Selects which present
            pipeline produces the reference amplitude.
        sign_along : multiply the along-look (Sy) seapol output by this (+1/-1)
            to match the present-pipeline sign convention; verify once per
            geometry (see anchor docs).

    Returns:
        (slope_x, slope_y, env, anchor) : the f-scaled cross/along slope stacks,
        the resolved SkyAwareEnv, and the AnchorResult.
    """
    require_seapol()
    S0 = np.asarray(S0, float); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    if s1.ndim == 2:
        S0, s1, s2 = S0[None], s1[None], s2[None]
    T, Ny, Nx = s1.shape
    if theta_i_mean_deg is None:
        theta_i_mean_deg = theta_v_deg
    if lookup_table is None:
        lookup_table = build_lookup_table(n_water=n_water)

    if env is None:
        if ref_index is None:
            S0r = np.nanmedian(S0, 0); s1r = np.nanmedian(s1, 0); s2r = np.nanmedian(s2, 0)
        else:
            S0r, s1r, s2r = S0[ref_index], s1[ref_index], s2[ref_index]
        env = resolve_environment(
            s1r, s2r, S0r, theta_v_deg=theta_v_deg, n_water=n_water,
            sun_zenith_deg=sun_zenith_deg,
            sun_azimuth_scene_deg=sun_azimuth_scene_deg,
            water_case=water_case, verbose=verbose)
    inv = build_skyaware_inverter(theta_v_deg=theta_v_deg, n_water=n_water, env=env)

    do_present = gain_mode != "none"
    if do_present and gain_mode == "empirical" and dolp_obs_median is None:
        dref = np.sqrt(np.nanmedian(s1, 0) ** 2 + np.nanmedian(s2, 0) ** 2)
        dolp_obs_median = float(np.nanmedian(dref))

    sx = np.empty((T, Ny, Nx)); sy = np.empty((T, Ny, Nx))
    px = np.empty((T, Ny, Nx)) if do_present else None
    py = np.empty((T, Ny, Nx)) if do_present else None
    for t in range(T):
        sc, sa = inv.invert(s1[t], s2[t], gain=env.gain, S0=S0[t],
                            mss0=env.mss_used)
        sx[t] = sc
        sy[t] = sign_along * sa
        if do_present:
            a, b = present_slope_from_stokes(
                s1[t], s2[t], gain_mode=gain_mode, lab_gain=lab_gain,
                theta_i_mean_deg=theta_i_mean_deg, n_water=n_water,
                lookup_table=lookup_table, dolp_obs_median=dolp_obs_median)
            px[t] = a; py[t] = b
    anchor = slope_anchor_gain(sx, sy, px, py, mode=gain_mode)
    return anchor.f * sx, anchor.f * sy, env, anchor
