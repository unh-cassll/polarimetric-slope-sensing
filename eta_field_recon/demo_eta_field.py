"""
Self-contained demonstration of reconstruct_eta_field.

Synthesizes a realistic 3 m x 3 m slope-image stack at 10 Hz with several
wave components spanning the long-wave to short-wave regime, then runs
reconstruct_eta_field and plots reconstruction vs truth.

Run as a script:
    python3 demo_eta_field.py

Outputs:
    demo_eta_field.png  -- diagnostic plot
    prints reconstruction quality metrics

Computational cost: synthesis is the slow step (~2 min on a typical CPU
for a 256x256 x 1024 frame x 80 modes dataset).  Reconstruction itself
runs in ~10 seconds.
"""
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Allow running from inside eta_field_recon/ (python demo_eta_field.py)
# OR from the repo root (python eta_field_recon/demo_eta_field.py).
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eta_field_recon import reconstruct_eta_field


# ---------------------------------------------------------------------------
# Synthetic slope-field generator
# ---------------------------------------------------------------------------
def synth_slope_field(Lx, Ly, Nx, Ny, t, components,
                       water_depth=100.0, n_freq=80, n_dir=36, seed=42):
    """
    Generate slope_x, slope_y, eta on an Nx x Ny grid spanning
    [-Lx/2, Lx/2] x [-Ly/2, Ly/2] for every time in `t`.

    Each component in `components` is a dict:
        f_peak     : peak frequency (Hz)
        df         : Gaussian width in f (Hz)
        theta_deg  : mean direction (deg, east=0, north=90)
        spread_n   : cos^n directional exponent (4=wind sea, 10=swell)
        Hs         : significant wave height (m)

    Returns:
        slope_x, slope_y, eta : (T, Ny, Nx) float arrays
        x, y                  : (Nx,), (Ny,) coordinate vectors (m)
        f, S                  : (n_freq,), (n_freq, n_dir) for diagnostic plotting
    """
    rng = np.random.default_rng(seed)
    g = 9.81

    f = np.linspace(0.05, 3.0, n_freq)
    df = f[1] - f[0]
    th = np.linspace(-np.pi, np.pi, n_dir, endpoint=False)
    dth = th[1] - th[0]
    omega = 2*np.pi*f

    # Build S(f, theta)
    S = np.zeros((n_freq, n_dir))
    for cmp in components:
        Sf = np.exp(-0.5*((f - cmp['f_peak'])/cmp['df'])**2)
        Sf /= (Sf.sum() * df + 1e-30)
        Sf *= (cmp['Hs']/4.0)**2
        diff = np.angle(np.exp(1j*(th - np.deg2rad(cmp['theta_deg']))))
        D = np.where(np.abs(diff) < np.pi/2, np.cos(diff)**cmp['spread_n'], 0.0)
        D /= (D.sum() * dth + 1e-30)
        S += Sf[:, None] * D[None, :]
    A = np.sqrt(2.0 * S * df * dth)

    # Dispersion (deep-water-corrected via Newton)
    k = omega**2/g
    for _ in range(40):
        k = omega**2 / (g*np.tanh(k*water_depth))
    kx = k[:, None] * np.cos(th)[None, :]
    ky = k[:, None] * np.sin(th)[None, :]
    phi_rand = rng.uniform(0, 2*np.pi, (n_freq, n_dir))

    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')

    M = n_freq * n_dir
    kx_flat = kx.ravel().astype(np.float32)
    ky_flat = ky.ravel().astype(np.float32)
    A_flat = A.ravel().astype(np.float32)
    phi_flat = phi_rand.ravel().astype(np.float32)
    omega_flat = np.repeat(omega, n_dir).astype(np.float32)
    kxA = kx_flat * A_flat
    kyA = ky_flat * A_flat
    pix_x = X.ravel().astype(np.float32)
    pix_y = Y.ravel().astype(np.float32)

    T = len(t)
    eta_f = np.zeros((T, Ny, Nx), dtype=np.float32)
    slope_x = np.zeros((T, Ny, Nx), dtype=np.float32)
    slope_y = np.zeros((T, Ny, Nx), dtype=np.float32)

    # Vectorized chunked evaluation: cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
    chunk_time = 32
    for i0 in range(0, T, chunk_time):
        ts = t[i0:i0+chunk_time].astype(np.float32)
        Tc = len(ts)
        ct = np.cos(omega_flat[None, :] * ts[:, None])          # (Tc, M)
        st = np.sin(omega_flat[None, :] * ts[:, None])
        for py in range(Ny):
            rx = pix_x[py*Nx:(py+1)*Nx]
            ry = pix_y[py*Nx:(py+1)*Nx]
            psi_sp = (rx[:, None]*kx_flat[None, :]
                      + ry[:, None]*ky_flat[None, :]
                      + phi_flat[None, :])                       # (Nx, M)
            cs = np.cos(psi_sp); ss = np.sin(psi_sp)
            cs_A = cs * A_flat[None, :]
            ss_A = ss * A_flat[None, :]
            eta_row = cs_A @ ct.T + ss_A @ st.T                  # (Nx, Tc)
            cs_kxA = cs * kxA[None, :]; ss_kxA = ss * kxA[None, :]
            sx_row = -(ss_kxA @ ct.T - cs_kxA @ st.T)
            cs_kyA = cs * kyA[None, :]; ss_kyA = ss * kyA[None, :]
            sy_row = -(ss_kyA @ ct.T - cs_kyA @ st.T)
            eta_f[i0:i0+Tc, py, :] = eta_row.T
            slope_x[i0:i0+Tc, py, :] = sx_row.T
            slope_y[i0:i0+Tc, py, :] = sy_row.T

    return slope_x, slope_y, eta_f, x, y, f, S


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main():
    # Grid: 3 m frame at 12 mm pixels = 256x256 (production: 6 mm, 512x512;
    # we use 12 mm here so the demo runs in a reasonable time).
    Nx = Ny = 256
    dx = 0.012
    Lx, Ly = Nx*dx, Ny*dx
    fs = 10.0
    T = 1024
    dt = 1.0/fs
    t = np.arange(T)*dt

    print(f"Demo configuration:")
    print(f"  Frame: {Nx}x{Ny}, dx={dx*1000:.0f} mm -> {Lx:.2f} m square")
    print(f"  Time:  fs={fs} Hz, T={T}, duration={T*dt:.1f} s")

    # Wave components covering long/medium/short
    components = [
        dict(f_peak=0.15, df=0.02, theta_deg=  0.0, spread_n=10, Hs=0.30),  # swell
        dict(f_peak=0.50, df=0.05, theta_deg= 60.0, spread_n=4,  Hs=0.25),  # mid
        dict(f_peak=1.20, df=0.20, theta_deg=120.0, spread_n=4,  Hs=0.45),  # wind
        dict(f_peak=2.50, df=0.30, theta_deg= 30.0, spread_n=4,  Hs=0.20),  # short
    ]
    g = 9.81
    print("\nWave components:")
    for c in components:
        lam = g/(2*np.pi*c['f_peak']**2)
        print(f"  f={c['f_peak']:.2f} Hz, dir={c['theta_deg']:+4.0f} deg, "
              f"Hs={c['Hs']:.2f} m, lambda={lam:.2f} m (lambda/L={lam/Lx:.2f})")

    # Synthesize
    print("\nSynthesizing slope/eta field (this is the slow step)...")
    t0 = time.perf_counter()
    sx_field, sy_field, eta_field, x_pix, y_pix, f_synth, S_synth = synth_slope_field(
        Lx, Ly, Nx, Ny, t, components, water_depth=100.0, seed=42,
        n_freq=80, n_dir=36)
    print(f"  synthesis: {time.perf_counter()-t0:.1f} s")

    # Reconstruct
    print("\nRunning reconstruction (downsample=4 -> 64x64) ...")
    t0 = time.perf_counter()
    eta_xyt, eta_long, eta_short, conf, diag = reconstruct_eta_field(
        sx_field, sy_field, dx=dx, fs=fs,
        water_depth_m=100.0, downsample=4,
        spatial_alpha=0.1, spatial_pad_frac=0.10,
        temporal_window='tukey', temporal_alpha=0.25)
    print(f"  reconstruction: {time.perf_counter()-t0:.1f} s")
    print(f"  output: {eta_xyt.shape} ({eta_xyt.nbytes/1e6:.1f} MB)")

    # Downsampled truth for direct comparison
    ds = 4
    eta_truth_ds = eta_field[:, ::ds, ::ds]
    Ny_d, Nx_d = eta_truth_ds.shape[1], eta_truth_ds.shape[2]

    # Score
    print("\nReconstruction quality:")
    et = eta_truth_ds[:, Ny_d//2, Nx_d//2]
    er = eta_xyt[:, Ny_d//2, Nx_d//2]
    et0 = et - et.mean(); er0 = er - er.mean()
    print(f"  center time series: std(truth)={et.std():.4f} m, "
          f"std(recon)={er.std():.4f} m")
    print(f"    RMSE = {np.sqrt(np.mean((et0-er0)**2)):.4f} m, "
          f"corr = {np.corrcoef(et0, er0)[0,1]:+.4f}")

    cw = conf[:, Ny_d//2, Nx_d//2]
    et_w = et - np.average(et, weights=cw)
    er_w = er - np.average(er, weights=cw)
    rmse_w = np.sqrt(np.average((et_w - er_w)**2, weights=cw))
    cc_w = (np.average(et_w*er_w, weights=cw) /
            np.sqrt(np.average(et_w**2, weights=cw) *
                    np.average(er_w**2, weights=cw)))
    print(f"    confidence-weighted: RMSE = {rmse_w:.4f} m, corr = {cc_w:+.4f}")

    et_full = eta_truth_ds - eta_truth_ds.mean()
    er_full = eta_xyt - eta_xyt.mean()
    print(f"  full field        : RMSE = "
          f"{np.sqrt(np.mean((et_full - er_full)**2)):.4f} m, "
          f"corr = {np.corrcoef(et_full.ravel(), er_full.ravel())[0,1]:+.4f}")

    # ----- Plot -----
    plot_demo(eta_truth_ds, eta_xyt, eta_long, eta_short, conf, diag,
              fs, components, S_synth, f_synth)


def plot_demo(eta_truth, eta_xyt, eta_long, eta_short, conf, diag,
              fs, components, S_synth, f_synth):
    from scipy.signal import welch
    T, Ny_d, Nx_d = eta_truth.shape
    t = np.arange(T) / fs
    xd = diag['x_ds']; yd = diag['y_ds']

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.35)

    # Row 0: snapshot truth/recon/diff
    ti = T//2
    vmax = np.abs(eta_truth[ti]).max()
    for col, (lab, fld) in enumerate([
            ('eta truth',         eta_truth[ti]),
            ('eta reconstructed', eta_xyt[ti]),
            ('residual',          eta_xyt[ti] - eta_truth[ti])]):
        ax = fig.add_subplot(gs[0, col])
        vm = vmax if col < 2 else vmax/2
        im = ax.imshow(fld, extent=[xd[0], xd[-1], yd[0], yd[-1]],
                        origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax.set_title(f'{lab} @ t={ti/fs:.0f}s')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax)

    # Row 1 (full width): center pixel time series, first 30 s
    mask = t < 30
    ax = fig.add_subplot(gs[1, :])
    et = eta_truth[:, Ny_d//2, Nx_d//2]
    er = eta_xyt[:, Ny_d//2, Nx_d//2]
    el = eta_long
    ax.plot(t[mask], (et - et.mean())[mask], 'k', lw=1.2, label='truth')
    ax.plot(t[mask], (er - er.mean())[mask], '--', color='#A52A2A', lw=0.9,
            label='reconstruction')
    ax.plot(t[mask], (el - el.mean())[mask], ':', color='#2A52BE', lw=0.8,
            label='eta_long only (DC path)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('eta (m) at frame center')
    ax.set_title('center pixel: truth vs reconstruction (first 30 s)')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.4)

    # Row 2 col 0: PSD comparison
    ax = fig.add_subplot(gs[2, 0])
    nperseg = 256
    for sig, color, label, lw, ls in [
        (et - et.mean(), 'k', 'truth', 1.6, '-'),
        (er - er.mean(), '#A52A2A', 'recon (long+short)', 1.0, '--'),
        (el - el.mean(), '#2A52BE', 'long only', 1.0, ':'),
        (eta_short[:, Ny_d//2, Nx_d//2]
         - eta_short[:, Ny_d//2, Nx_d//2].mean(), '#4C9F39', 'short only',
         1.0, ':')]:
        ff, P = welch(sig, fs=fs, nperseg=nperseg)
        ax.loglog(ff, P, color=color, label=label, lw=lw, linestyle=ls)
    for c in components:
        ax.axvline(c['f_peak'], color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('freq (Hz)'); ax.set_ylabel('PSD (m^2/Hz)')
    ax.set_title('PSD of eta at frame center')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    # Row 2 col 1: spatial std vs time
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, eta_truth.std(axis=(1, 2)), 'k', lw=1.0, label='truth')
    ax.plot(t, eta_xyt.std(axis=(1, 2)), '--', color='#A52A2A', lw=1.0,
            label='reconstruction')
    ax.set_xlabel('t (s)'); ax.set_ylabel('spatial std of eta (m)')
    ax.set_title('Spatial std of eta(x,y) vs time')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    # Row 2 col 2: confidence mask snapshot
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(conf[T//2], extent=[xd[0], xd[-1], yd[0], yd[-1]],
                    origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'confidence(x,y) @ t={T//(2*fs):.0f}s')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('demo_eta_field.png', dpi=120)
    print("\nSaved demo_eta_field.png")


if __name__ == '__main__':
    main()
