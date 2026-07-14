function eta = invert_mean_slope_series(sx_mean, sy_mean, fs, opts)
% Long-wave elevation eta_long(t) from a spatial-mean slope series.
%
% Single-point form of the long-wave path in reconstruct_eta_field: the
% directionally-complete direct amplitude sqrt(|Sx|^2+|Sy|^2)/k carries the
% magnitude (no aperture jinc -- the input is already a spatial mean) and
% the per-frequency signed projection carries the phase. Fourier method
% only; the 'wavelet' phase source is EWDM-specific and not ported.
% Port of eta_field_recon.recon.invert_mean_slope_series.
%
%   sx_mean, sy_mean : (T,1) cross-look / along-look spatial-mean slope
%   fs               : sample rate (Hz)
%
% Name-value options: water_depth_m (100), long_wave_method ('fourier'),
% hp_fmin (0.08), temporal_alpha (0.25).
%
% Returns eta, a zero-mean (T,1) elevation series (m), up-positive.

arguments
    sx_mean double
    sy_mean double
    fs (1, 1) double
    opts.water_depth_m (1, 1) double = 100.0
    opts.long_wave_method (1, :) char = 'fourier'
    opts.hp_fmin (1, 1) double = 0.08
    opts.temporal_alpha (1, 1) double = 0.25
end

if ~strcmpi(opts.long_wave_method, 'fourier')
    error('etaFieldRecon:notPorted', ...
        ['long_wave_method ''%s'' is not ported (EWDM/CWT-specific); ' ...
         'use ''fourier'' or the Python implementation.'], ...
        opts.long_wave_method);
end

[A, Sx, Sy, T] = direct_complete_amplitude(sx_mean, sy_mean, fs, ...
    opts.water_depth_m, [], false, opts.hp_fmin, 0.25, ...
    opts.temporal_alpha, 'tukey');
m = sqrt(abs(Sx).^2 + abs(Sy).^2) + 1e-30;
rel = sign(real(Sy .* conj(Sx)));
rel(rel == 0) = 1;
phase = angle(1i * ((abs(Sx) ./ m) .* Sx + (abs(Sy) ./ m) .* rel .* Sy));
eta = irfft_half(A .* exp(1i * phase), T);
eta = eta - mean(eta);
end
