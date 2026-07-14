function eta = fourier_slope_projection(slope_x_field, slope_y_field, dx, fs, depth, opts)
% Long-wave eta(t) by per-frequency signed slope projection.
%
% Disc-mean slope rffts. Per frequency: direction cos = |Sx|/m,
% sin = (|Sy|/m)*sign(Re(Sy conj Sx)) (180-deg ambiguity from the channels'
% relative phase); the projection carries only the phase, the
% directionally-complete direct amplitude the magnitude:
% eta = irfft(A * exp(i*angle(+1i*(cos*Sx + sin*Sy)))). Fourier-amplitude
% form of the directional estimator of Krogstad, Magnusson & Donelan (2006).
% Port of eta_field_recon.recon.fourier_slope_projection.
%
%   slope_x_field, slope_y_field : (Ny,Nx,T) slope stacks (rad)
%   dx, fs, depth                : grid spacing (m), frame rate (Hz), depth (m)
%
% Name-value options (defaults match the Python source):
%   aperture_diameter_m ([]=full frame), jinc (true), hp_fmin (0.08),
%   hp_width_oct (0.25), temporal_alpha (0.25), temporal_window ('tukey')
%
% Returns eta, a zero-mean (T,1) elevation series (m).

arguments
    slope_x_field (:, :, :) double
    slope_y_field (:, :, :) double
    dx (1, 1) double
    fs (1, 1) double
    depth (1, 1) double
    opts.aperture_diameter_m double = []
    opts.jinc (1, 1) logical = true
    opts.hp_fmin (1, 1) double = 0.08
    opts.hp_width_oct (1, 1) double = 0.25
    opts.temporal_alpha (1, 1) double = 0.25
    opts.temporal_window = 'tukey'
end

[sxm, sym_, diam] = aperture_disc(slope_x_field, slope_y_field, dx, ...
    opts.aperture_diameter_m);
[A, Sx, Sy, T] = direct_complete_amplitude(sxm, sym_, fs, depth, diam, ...
    opts.jinc, opts.hp_fmin, opts.hp_width_oct, opts.temporal_alpha, ...
    opts.temporal_window);
m = sqrt(abs(Sx).^2 + abs(Sy).^2) + 1e-30;
rel = sign(real(Sy .* conj(Sx)));
rel(rel == 0) = 1;
carrier = 1i * ((abs(Sx) ./ m) .* Sx + (abs(Sy) ./ m) .* rel .* Sy);
eta = irfft_half(A .* exp(1i * angle(carrier)), T);
eta = eta - mean(eta);
end
