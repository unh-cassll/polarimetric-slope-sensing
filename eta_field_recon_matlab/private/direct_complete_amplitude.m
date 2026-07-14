function [A, Sx, Sy, T, win] = direct_complete_amplitude(sx_mean, sy_mean, ...
    fs, depth, diameter_m, jinc, hp_fmin, hp_width_oct, temporal_alpha, ...
    temporal_window)
% rfft-grid directionally-complete long-wave amplitude
% A(f) = sqrt(|Sx|^2 + |Sy|^2)/k (Phillips 1977), jinc aperture-corrected
% and logistic high-passed. Sx, Sy are the windowed disc-mean slope rffts
% and win the temporal window applied. Port of
% eta_field_recon.recon._direct_complete_amplitude.
sE = detrend(double(sx_mean(:)));
sN = detrend(double(sy_mean(:)));
T = numel(sE);
win = temporal_window_vec(T, temporal_window, temporal_alpha);
wn = sqrt(mean(win.^2));
f = (0:floor(T/2))' * (fs / T);
[~, k] = lindisp_with_current(2*pi*f, depth, 0.0);
Sx = rfft_half(sE .* win) / wn;
Sy = rfft_half(sN .* win) / wn;
m = sqrt(abs(Sx).^2 + abs(Sy).^2) + 1e-30;
r = m ./ k;
A = zeros(size(r));
fin = isfinite(r);
A(fin) = r(fin);
if jinc && ~isempty(diameter_m)
    ws = warning('off', 'etaFieldRecon:apertureNull');   % null bands expected
    g = aperture_transfer_gain(f, k, diameter_m, 'circular', 0.3);
    warning(ws);
    g(~isfinite(g)) = 0;
    A = A .* g;
end
lr = (log2(max(f, 1e-12)) - log2(hp_fmin)) / hp_width_oct;
A = A .* min(max(1 ./ (1 + exp(-lr)), 0), 1);
end
