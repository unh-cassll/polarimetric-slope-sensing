function gain = aperture_transfer_gain(freqs, k_disp, diameter_m, shape, min_transfer)
% Per-frequency correction 1/H(k(f)) undoing aperture low-passing.
%
% Bands where |H| < min_transfer (at/beyond the aperture null) return NaN
% with a warning -- nulled energy cannot be recovered. Port of
% eta_field_recon.wavelet_core.aperture_transfer_gain.
%
%   freqs        : (nf,1) frequency grid (Hz); used only for the message
%   k_disp       : (nf,1) dispersion wavenumber on freqs (rad/m)
%   diameter_m   : aperture diameter (circular) or frame side (square), m
%   shape        : 'circular' (default) or 'square'
%   min_transfer : smallest |H| inverted (default 0.3)

arguments
    freqs double
    k_disp double
    diameter_m
    shape (1, :) char = 'circular'
    min_transfer (1, 1) double = 0.3
end

H = aperture_transfer_function(double(k_disp), diameter_m, shape);
gain = NaN(size(H));
ok = isfinite(H) & (abs(H) >= min_transfer);
gain(ok) = 1 ./ H(ok);

n_bad = sum(~isfinite(gain(:)));
if n_bad > 0
    warning('etaFieldRecon:apertureNull', ...
        ['%d/%d band(s) at/beyond the aperture transfer null (|H| < %g); ' ...
         'returning NaN (nulled energy is unrecoverable).'], ...
        n_bad, numel(gain), min_transfer);
end
end
