function H = aperture_transfer_function(k, diameter_m, shape)
% Amplitude transfer of the spatial-mean slope over a measurement aperture.
%
% Circular disc of diameter D: isotropic jinc H(k) = 2*J1(kR)/(kR), H(0)=1,
% first null at kR = 3.832. Square frame of side L: RMS azimuthal average of
% sinc(kx*L/2)*sinc(ky*L/2). NaN k yields NaN H. Port of
% eta_field_recon.wavelet_core.aperture_transfer_function.
%
%   k          : scalar or array of wavenumber magnitudes (rad/m)
%   diameter_m : disc diameter (circular) or frame side L (square), meters
%   shape      : 'circular' (default) or 'square'
%
% Returns H, same size as k, amplitude transfer <= 1.

arguments
    k double
    diameter_m
    shape (1, :) char = 'circular'
end

if isempty(diameter_m) || diameter_m <= 0
    error('etaFieldRecon:badDiameter', 'diameter_m must be positive');
end

switch shape
    case 'circular'
        x = k .* (diameter_m / 2);
        H = NaN(size(x));
        lo = isfinite(x) & (x <= 1e-9);
        hi = isfinite(x) & (x > 1e-9);
        H(lo) = 1;
        H(hi) = 2 * besselj(1, x(hi)) ./ x(hi);
    case 'square'
        L = double(diameter_m);
        th = linspace(0, pi/2, 64);              % quarter-symmetry
        kx = k(:) * cos(th);
        ky = k(:) * sin(th);
        H2 = mean((nsinc(kx*L/(2*pi)) .* nsinc(ky*L/(2*pi))).^2, 2);
        H = reshape(sqrt(H2), size(k));
    otherwise
        error('etaFieldRecon:badShape', 'unknown aperture shape ''%s''', shape);
end
end

function s = nsinc(x)
% Normalized sinc sin(pi*x)/(pi*x) with s(0)=1 (numpy convention).
s = ones(size(x));
nz = (x ~= 0);
s(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end
