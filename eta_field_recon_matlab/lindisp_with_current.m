function [c, k] = lindisp_with_current(omega, h, current_m_s)
% Linear water-wave dispersion relation with surface tension and current.
%
% Solves omega = sqrt((g*k + sigma/rho*k^3)*tanh(k*h)) + k*U for k(omega)
% by tabulated cubic-spline interpolation. Port of
% eta_field_recon.wavelet_core.lindisp_with_current.
%
%   omega       : scalar or array of angular frequencies (rad/s)
%   h           : water depth (m)
%   current_m_s : steady current projected onto the wave direction (m/s)
%
% Returns column vectors on the flattened omega:
%   c : phase speed omega/k (m/s)
%   k : wavenumber (rad/m); NaN for omega==0 and out-of-range omega

omega = double(omega(:));
h = double(h(1));
U = double(current_m_s(1));
omega(omega == 0) = NaN;

g = 9.806; rho_w = 1020.0; sigma = 0.072;
k_vec = logspace(-4, 4, 200)';
omega_disp = sqrt((g*k_vec + sigma/rho_w*k_vec.^3) .* tanh(k_vec*h)) + k_vec*U;

% Opposing current (U<0) makes omega(k) non-monotonic: keep the leading
% increasing branch; omega above the blocking maximum returns NaN k.
inc = diff(omega_disp) > 0;
if ~all(inc)
    cut = find(~inc, 1);
    warning('etaFieldRecon:dispersionNonMonotonic', ...
        ['dispersion omega(k) non-monotonic (opposing current U=%.2f m/s ' ...
         'blocks waves above %.3f rad/s); using the increasing branch, ' ...
         'higher omega -> NaN k.'], U, max(omega_disp(1:cut)));
    k_vec = k_vec(1:cut);
    omega_disp = omega_disp(1:cut);
end
if numel(k_vec) < 2
    % Supercritical opposing current: no increasing branch to invert.
    c = NaN(size(omega));
    k = NaN(size(omega));
    return
end

if numel(k_vec) >= 4
    k = interp1(omega_disp, k_vec, omega, 'spline', NaN);
else
    k = interp1(omega_disp, k_vec, omega, 'linear', NaN);
end
c = omega ./ k;
end
