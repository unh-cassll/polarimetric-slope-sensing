function [eta_xyt, eta_long, eta_short, confidence, diagn] = reconstruct_eta_field(slope_x_field, slope_y_field, dx, fs, opts)
% Reconstruct eta(x,y,t) from a stack of slope images.
%
% MATLAB port of eta_field_recon.recon.reconstruct_eta_field (Python).
% Combines a per-frame Harker-O'Leary g2s integration of slope (short-wave
% SHAPE, zero spatial mean per frame) with a directionally-complete Fourier
% slope projection of the disc-mean slope (long-wave "integration constant"
% eta_long(t)):
%
%   eta(x,y,t) = eta_short(x,y,t) + eta_long(t)
%
% ARRAY CONVENTION: slope stacks are (Ny, Nx, T) -- frames along dim 3
% (MATLAB-native), NOT the Python (T, Ny, Nx). permute(a, [2 3 1]) converts
% a Python-ordered array loaded from .mat.
%
% DEPENDENCIES: g2s + dopDiffLocal (Grad2Surf and DOPbox, Harker & O'Leary)
% are required for the short-wave step: either addpath both packages before
% calling, or pass their locations via the grad2surf_dir/dopbox_dir options.
% The 'wavelet' long-wave method is EWDM-specific and not ported; only
% 'fourier' (the Python default) is supported.
%
%   slope_x_field, slope_y_field : (Ny,Nx,T) slope stacks (rad)
%   dx : pixel size (m), square pixels
%   fs : frame rate (Hz)
%
% Name-value options (defaults match the Python source):
%   water_depth_m       (100.0)  depth for the dispersion relation
%   downsample          (4)      spatial subsample factor
%   long_wave_method    ('fourier')
%   spatial_alpha       (0.1)    Tukey alpha for the 2-D slope window
%   spatial_pad_frac    (0.10)   reflection-padding fraction per axis
%   temporal_window     ('tukey')  'tukey' | 'hann' | 'rect'
%   temporal_alpha      (0.25)   Tukey alpha when kind='tukey'
%   long_wave           (true)   run the long-wave projection
%   short_wave          (true)   run the per-frame g2s integration
%   aperture_diameter_m ([])     centered disc (m) for the mean slope;
%                                [] = full frame
%   verbose             (true)
%   grad2surf_dir       ('')     Grad2Surf package folder to addpath
%   dopbox_dir          ('')     DOPbox package folder to addpath
%
% Returns:
%   eta_xyt    : (Ny_d,Nx_d,T) elevation field, [] if short_wave=false
%   eta_long   : (T,1) long-wave series (zeros if long_wave=false)
%   eta_short  : (Ny_d,Nx_d,T) zero-mean-per-frame short-wave field, or []
%   confidence : (Ny_d,Nx_d,T) on [0,1]: spatial_W x temporal_W
%   diagn      : struct of intermediates (disc-mean slope series, windows,
%                aperture mask, coordinate vectors, pad sizes)

arguments
    slope_x_field (:, :, :) double
    slope_y_field (:, :, :) double
    dx (1, 1) double
    fs (1, 1) double
    opts.water_depth_m (1, 1) double = 100.0
    opts.downsample (1, 1) double = 4
    opts.long_wave_method (1, :) char = 'fourier'
    opts.spatial_alpha (1, 1) double = 0.1
    opts.spatial_pad_frac (1, 1) double = 0.10
    opts.temporal_window = 'tukey'
    opts.temporal_alpha (1, 1) double = 0.25
    opts.long_wave (1, 1) logical = true
    opts.short_wave (1, 1) logical = true
    opts.aperture_diameter_m double = []
    opts.verbose (1, 1) logical = true
    opts.grad2surf_dir char = ''
    opts.dopbox_dir char = ''
end

if ~strcmpi(opts.long_wave_method, 'fourier')
    error('etaFieldRecon:notPorted', ...
        ['long_wave_method ''%s'' is not ported (EWDM/CWT-specific); ' ...
         'use ''fourier'' or the Python implementation.'], ...
        opts.long_wave_method);
end

[Ny, Nx, T] = size(slope_x_field);
ds = opts.downsample;
sx_ds = slope_x_field(1:ds:end, 1:ds:end, :);
sy_ds = slope_y_field(1:ds:end, 1:ds:end, :);
[Ny_d, Nx_d, ~] = size(sx_ds);
dx_ds = dx * ds;

x_ds = ((0:Nx_d-1)' - Nx_d/2) * dx_ds;
y_ds = ((0:Ny_d-1)' - Ny_d/2) * dx_ds;

% Centered circular aperture over which the spatial-mean slope is formed
% for the long-wave inversion; [] -> full frame (all-true mask).
aperture_mask = circular_aperture_mask(Ny_d, Nx_d, dx_ds, opts.aperture_diameter_m);

if opts.verbose
    fprintf('  reconstruct_eta_field:\n');
    fprintf('    input : %d frames of %dx%d, dx=%.1f mm\n', T, Ny, Nx, dx*1000);
    fprintf('    output: %d frames of %dx%d, dx=%.1f mm (%.2f x %.2f m)\n', ...
        T, Ny_d, Nx_d, dx_ds*1000, Ny_d*dx_ds, Nx_d*dx_ds);
    fprintf('    spatial : Tukey alpha=%g, reflection pad frac=%g\n', ...
        opts.spatial_alpha, opts.spatial_pad_frac);
    if strcmpi(char(opts.temporal_window), 'tukey')
        fprintf('    temporal: ''%s''/alpha=%g\n', char(opts.temporal_window), ...
            opts.temporal_alpha);
    else
        fprintf('    temporal: ''%s''\n', char(opts.temporal_window));
    end
    if isempty(opts.aperture_diameter_m)
        fprintf('    aperture: full frame (%d cells)\n', sum(aperture_mask(:)));
    else
        fprintf('    aperture: circular D=%.3f m (%d/%d cells)\n', ...
            opts.aperture_diameter_m, sum(aperture_mask(:)), Ny_d*Nx_d);
    end
end

% ----------------------------------------------------------------------
% eta_short(x,y,t): per-frame Harker-O'Leary integration with reflection
% padding + Tukey taper.
% ----------------------------------------------------------------------
pad_y = round(Ny_d * opts.spatial_pad_frac);
pad_x = round(Nx_d * opts.spatial_pad_frac);
Ny_p = Ny_d + 2*pad_y;
Nx_p = Nx_d + 2*pad_x;

x_p = ((0:Nx_p-1)' - Nx_p/2) * dx_ds;
y_p = ((0:Ny_p-1)' - Ny_p/2) * dx_ds;

spatial_W_padded = tukey_win(Ny_p, opts.spatial_alpha) * tukey_win(Nx_p, opts.spatial_alpha)';
spatial_W = spatial_W_padded(pad_y+1:pad_y+Ny_d, pad_x+1:pad_x+Nx_d);

if opts.short_wave
    if ~isempty(opts.grad2surf_dir), addpath(validated_dir(opts.grad2surf_dir)); end
    if ~isempty(opts.dopbox_dir), addpath(validated_dir(opts.dopbox_dir)); end
    if exist('g2s', 'file') ~= 2 || exist('dopDiffLocal', 'file') ~= 2
        error('etaFieldRecon:missingG2S', ...
            ['g2s/dopDiffLocal not found: pass grad2surf_dir/dopbox_dir or ' ...
             'addpath the Grad2Surf and DOPbox packages (Harker & O''Leary) ' ...
             'before calling with short_wave=true.']);
    end
    eta_short = zeros(Ny_d, Nx_d, T);
    if opts.verbose
        fprintf('    integrating slope -> eta_short per frame (padded %dx%d) ...\n', ...
            Ny_p, Nx_p);
    end
    for ti = 1:T
        sx_pw = pad_reflect(sx_ds(:, :, ti), pad_y, pad_x) .* spatial_W_padded;
        sy_pw = pad_reflect(sy_ds(:, :, ti), pad_y, pad_x) .* spatial_W_padded;
        eta_p = g2s(sx_pw, sy_pw, x_p, y_p);
        eta_c = eta_p(pad_y+1:pad_y+Ny_d, pad_x+1:pad_x+Nx_d) .* spatial_W;
        eta_short(:, :, ti) = eta_c - mean(eta_c(:));
    end
else
    eta_short = [];
    if opts.verbose
        fprintf(['    short_wave=false: skipping per-frame g2s integration ' ...
                 '(eta_short and eta_xyt are []).\n']);
    end
end

% ----------------------------------------------------------------------
% eta_long(t): directionally-complete Fourier slope projection of the
% disc-mean slopes. Skipped (zeros) when long_wave=false.
% ----------------------------------------------------------------------
temporal_W = temporal_window_vec(T, opts.temporal_window, opts.temporal_alpha);
sx_mean = aperture_spatial_mean(sx_ds, aperture_mask);
sy_mean = aperture_spatial_mean(sy_ds, aperture_mask);

if opts.long_wave
    if opts.verbose
        fprintf('    computing eta_long(t) from disc-mean slopes (%s) ...\n', ...
            opts.long_wave_method);
    end
    eta_long = fourier_slope_projection(sx_ds, sy_ds, dx_ds, fs, ...
        opts.water_depth_m, ...
        aperture_diameter_m=opts.aperture_diameter_m, ...
        temporal_window=opts.temporal_window, ...
        temporal_alpha=opts.temporal_alpha);
else
    if opts.verbose
        fprintf('    long_wave=false: skipping eta_long(t) (set to zero).\n');
    end
    eta_long = zeros(T, 1);
end

% ----------------------------------------------------------------------
% Combine and confidence mask
% ----------------------------------------------------------------------
if opts.short_wave
    eta_xyt = eta_short + reshape(eta_long, 1, 1, []);
else
    eta_xyt = [];
end
confidence = spatial_W .* reshape(temporal_W, 1, 1, []);

diagn = struct( ...
    'long_wave', opts.long_wave, 'long_wave_method', opts.long_wave_method, ...
    'sx_mean', sx_mean, 'sy_mean', sy_mean, ...
    'x_ds', x_ds, 'y_ds', y_ds, 'dx_ds', dx_ds, ...
    'spatial_W', spatial_W, 'spatial_W_padded', spatial_W_padded, ...
    'temporal_W', temporal_W, ...
    'aperture_mask', aperture_mask, ...
    'aperture_diameter_m', opts.aperture_diameter_m, ...
    'pad_y', pad_y, 'pad_x', pad_x);
end


function d = validated_dir(d)
if ~isfolder(d)
    error('etaFieldRecon:badPath', 'not a directory: %s', d);
end
end
