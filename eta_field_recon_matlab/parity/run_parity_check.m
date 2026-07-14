function run_parity_check(case_file, grad2surf_dir, dopbox_dir)
% End-to-end parity check of the MATLAB eta_field_recon port against the
% Python reference produced by make_parity_reference.py. Loads the same
% input slope fields, runs reconstruct_eta_field with the same options,
% compares the elevation arrays, writes a comparison figure and a .mat of
% the MATLAB outputs next to the case file. Errors (nonzero -batch exit) if
% any relative difference exceeds 1e-6.
%
% All arguments are required; grad2surf_dir and dopbox_dir point at local
% copies of the Harker & O'Leary Grad2Surf and DOPbox packages.

arguments
    case_file (1, :) char
    grad2surf_dir (1, :) char
    dopbox_dir (1, :) char
end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));

S = load(case_file);

% Python arrays are (T, Ny, Nx); the port is MATLAB-native (Ny, Nx, T).
sx = permute(S.slope_x_field, [2 3 1]);
sy = permute(S.slope_y_field, [2 3 1]);

tic;
[eta_xyt, eta_long, eta_short, confidence] = reconstruct_eta_field( ...
    sx, sy, S.dx, S.fs, ...
    water_depth_m=S.water_depth_m, downsample=double(S.downsample), ...
    grad2surf_dir=grad2surf_dir, dopbox_dir=dopbox_dir, verbose=true);
fprintf('  matlab reconstruction: %.1f s\n', toc);

fprintf('\n  parity vs Python reference (max|diff|, rel to std of field):\n');
tol = 1e-6;
ok = true;
ok = compare('eta_xyt', permute(eta_xyt, [3 1 2]), S.eta_xyt_py, tol) && ok;
ok = compare('eta_short', permute(eta_short, [3 1 2]), S.eta_short_py, tol) && ok;
ok = compare('eta_long', eta_long, S.eta_long_py, tol) && ok;
ok = compare('confidence', permute(confidence, [3 1 2]), S.confidence_py, tol) && ok;

[case_dir, ~, ~] = fileparts(case_file);
if isempty(case_dir), case_dir = '.'; end
out_mat = fullfile(case_dir, 'parity_matlab_out.mat');
save(out_mat, 'eta_xyt', 'eta_long', 'eta_short', 'confidence', '-v7');
fprintf('  wrote %s\n', out_mat);

write_figure(S, eta_xyt, eta_long, fullfile(case_dir, 'parity_comparison.png'));

if ok
    fprintf('\n  PASS: all fields match within rel tol %g\n', tol);
else
    error('etaFieldRecon:parityFail', 'parity check FAILED (rel tol %g)', tol);
end
end


function ok = compare(name, ml, py, tol)
d = max(abs(ml(:) - py(:)));
s = std(py(:));
if s > 0
    rel = d / s;
else
    rel = d;
end
ok = rel < tol;
status = 'ok';
if ~ok, status = 'FAIL'; end
fprintf('    %-11s max|diff| = %.3e   rel = %.3e   [%s]\n', name, d, rel, status);
end


function write_figure(S, eta_xyt, eta_long, png_path)
[~, ~, T] = size(eta_xyt);
ti = round(T/2);
py_frame = squeeze(S.eta_xyt_py(ti, :, :));
ml_frame = eta_xyt(:, :, ti);
t = (0:T-1)' / S.fs;

fig = figure('Visible', 'off', 'Position', [100 100 1250 640]);
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

cl = max(abs(py_frame(:))) * [-1 1];
nexttile(tl);
imagesc(py_frame, cl); axis image; colorbar;
title(sprintf('Python \\eta(x,y) frame %d (m)', ti));

nexttile(tl);
imagesc(ml_frame, cl); axis image; colorbar;
title('MATLAB \eta(x,y) same frame (m)');

nexttile(tl);
imagesc(abs(ml_frame - py_frame)); axis image; colorbar;
title('|MATLAB - Python| (m)');

nexttile(tl, [1 3]);
plot(t, S.eta_long_py, 'LineWidth', 2.5, 'Color', [0.3 0.5 0.9]); hold on;
plot(t, eta_long, '--', 'LineWidth', 1.2, 'Color', [0.9 0.3 0.2]);
grid on; xlabel('t (s)'); ylabel('\eta_{long} (m)');
legend('Python', 'MATLAB', 'Location', 'best');
title(sprintf('\\eta_{long}(t): max|diff| = %.2e m', ...
    max(abs(eta_long - S.eta_long_py))));

exportgraphics(fig, png_path, 'Resolution', 150);
close(fig);
fprintf('  wrote %s\n', png_path);
end
