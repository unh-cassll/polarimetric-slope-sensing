function mask = circular_aperture_mask(Ny, Nx, dx, diameter_m)
% Centered circular aperture mask of physical diameter diameter_m (meters)
% on a (Ny,Nx) grid of spacing dx. Empty diameter -> all-true (full frame).
% Cell centers within diameter_m/2 of the grid center are true; a disc that
% overflows the frame is clipped by the frame edges.
if isempty(diameter_m)
    mask = true(Ny, Nx);
    return
end
if diameter_m <= 0
    error('etaFieldRecon:badDiameter', ...
        'aperture_diameter_m must be positive (or [] for full frame); got %g', ...
        diameter_m);
end
yc = ((0:Ny-1)' - (Ny - 1)/2) * dx;
xc = ((0:Nx-1) - (Nx - 1)/2) * dx;
[XX, YY] = meshgrid(xc, yc);
r = sqrt(XX.^2 + YY.^2);
mask = r <= (diameter_m / 2);
if ~any(mask(:))
    error('etaFieldRecon:emptyAperture', ...
        ['aperture_diameter_m=%g selects no grid cells at dx=%g on a ' ...
         '%dx%d grid; use a diameter >= the cell spacing'], ...
        diameter_m, dx, Ny, Nx);
end
end
