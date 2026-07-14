function [sxm, sym_, diam] = aperture_disc(slope_x_field, slope_y_field, dx, aperture_diameter_m)
% Disc-mean slope series and the disc's physical diameter (full frame if []).
% slope fields are (Ny,Nx,T); returns (T,1) series.
[Ny, Nx, ~] = size(slope_x_field);
mask = circular_aperture_mask(Ny, Nx, dx, aperture_diameter_m);
sxm = aperture_spatial_mean(slope_x_field, mask);
sym_ = aperture_spatial_mean(slope_y_field, mask);
if isempty(aperture_diameter_m)
    diam = sqrt(Ny * Nx) * dx;
else
    diam = aperture_diameter_m;
end
end
