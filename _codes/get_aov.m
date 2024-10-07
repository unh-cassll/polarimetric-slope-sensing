% Given sensor size n_h by n_v, pixel pitch in microns, and focal length,
% returns horizontal and vertical angles of view aov_h and aov_v
% Original code by Scott Brown
% Used to generate function by Nathan Laxague 2020
%
function [aov_h,aov_v] = get_aov(n_h,n_v,pixp_microns,flen_mm)

% Convert pixel pitch to millimeters
pixp_mm = pixp_microns/1000;

% Compute angles of view in degrees
aov_h = 2*atand(pixp_mm*n_h*0.5./flen_mm);
aov_v = 2*atand(pixp_mm*n_v*0.5./flen_mm);