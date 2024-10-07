% Given angles (in degrees) corresponding to right-handed rotations
% about x, y, and z, produces rotated vector as output
%
% Inputs may be multi-dimensional arrays
%
% Typed up by N. Laxague, from the classic Goldstein [1950] physics text
%
function [out_x,out_y,out_z] = rot3d(in_x,in_y,in_z,ang_x,ang_y,ang_z)

% RxRyRz = [A B C; D E F; G H I];

A = cosd(ang_y).*cosd(ang_z);
B = sind(ang_y).*sind(ang_x).*cosd(ang_z) - cosd(ang_x).*sind(ang_z);
C = sind(ang_z).*sind(ang_x) + sind(ang_y).*cosd(ang_x).*cosd(ang_z);

D = cosd(ang_y).*sind(ang_z);
E = cosd(ang_x).*cosd(ang_z) + sind(ang_x).*sind(ang_y).*sind(ang_z);
F = -sind(ang_x).*cosd(ang_z) + cosd(ang_x).*sind(ang_z).*sind(ang_y);

G = -sind(ang_y);
H = cosd(ang_y).*sind(ang_x);
I = cosd(ang_y).*cosd(ang_x);

out_x = in_x.*A + in_y.*B + in_z.*C;
out_y = in_x.*D + in_y.*E + in_z.*F;
out_z = in_x.*G + in_y.*H + in_z.*I;

