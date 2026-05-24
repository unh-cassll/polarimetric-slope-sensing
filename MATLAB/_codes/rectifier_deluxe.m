% Rectifies camera frame based on imaging properties, incorporating
% the camera elevation and attitude
%
% The principle: treat the corners of each unrectified frame as vectors
% originating from the camera lens. Apply rotations corresponding to
% pitching, rolling, and yawing
%
% v1 by Nathan Laxague
% between 2015-2021
%
% v2 by Goksu Duvarci 2023
% *updated to ingest pitch/roll/yaw angles
%
% v3 by Nathan Laxague 2024
% *corrected erroneous interaction between roll and yaw
% *re-scaled output array to keep it from getting too large
%   (enforcing new m_per_px)

% VectorNav Coordinate system
% x (roll)              
% ^              
% |              
% |              
% |              
% |              
% |              
% x - - - - - - > y (pitch)
% z (down)
%
% x (sensor looking through x-direction) so x direction is vertical, y
% direction is horizontal frame axes

% Frame corner naming convention
% A B
% D C


function [outframe,m_per_px,frame_extrema_SN_WE] = rectifier_deluxe(inframe,aov_h,freeboard,pitch,roll,heading,order)
%%
rot_x = roll;
rot_y = pitch;
rot_z = heading;

[n_rows,n_cols] = size(inframe);

aov_v = aov_h*n_rows/n_cols;

% pre-rotation frame corner positions
h1 = -1*freeboard*tand(aov_h/2);
h2 = -h1;
v1 = freeboard*tand(aov_v/2);
v2 = -v1;
scale_x = abs(2*v1/n_rows);
scale_y = abs(2*h1/n_cols);
m_per_px_old = mean([scale_x scale_y],2);

% corner rotation preparation
% A B
% D C
corner_A = [freeboard h1 -v1];
corner_B = [freeboard h2 -v1];
corner_C = [freeboard h2 -v2];
corner_D = [freeboard h1 -v2];
% figure(5); clf; 
% plot([corner_A(1,3),corner_B(1,3),corner_C(1,3),corner_D(1,3)], [corner_A(1,2),corner_B(1,2), corner_C(1,2), corner_D(1,2)],'-*')
% rotate each frame corner in three dimensions
switch nargin
    case 7
        [A_x,A_y,A_z] = myrot3D(corner_A(:,1),corner_A(:,2),corner_A(:,3),rot_x,rot_y,rot_z,order);
        [B_x,B_y,B_z] = myrot3D(corner_B(:,1),corner_B(:,2),corner_B(:,3),rot_x,rot_y,rot_z,order);
        [C_x,C_y,C_z] = myrot3D(corner_C(:,1),corner_C(:,2),corner_C(:,3),rot_x,rot_y,rot_z,order);
        [D_x,D_y,D_z] = myrot3D(corner_D(:,1),corner_D(:,2),corner_D(:,3),rot_x,rot_y,rot_z,order);
    case 6
       
        [A_x,A_y,A_z] = myrot3D(corner_A(:,1),corner_A(:,2),corner_A(:,3),rot_x,rot_y,rot_z,'Rzyx');
        [B_x,B_y,B_z] = myrot3D(corner_B(:,1),corner_B(:,2),corner_B(:,3),rot_x,rot_y,rot_z,'Rzyx');
        [C_x,C_y,C_z] = myrot3D(corner_C(:,1),corner_C(:,2),corner_C(:,3),rot_x,rot_y,rot_z,'Rzyx');
        [D_x,D_y,D_z] = myrot3D(corner_D(:,1),corner_D(:,2),corner_D(:,3),rot_x,rot_y,rot_z,'Rzyx');
end
% hold on 
% plot([A_x,B_x,C_x,D_x], [A_y,B_y, C_y, D_y],'--')
%%
% find intersection of rotated vectors with (assumed flat) water surface
A_x_new = (freeboard./A_z).*A_x;
A_y_new = (freeboard./A_z).*A_y;

B_x_new = (freeboard./B_z).*B_x;
B_y_new = (freeboard./B_z).*B_y;

C_x_new = (freeboard./C_z).*C_x;
C_y_new = (freeboard./C_z).*C_y;

D_x_new = (freeboard./D_z).*D_x;
D_y_new = (freeboard./D_z).*D_y;

% find distance between top two corners
R_top = sqrt((A_x_new-B_x_new)^2+(A_y_new-B_y_new)^2);

% compute scaling factor for m_per_px
scaling_factor = R_top/(h2-h1);
m_per_px_new = m_per_px_old*scaling_factor;

% compute the extent of the rectified region
ymin = min([A_y_new B_y_new C_y_new D_y_new]);
xmin = min([A_x_new B_x_new C_x_new D_x_new]);
ymax = max([A_y_new B_y_new C_y_new D_y_new]);
xmax = max([A_x_new B_x_new C_x_new D_x_new]);

frame_width_m = xmax - xmin;
frame_height_m = ymax - ymin;

frame_extrema_SN_WE = [xmin xmax; ymin ymax]; %first row is vertical frame limits, second raw is horizontal frame limits

% prepare projective transform
iptsetpref('ImshowAxesVisible','on');
base_points = floor(([A_y_new, A_x_new; B_y_new, B_x_new; D_y_new, D_x_new; C_y_new, C_x_new]/m_per_px_new));
input_points = floor([h1, v1; h2, v1; h1, v2; h2, v2]/m_per_px_old);
base_points = base_points - [1;1;1;1]*[min(base_points(:,1)) min(base_points(:,2))] + ones(4,2);
input_points = input_points - [1;1;1;1]*[min(input_points(:,1)) min(input_points(:,2))] + ones(4,2);
tform = fitgeotrans(input_points,base_points,'projective');

% execute transformation
outframe = imwarp(flipud((inframe)),tform);
%outframe = imwarp(inframe,tform);
% outframe = imwarp((flipud((inframe))),tform);

[s1,s2] = size(outframe);
m_per_px = mean([frame_width_m/s1 frame_height_m/s2]);
