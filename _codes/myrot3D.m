
function [out_x,out_y,out_z] = myrot3D(in_x,in_y,in_z, roll,pitch,heading, order)

alpha = heading; %rotation about z axis
beta = pitch; %rotation about y axis
gamma = roll; %rotation about x axis

Rz = [cosd(alpha) -sind(alpha) 0; sind(alpha) cosd(alpha) 0; 0 0 1];
Ry = [cosd(beta) 0 sind(beta); 0 1 0; -sind(beta) 0 cosd(beta)];
Rx = [1 0 0; 0 cosd(gamma) -sind(gamma); 0 sind(gamma) cosd(gamma)];

if strcmp(order, 'Rxyz')
    vec_out = Rx*Ry*Rz*[in_x;in_y;in_z];
elseif strcmp(order,'Rxzy')
    vec_out = Rx*Rz*Ry*[in_x;in_y;in_z];
elseif strcmp(order,'Ryxz')
    vec_out = Ry*Rx*Rz*[in_x;in_y;in_z];
elseif strcmp(order,'Ryzx')
    vec_out = Ry*Rz*Rx*[in_x;in_y;in_z];
elseif strcmp(order,'Rzxy')
    vec_out = Rz*Rx*Ry*[in_x;in_y;in_z];
elseif strcmp(order,'Rzyx')
    vec_out = Rz*Ry*Rx*[in_x;in_y;in_z];
end
    
out_x = vec_out(1,1);
out_y = vec_out(2,1);
out_z = vec_out(3,1);

end