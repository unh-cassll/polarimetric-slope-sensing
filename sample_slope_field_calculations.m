addpath _codes/
addpath _data/
CASSLL_figure_defaults

freeboard = 72.0867;
pitch = -66.5100;
roll = 3.4120;
heading = 22.5580;
focal_length = 75;
pixp_microns = 3.48;
subnum = 4;
ang_lims = [-1 1]*15;
s = load('dolp_theta_vecs.mat');
DOLP_vec = s.DOLP_full;
theta_vec = s.theta_full;
ind_max = find(DOLP_vec==max(DOLP_vec),1,'first');
DOLP_full = linspace(0,1,10000)';
theta_full = interp1(DOLP_vec(1:ind_max),theta_vec(1:ind_max),DOLP_full,'pchip');

frame_raw = imread('sample_wave_image.tiff');


    [n_v,n_h] = size(frame_raw);

    [aov_h,~] = get_aov(n_h,n_v,pixp_microns,focal_length);
[~,S1,S2] = Compute_StokesVecs_by_BilinearInterpolation(frame_raw);

figure(10);clf;
tiledlayout(1,3,"TileSpacing","tight")
nexttile
imagesc(S1);shading('flat');colormap('gray')

[~,S1,S2] = Compute_StokesVecs_by_KernelAveraging(frame_raw,'4x4');
nexttile
imagesc(S1);shading('flat');colormap('gray')

% [~,S1,S2] = Compute_StokesVecs_by_Conv_Demodul(frame_raw, '4x4');
% nexttile
% imagesc(S1);shading('flat');colormap('gray')

S1 = S1*1.2185; %11/21/2023 Update: MULTIPLY BY GAIN obtained from polarimeter_cal_script.m
S2 = S2*1.2197;
DOLP = sqrt(S1.^2+S2.^2);
ORI = 0.5*atan2(S2,S1)*180/pi;
DOLP_int = floor(DOLP*10000);
DOLP_int(DOLP_int<1) = 1;
DOLP_int(DOLP_int>10000) = 10000;
AOI = theta_full(DOLP_int);



Sx = sind(ORI).*tand(AOI);
Sy = cosd(ORI).*tand(AOI);

Sx = Sx - mean(Sx,'all','omitnan');
Sy = Sy - mean(Sy,'all','omitnan');

Sx_vec = sort(reshape(Sx,[],1));

Sx_99 = Sx_vec(floor(0.99*length(Sx_vec)));
Sx(Sx>Sx_99) = NaN;

Sx = inpaint_nans(Sx);

Ax = atand(Sx);
Ay = atand(Sy);



[n_v,n_h] = size(frame_raw);

[aov_h,~] = get_aov(n_h,n_v,pixp_microns,focal_length);



Ax = Ax + 1000;
Ay = Ay + 1000;


% Create Georectified Image
[Ax_out,m_per_px,frame_extrema_SN_WE] = rectifier_deluxe(Ax,aov_h,freeboard,pitch,roll,heading,'Rzyx');

%Ax_out(Ax_out==0) = NaN;

[s1,s2] = size(Ax_out);

[Ay_out,~,~] = rectifier_deluxe(Ay,aov_h,freeboard,pitch,roll,heading,'Rzyx');


Ax_out(Ax_out < 900) = NaN;
Ay_out(Ay_out < 900) = NaN;

Ax_out = Ax_out - 1000;
Ay_out = Ay_out - 1000;

Ay_out = -1*Ay_out;
camera_heading_deg = heading;
A_N = Ay_out*cosd(camera_heading_deg) - Ax_out*sind(camera_heading_deg);
A_E = Ay_out*sind(camera_heading_deg) + Ax_out*cosd(camera_heading_deg);

[s1,s2] = size(Ax_out);

%  Create spatial matrices


[xmat,ymat] = meshgrid((0:s2-1)*m_per_px,(0:s1-1)*m_per_px);
%ymat = flipud(ymat);

northings_m = ymat + frame_extrema_SN_WE(1,1);% - SN_mean;
eastings_m = xmat + frame_extrema_SN_WE(2,1);% - WE_mean;

% Plot Subsampled Georectified Image

xsub = subsample_array(eastings_m,subnum,subnum);
ysub = subsample_array(northings_m,subnum,subnum);


%   Ax_outsub = subsample_array(Ax_out,subnum,subnum);
A_N_outsub = subsample_array(A_N,subnum,subnum);

% Ay_out(Ay_out==0) = NaN;

%    Ay_outsub = subsample_array(Ay_out,subnum,subnum);
A_E_outsub = subsample_array(A_E,subnum,subnum);

    xcenter = median(xsub,'all');
    ycenter = median(ysub,'all');
%%
figure(105);
tiledlayout(1,2)
nexttile(1)
imagesc(xsub(1,:)-xcenter, ysub(:,1)-ycenter, A_E_outsub);shading('flat');colormap('gray')
% hold on
% plot(0,0,'r.','markersize',10)

clim(ang_lims)
xlabel('Eastings [m]')
ylabel('Northings [m]')
ax_struc(1).ax = gca;
ax_struc(1).ax.FontSize = 12;
ax_struc(1).ax.XTick = -100:2:100;
ax_struc(1).ax.YTick = -100:2:100;
pbaspect([1 20/15 1])
set(gcf,'Color','w')
title('cross look [\circ]')

 nexttile(2)
        imagesc(xsub(1,:)-xcenter, ysub(:,1)-ycenter, A_N_outsub);shading('flat');colormap('gray')
        % hold on
        % plot(0,0,'r.','markersize',10)
        % hold off
     
        clim(ang_lims)
        xlabel('Eastings [m]')
        ylabel('Northings [m]')
        ax_struc(2).ax = gca;
        ax_struc(2).ax.FontSize = 12;
        ax_struc(2).ax.XTick = -100:2:100;
        ax_struc(2).ax.YTick = -100:2:100;
        pbaspect([1 20/15 1])
        set(gcf,'Color','w')
        title('along look [\circ]')
        cbar = colorbar;



