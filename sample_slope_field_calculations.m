addpath _codes/
addpath _data/
CASSLL_figure_defaults

freeboard = 72.0867; %distance between camera and sea surface
%pitch,roll and heading values are based on VectorNav coordinate System
pitch = -66.5100; %degrees
roll = 3.4120; %degrees
heading = 22.5580; %degrees
focal_length = 75; %mm
pixp_microns = 3.48;

s = load('dolp_theta_vecs.mat');
DOLP_vec = s.DOLP_full;
theta_vec = s.theta_full;
ind_max = find(DOLP_vec==max(DOLP_vec),1,'first');
DOLP_full = linspace(0,1,10000)';
theta_full = interp1(DOLP_vec(1:ind_max),theta_vec(1:ind_max),DOLP_full,'pchip');

frame_raw = imread('sample_wave_image.tiff');
[s1,s2] = size(frame_raw);
%% Loop through Methods
mss = zeros(3,1); %preallocate mean square slope

for i = 1:3
    if i==1
        % Calculate Stokes Vectors from Bilinear Interpolation
        [~,S1,S2] = Compute_StokesVecs_by_BilinearInterpolation(frame_raw);
    end
    if i ==2
        % Calculate Stokes Vectors from Kernel Averaging
        [~,S1,S2] = Compute_StokesVecs_by_KernelAveraging(frame_raw,'4x4');
    end
    if i==3
        % Calculate Stokes Vectors from 12 pixel Convolution Demodulation
        [~,S1,S2] = Compute_StokesVecs_by_Conv_Demodul(double(frame_raw),'4x4');
    end

    S1 = S1*1.2185; %11/21/2023 Update: MULTIPLY BY GAIN obtained from polarimeter_cal_script.m
    S2 = S2*1.2197;
    
    figure(i);clf;
    set(gcf,'Position',[120,70,1050,850])
    tlayout = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile
    imagesc(S1);shading('flat');colormap('gray')
    pbaspect([1 s1/s2 1])
    title('S1')
    colorbar;

    nexttile
    imagesc(S2);shading('flat');colormap('gray')
    pbaspect([1 s1/s2 1])
    title('S2')
    colorbar;

    % Calculate Degree of Linear Polarization and Angle of Incidence
    DOLP = sqrt(S1.^2+S2.^2);
    ORI = 0.5*atan2(S2,S1)*180/pi;
    DOLP_int = floor(DOLP*10000);
    DOLP_int(DOLP_int<1) = 1;
    DOLP_int(DOLP_int>10000) = 10000;
    AOI = theta_full(DOLP_int);


    %Calculate Along-Look and Cross-Look Slopes
    Sx = sind(ORI).*tand(AOI);
    Sy = cosd(ORI).*tand(AOI);

    Sx = Sx - mean(Sx,'all','omitnan');
    Sy = Sy - mean(Sy,'all','omitnan');

    % Along-Look and Cross-Look Angles
    Ax = atand(Sx);
    Ay = atand(Sy);

    % Create Georectified Image
    [n_v,n_h] = size(frame_raw);
    [aov_h,~] = get_aov(n_h,n_v,pixp_microns,focal_length);

    Ax = Ax + 1000;
    Ay = Ay + 1000;

    [Ax_out,m_per_px,frame_extrema_SN_WE] = rectifier_deluxe(Ax,aov_h,freeboard,pitch,roll,heading,'Rzyx');
    [Ay_out,~,~] = rectifier_deluxe(Ay,aov_h,freeboard,pitch,roll,heading,'Rzyx');

    Ax_out(Ax_out < 900) = NaN;
    Ay_out(Ay_out < 900) = NaN;

    Ax_out = Ax_out - 1000;
    Ay_out = Ay_out - 1000;

    [s1,s2] = size(Ax_out);

    % Create spatial matrices
    [xmat,ymat] = meshgrid((0:s2-1)*m_per_px,(0:s1-1)*m_per_px);

    northings_m = ymat + frame_extrema_SN_WE(1,1);
    eastings_m = xmat + frame_extrema_SN_WE(2,1);

    % Plot Subsampled Georectified Image
    subnum = 4; %subsampling scale
    xsub = subsample_array(eastings_m,subnum,subnum);
    ysub = subsample_array(northings_m,subnum,subnum);


    Ax_outsub = subsample_array(Ax_out,subnum,subnum);
    Ay_outsub = subsample_array(Ay_out,subnum,subnum);

    xcenter = median(xsub,'all');
    ycenter = median(ysub,'all');

    nexttile
    imagesc(xsub(1,:)-xcenter, ysub(:,1)-ycenter, Ax_outsub);shading('flat');colormap('gray')
    clim([-15 15])
    xlabel('Eastings [m]')
    ylabel('Northings [m]')
    ax_struc(1).ax = gca;
    ax_struc(1).ax.FontSize = 12;
    ax_struc(1).ax.XTick = -100:2:100;
    ax_struc(1).ax.YTick = -100:2:100;
    pbaspect([1 s1/s2 1])
    set(gcf,'Color','w')
    title('cross look [\circ]')
    cbar = colorbar;

    nexttile
    imagesc(xsub(1,:)-xcenter, ysub(:,1)-ycenter, Ay_outsub);shading('flat');colormap('gray')
    clim([-15 15])
    xlabel('Eastings [m]')
    ylabel('Northings [m]')
    ax_struc(1).ax = gca;
    ax_struc(1).ax.FontSize = 12;
    ax_struc(1).ax.XTick = -100:2:100;
    ax_struc(1).ax.YTick = -100:2:100;
    pbaspect([1 s1/s2 1])
    set(gcf,'Color','w')
    title('along look [\circ]')
    cbar = colorbar;

    % variance
    mss_x = var(atand(Ax_out),[],'all','omitnan');
    mss_y = var(atand(Ay_out),[],'all','omitnan');
    mss(i) = mss_x + mss_y;

end