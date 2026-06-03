addpath _codes/
addpath _data/
CASSLL_figure_defaults

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

    nexttile
    imagesc(Ax);shading('flat');colormap('gray')
    pbaspect([1 s1/s2 1])
    title('cross look [\circ]')
    colorbar;

    nexttile
    imagesc(Ay);shading('flat');colormap('gray')
    pbaspect([1 s1/s2 1])
    title('along look [\circ]')
    colorbar;

    xlabel('# of pixels','Parent',tlayout,'FontSize',16)
    ylabel('# of pixels','Parent',tlayout,'FontSize',16)
    

    % variance
    mss_x = var(atand(Ax),[],'all','omitnan');
    mss_y = var(atand(Ay),[],'all','omitnan');
    mss(i) = mss_x + mss_y;

end

