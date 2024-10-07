
wiper

set(0,'DefaultAxesFontName','Arial')
set(0,'DefaultAxesFontSize',24)
set(0,'DefaultFigureColormap',viridis)
set(0,'DefaultAxesLineWidth',2)
set(0,'DefaultLineMarkerSize',20)
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on')
set(groot, 'defaultAxesTickDir', 'out');
set(groot,  'defaultAxesTickDirMode', 'manual');
set(groot,'defaultFigureRenderer','painters')
v_order = [7 6 3 1 5 2];
cmap = spectral(7);
set(0,'DefaultAxesColorOrder',cmap(v_order,:));
clear
