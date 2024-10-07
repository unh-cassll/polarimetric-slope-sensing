% Removes extraneous labels and ticks from tiledlayout
% Nathan Laxague 2021

function tile_cleaner(ax_struc,tlayout)

num_rows = tlayout.GridSize(1);
num_cols = tlayout.GridSize(2);

total_boxes = num_rows*num_cols;

left_boxes = 1:num_cols:total_boxes;
bottom_boxes = left_boxes(end):total_boxes;
keep_xlabel = bottom_boxes;
keep_ylabel = left_boxes;

for i = 1:total_boxes
    
x_cond = ~sum(i==keep_xlabel);
    y_cond = ~sum(i==keep_ylabel);
    if x_cond
        ax_struc(i).ax.XLabel.String = '';    
        ax_struc(i).ax.XTickLabels = '';
    end
    if y_cond
        ax_struc(i).ax.YLabel.String = '';    
        ax_struc(i).ax.YTickLabels = '';
    end
    
end

tlayout.TileSpacing = 'compact';