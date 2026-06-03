% Feed in one raw Pyxis frame; obtain Stokes parameters
%
% Implementation of
% Ratliff et al. (2009)
% using 12 kernel approximation
%
function [S0,S1,S2] = Compute_StokesVecs_by_KernelAveraging(im,kernel_type)

[rows,cols] = size(im);

switch kernel_type
       
    case '4x4'

        num_blocks_rows = floor(rows / 4);
        num_blocks_cols = floor(cols / 4);
        I0_stack = im(1:4*num_blocks_rows,1:4*num_blocks_cols);
        I45_stack = im(1:4*num_blocks_rows,1:4*num_blocks_cols);
        I90_stack = im(1:4*num_blocks_rows,1:4*num_blocks_cols);
        I135_stack = im(1:4*num_blocks_rows,1:4*num_blocks_cols);

        % KERNELS
        A = 0.4086;
        B = 0.2957;
        H0 = [0 0 0 0; 0 A 0 B; 0 0 0 0; 0 B 0 0]; %inds0
        H45 = [0 B 0 0; 0 0 0 0; 0 A 0 B; 0 0 0 0]; %inds45
        H90 = [0 0 B 0; 0 0 0 0; B 0 A 0; 0 0 0 0]; %inds90
        H135 = [0 0 0 0; B 0 A 0; 0 0 0 0; 0 0 B 0]; %inds135

        H0 = repmat(H0,[1 1 num_blocks_rows num_blocks_cols]);
        H45 = repmat(H45,[1 1 num_blocks_rows num_blocks_cols]);
        H90 = repmat(H90,[1 1 num_blocks_rows num_blocks_cols]);
        H135 = repmat(H135,[1 1 num_blocks_rows num_blocks_cols]);

        im0 = reshape(I0_stack,[4 num_blocks_rows 4 num_blocks_cols]);
        im0 = permute(im0,[1 3 2 4]);
        I0_frame = squeeze(sum(double(im0).*H0,[1 2]));

        im45 = reshape(I45_stack,[4 num_blocks_rows 4 num_blocks_cols]);
        im45 = permute(im45,[1 3 2 4]);
        I45_frame = squeeze(sum(double(im45).*H45,[1 2]));

        im90 = reshape(I90_stack,[4 num_blocks_rows 4 num_blocks_cols]);
        im90 = permute(im90,[1 3 2 4]);
        I90_frame = squeeze(sum(double(im90).*H90,[1 2]));

        im135 = reshape(I135_stack,[4 num_blocks_rows 4 num_blocks_cols]);
        im135 = permute(im135,[1 3 2 4]);
        I135_frame = squeeze(sum(double(im135).*H135,[1 2]));

end


S0 = (I0_frame + I90_frame + I45_frame + I135_frame)/2;
S1 = (I0_frame - I90_frame)./(I0_frame + I90_frame);
S2 = (I45_frame - I135_frame)./(I45_frame + I135_frame);
