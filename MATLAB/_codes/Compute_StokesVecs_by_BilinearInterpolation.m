% Feed in one raw Pyxis frame; obtain Stokes parameters
%
% N. Laxague 2022
% accept both image stacks and single images G. Duvarci 2024
% 
function [S0,S1,S2] = Compute_StokesVecs_by_BilinearInterpolation(im_stack)

[rows,cols,nframes] = size(im_stack);

% identify indices for each intensity within a superpixel
inds0 = [0 0; 0 1];
inds45 = [0 1; 0 0];
inds90 = [1 0; 0 0];
inds135 = [0 0; 1 0];
if nframes>1
    % repeat 2x2 matrices to full frame size
    inds0 = logical(repmat(inds0,[[rows cols]/2 nframes]));
    inds45 = logical(repmat(inds45,[[rows cols]/2 nframes]));
    inds90 = logical(repmat(inds90,[[rows cols]/2 nframes]));
    inds135 = logical(repmat(inds135,[[rows cols]/2 nframes]));

    % grab individual polarized light intensities from each frame
    I0_frame = reshape(im_stack(inds0),[[rows cols]/2 nframes]);
    I45_frame = reshape(im_stack(inds45),[[rows cols]/2 nframes]);
    I90_frame = reshape(im_stack(inds90),[[rows cols]/2 nframes]);
    I135_frame = reshape(im_stack(inds135),[[rows cols]/2 nframes]);

    % bilinear interpolation to full frame size
    I0_frame=imresize3(I0_frame,[rows cols nframes],'linear');
    I45_frame=imresize3(I45_frame,[rows cols nframes],'linear');
    I90_frame=imresize3(I90_frame,[rows cols nframes],'linear');
    I135_frame=imresize3(I135_frame,[rows cols nframes],'linear');
else
    % repeat 2x2 matrices to full frame size
    inds0 = logical(repmat(inds0,[rows cols]/2));
    inds45 = logical(repmat(inds45,[rows cols]/2));
    inds90 = logical(repmat(inds90,[rows cols]/2));
    inds135 = logical(repmat(inds135,[rows cols]/2));

    % grab individual polarized light intensities from each frame
    I0_frame = reshape(im_stack(inds0),[rows cols]/2);
    I45_frame = reshape(im_stack(inds45),[rows cols]/2);
    I90_frame = reshape(im_stack(inds90),[rows cols]/2);
    I135_frame = reshape(im_stack(inds135),[rows cols]/2);

    % bilinear interpolation to full frame size
    I0_frame=imresize(I0_frame,[rows cols],'bilinear');
    I45_frame=imresize(I45_frame,[rows cols],'bilinear');
    I90_frame=imresize(I90_frame,[rows cols],'bilinear');
    I135_frame=imresize(I135_frame,[rows cols],'bilinear');
end

% Register - measurements to initial positions
% shift 45 right 1
I45_frame(:,cols,:) = []; % remove last column
I45_frame=horzcat(I45_frame(:,1,:), I45_frame); % duplicate first col
% shift 135 down 1
I135_frame(rows,:,:) = []; % remove last row
I135_frame=vertcat(I135_frame(1,:,:), I135_frame); % dupicate first row
% shift 0 right 1 and down 1
I0_frame(:,cols,:) = []; % remove last col
I0_frame=horzcat(I0_frame(:,1,:),I0_frame); % duplicate first col
I0_frame(rows,:,:) = []; % remove last row
I0_frame=vertcat(I0_frame(1,:,:),I0_frame); % duplicate first row

I0_frame(I0_frame==0) = NaN;
I45_frame(I45_frame==0) = NaN;
I90_frame(I90_frame==0) = NaN;
I135_frame(I135_frame==0) = NaN;

I0 = double(I0_frame);
I45 = double(I45_frame);
I90 = double(I90_frame);
I135 = double(I135_frame);

S0 = (I0 + I90 + I45 + I135)/2;
S1 = (I0 - I90)./(I0 + I90);
S2 = (I45 - I135)./(I45 + I135);

% S0 = inpaint_nans(S0);
% S1 = inpaint_nans(S1);
% S2 = inpaint_nans(S2);

% S0 = S0(1:2:size(S0,1),1:2:size(S0,2));
% S1 = S1(1:2:size(S1,1),1:2:size(S1,2));
% S2 = S2(1:2:size(S2,1),1:2:size(S2,2));
