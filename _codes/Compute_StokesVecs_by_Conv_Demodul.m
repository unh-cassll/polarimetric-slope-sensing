% Feed in one raw Pyxis frame; obtain Stokes parameters
% Convolution - Demodulation Schemes Applied

function [S0,S1,S2] = Compute_StokesVecs_by_Conv_Demodul(im,kernel_type)
%%

[rows,cols] = size(im);


switch kernel_type

    case '4x4'

        % KERNELS
        A = 0.4086;
        B = 0.2957;
        inds0 =  [0 0 0 0; 0 A 0 B; 0 0 0 0; 0 B 0 0]; %inds0
        inds45 = [0 B 0 0; 0 0 0 0; 0 A 0 B; 0 0 0 0]; %inds45
        inds90 = [0 0 B 0; 0 0 0 0; B 0 A 0; 0 0 0 0]; %inds90
        inds135 = [0 0 0 0; B 0 A 0; 0 0 0 0; 0 0 B 0]; %inds135

        %
        I0_v2 = conv2(im, inds0, 'same');
        I45_v2 = conv2(im, inds45, 'same');
        I90_v2 = conv2(im, inds90, 'same');
        I135_v2 = conv2(im, inds135, 'same');


        % Demodulation
        I = NaN*zeros(rows,cols,4);
        m = repmat((1:rows)',1,cols);
        n = repmat((1:cols),rows,1);
        for j = 1:4
            l =  0.5*(-1).^(m + ceil(0.5 * (-1)^j)) + (-1).^(n + floor(0.5*(j-1))) + 2.5;
            I(:,:,j) = I0_v2.*~logical(l-1) + I45_v2.*~logical(l-2) + I90_v2.*~logical(l-3) + I135_v2.*~logical(l-4);
        end

        I0_frame = I(:,:,1);
        I45_frame = I(:,:,2);
        I90_frame = I(:,:,3);
        I135_frame = I(:,:,4);

        I0_frame(I0_frame==0) = NaN;
        I45_frame(I45_frame==0) = NaN;
        I90_frame(I90_frame==0) = NaN;
        I135_frame(I135_frame==0) = NaN;

        I0 = double(I0_frame);
        I45 = double(I45_frame);
        I90 = double(I90_frame);
        I135 = double(I135_frame);

end
S0 = (I0 + I90 + I45 + I135)/2;
S1 = (I0 - I90)./(I0 + I90);
S2 = (I45 - I135)./(I45 + I135);
S2 = inpaint_nans(S2);

