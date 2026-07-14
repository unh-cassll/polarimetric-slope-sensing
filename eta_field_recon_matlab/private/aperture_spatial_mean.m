function s = aperture_spatial_mean(stack, mask)
% Spatial mean of a (Ny,Nx,T) stack over a boolean aperture mask, per frame.
% Returns a (T,1) series; equivalent to the full-frame mean when mask is
% all-true.
[Ny, Nx, T] = size(stack);
a2 = reshape(stack, Ny*Nx, T);
if all(mask(:))
    s = mean(a2, 1)';
else
    s = (sum(a2(mask(:), :), 1) / sum(mask(:)))';
end
end
