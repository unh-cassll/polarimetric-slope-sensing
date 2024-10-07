% Subsamples input vector
function out_vec = subsample(in_vec,r)

in_vec = in_vec(:);

in_l = length(in_vec);

N = floor(in_l/r);

in_vec = in_vec(1:N*r);

reshaped = reshape(in_vec,[r N]);

out_vec = reshaped(1,:)';