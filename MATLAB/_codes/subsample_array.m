% Spatially subsamples array
function out_array = subsample_array(in_array,r1,r2)

[m,n] = size(in_array);

a = 1:m;
b = 1:n;

a_s = subsample(a,r1);
b_s = subsample(b,r2);

%smoothed_array = medfilt2(in_array,[r r]);

out_array = in_array(a_s,b_s);

