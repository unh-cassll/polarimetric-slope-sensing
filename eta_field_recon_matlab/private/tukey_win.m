function w = tukey_win(M, alpha)
% Symmetric Tukey window, replicating scipy.signal.windows.tukey(M, alpha).
% alpha<=0 -> rectangular; alpha>=1 -> Hann. Returns (M,1) column.
if M == 1
    w = 1;
    return
end
n = (0:M-1)';
if alpha <= 0
    w = ones(M, 1);
    return
end
if alpha >= 1
    w = 0.5 - 0.5*cos(2*pi*n/(M - 1));
    return
end
width = floor(alpha*(M - 1)/2);
w = ones(M, 1);
n1 = n(1:width+1);
w(1:width+1) = 0.5*(1 + cos(pi*(-1 + 2*n1/alpha/(M - 1))));
n3 = n(M-width:M);
w(M-width:M) = 0.5*(1 + cos(pi*(-2/alpha + 1 + 2*n3/alpha/(M - 1))));
end
