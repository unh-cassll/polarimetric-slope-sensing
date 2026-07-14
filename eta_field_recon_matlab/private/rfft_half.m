function X = rfft_half(x)
% One-sided FFT of a real series, matching np.fft.rfft: (floor(T/2)+1,1).
x = x(:);
T = numel(x);
Xf = fft(x);
X = Xf(1:floor(T/2)+1);
end
