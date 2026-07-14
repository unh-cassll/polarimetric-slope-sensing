function y = irfft_half(S, T)
% Inverse of the one-sided FFT, matching np.fft.irfft(S, n=T): rebuild the
% conjugate-symmetric full spectrum and inverse-transform to a real (T,1)
% series. Imaginary parts of the DC and Nyquist bins are ignored.
S = S(:);
nf = floor(T/2) + 1;
if numel(S) ~= nf
    error('etaFieldRecon:badSpectrum', ...
        'spectrum length %d does not match floor(T/2)+1 = %d', numel(S), nf);
end
if mod(T, 2) == 0
    full_spec = [S; conj(S(end-1:-1:2))];
else
    full_spec = [S; conj(S(end:-1:2))];
end
y = ifft(full_spec, 'symmetric');
end
