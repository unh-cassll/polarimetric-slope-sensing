function b = pad_reflect(a, pad_y, pad_x)
% Reflection padding matching np.pad(mode='reflect'): mirror about the edge
% sample without repeating it. Requires pad_y < size(a,1), pad_x < size(a,2).
b = [a(pad_y+1:-1:2, :); a; a(end-1:-1:end-pad_y, :)];
b = [b(:, pad_x+1:-1:2), b, b(:, end-1:-1:end-pad_x)];
end
