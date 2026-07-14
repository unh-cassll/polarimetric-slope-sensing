function w = temporal_window_vec(T, kind, alpha)
% 1-D temporal window; kind is 'tukey', 'hann', or 'rect'. Returns (T,1).
if isempty(kind)
    w = ones(T, 1);
    return
end
switch lower(char(kind))
    case 'hann'
        w = tukey_win(T, 1.0);
    case 'tukey'
        w = tukey_win(T, alpha);
    case 'rect'
        w = ones(T, 1);
    otherwise
        error('etaFieldRecon:badWindow', ...
            'unknown temporal window kind ''%s''', char(kind));
end
end
