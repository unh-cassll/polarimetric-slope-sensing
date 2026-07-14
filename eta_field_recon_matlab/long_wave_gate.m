function [enabled, threshold_s, f_min] = long_wave_gate(record_duration_s, freqs_cwt, min_periods)
% Physics gate for the long-wave inversion: the record must span at least
% min_periods periods of the lowest inversion frequency. Port of
% eta_field_recon.recon.long_wave_gate.
%
%   record_duration_s : record length (s)
%   freqs_cwt         : frequency grid (Hz); [] -> default floor 0.05 Hz
%   min_periods       : minimum periods required (default 0.5)

arguments
    record_duration_s (1, 1) double
    freqs_cwt double = []
    min_periods (1, 1) double = 0.5
end

if isempty(freqs_cwt)
    f_min = 0.05;
else
    f_min = min(freqs_cwt(:));
end
threshold_s = min_periods / f_min;
enabled = record_duration_s >= threshold_s;
end
