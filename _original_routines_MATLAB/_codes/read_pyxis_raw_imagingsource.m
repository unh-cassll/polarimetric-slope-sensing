% N. Laxague, 10/2022
% Following Pyxis Operator's Manual, Polaris Sensor Technologies
% For use with ImagingSource camera on benchtop (no UAV acquisition)
%
function frame_raw = read_pyxis_raw_imagingsource(file_folder,file_prefix,index)

% Frame dimensions and precision
cols = 2448;
rows = 2048;
prec = 'uint16';

% Header sizes
FILE_DATA_BEGIN = 2048;
FRAME_HEADER_SIZE = 26;
%META_HEADER_SIZE = 40;
META_HEADER_SIZE = 28;

if length(index)>1
    frame_raw = zeros(rows,cols,length(index));
    counter = 0;
    for frame_num=index(1):index(end)
        % Compute byte offset
        Offset_to_index = (FILE_DATA_BEGIN + FRAME_HEADER_SIZE + META_HEADER_SIZE + (frame_num -1)*(cols*rows*2 + FRAME_HEADER_SIZE + META_HEADER_SIZE));
        fname = [file_folder file_prefix '.raw'];
        fid = fopen(fname);
        fseek(fid,Offset_to_index,'bof');
        counter = counter + 1; 
        frame_raw(:,:,counter) = uint16(fread(fid,[cols rows],prec)');
        fclose(fid);
    end
else
    Offset_to_index = (FILE_DATA_BEGIN + FRAME_HEADER_SIZE + META_HEADER_SIZE + (index -1)*(cols*rows*2 + FRAME_HEADER_SIZE + META_HEADER_SIZE));
    fname = [file_folder file_prefix '.raw'];
    fid = fopen(fname);
    fseek(fid,Offset_to_index,'bof');
    frame_raw = uint16(fread(fid,[cols rows],prec)');
    fclose(fid);
end
