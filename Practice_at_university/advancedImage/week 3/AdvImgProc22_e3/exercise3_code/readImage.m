%--------------------------------------------------------------------------
% Read YUV 4:2:0 images
%
% Input: 
%       - Image name file
%       - Image information name file
%
% Output:
%       - Y, U, V components
%--------------------------------------------------------------------------
function [Y, U, V] = readImage(imageName, imageInfoName)

% Read image information
fileID = fopen(imageInfoName,'r');
imgInfo = textscan(fileID, '%s %s', 8, 'delimiter','=', 'whitespace', '');
fclose(fileID);

% Read image yuv420 (chroma subsampling)
width = str2double(imgInfo{1,2}{1});
height = str2double(imgInfo{1,2}{2});

[Y, U, V] = yuvRead(imageName, width, height, 1);

end