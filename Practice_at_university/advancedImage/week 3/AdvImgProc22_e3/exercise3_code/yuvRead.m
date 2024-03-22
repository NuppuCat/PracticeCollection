%--------------------------------------------------------------------------
% Read YUV 4:2:0 images
%
% - Chroma subsampling: "J:a:b" notation is used to describe how often U 
%and V are sampled relative to Y. 4:2:0 means 2:1 horizontal downsampling,
%with 2:1 vertical downsampling.
%
% Input: 
%       - Image name file
%       - Image width
%       - Image height
%       - Number of frames
%
% Output:
%       - Y, U, V components
%--------------------------------------------------------------------------
function [Y, U, V] = yuvRead(imgName, width, height, nFrame)

% Open file, read and compute length of a single frame
fileID = fopen(imgName,'r');          
stream = fread(fileID,'*uchar');    
length = 1.5 * width * height;  

Y = uint8(zeros(height,   width,   nFrame));
U = uint8(zeros(height/2, width/2, nFrame));
V = uint8(zeros(height/2, width/2, nFrame));

for iFrame = 1:nFrame
    
    frame = stream((iFrame-1)*length+1:iFrame*length);
    
    % Y component of the frame
    yImage = reshape(frame(1:width*height), width, height)';
    % U component of the frame
    uImage = reshape(frame(width*height+1:1.25*width*height), width/2, height/2)';
    % V component of the frame
    vImage = reshape(frame(1.25*width*height+1:1.5*width*height), width/2, height/2)';
    
    Y(:,:,iFrame) = uint8(yImage);
    U(:,:,iFrame) = uint8(uImage);
    V(:,:,iFrame) = uint8(vImage);

end
