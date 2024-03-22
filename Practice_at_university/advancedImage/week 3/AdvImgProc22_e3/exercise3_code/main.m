clear; close all; clc;

%% Load YUV (raw) image
addpath('Images/');

imageName = 'pepper.yuv';
imageInfoName = 'pepper.inf';

[Y, U, V] = readImage(imageName, imageInfoName);

%% Encoder

% Double precision plus center values around 0 (shifted block)
YCh = double(Y) - 128;
UCh = double(U) - 128;
VCh = double(V) - 128;

%--------------------------------------------------------------------------
% Forward Discret Cosine Transform: 
%
% - Discrete Cosine Transform (DCT) of NxN image blocks 
%(T = H * F * H_transposed), where H is the matrix containing the DCT 
%coefficients (NxN matrix) and F is an NxN image block.  
%--------------------------------------------------------------------------
%%
% TASK: DCT coefficients (create NxN DCT matrix)
N=[2 4 8 16 32 64];

H = dctmtx(N(5));

% TASK: Compute DCT for each block of the image 
dct = @(block_struct) H * block_struct.data * H';
B = blockproc(YCh,[N(5) N(5)],dct);
%--------------------------------------------------------------------------
% Quantization of the DCT coefficients: 
%
% - Standard JPEG quantization tables that represent a quality of 50%:
%
% For luminance (Y): 
%
 Q_table_Y = [16 11 10 16 24 40 51 61;
              12 12 14 19 26 58 60 55;
              14 13 16 24 40 57 69 56;
              14 17 22 29 51 87 80 62; 
              18 22 37 56 68 109 103 77;
              24 35 55 64 81 104 113 92;
              49 64 78 87 103 121 120 101;
              72 92 95 98 112 100 103 99];

  
%
%
% For chrominance (U and V):
%
% Q_table_UV = [17 18 24 47 99 99 99 99;
%               18 21 26 66 99 99 99 99;
%               24 26 56 99 99 99 99 99;
%               47 66 99 99 99 99 99 99;
%               99 99 99 99 99 99 99 99;
%               99 99 99 99 99 99 99 99;
%               99 99 99 99 99 99 99 99;
%               99 99 99 99 99 99 99 99];
%
%
% - Quantization: 
%               DCT element
% q = round( ----------------- )
%             Q_table element
%
%--------------------------------------------------------------------------

% TASK: Apply quantization
%q = @(block_struct) round((block_struct.data) ./ Q_table_Y);
%Q = blockproc(B,[N(3) N(3)],q);

%% Decoder 
%--------------------------------------------------------------------------
% Inverse quantization of DCT Coefficients
%--------------------------------------------------------------------------

% TASK: Apply inverse quantization

%iQ = blockproc(Q,[N(3) N(3)],@(block_struct) Q_table_Y .* block_struct.data);

%--------------------------------------------------------------------------
% Inverse Discrete Cosine Transform:
%--------------------------------------------------------------------------
%%
% TASK: Apply inverse DCT
idctf = @(block_struct) inv(H)* block_struct.data*inv(H');
iB = blockproc(B,[N(5) N(5)],idctf);

%--------------------------------------------------------------------------
% Inverse Shifted Block
%--------------------------------------------------------------------------

% TASK: Apply inverse Shifted Block operation
iB=uint8(iB+128);

%----------------------------------------------------------
% Display results
%----------------------------------------------------------

% TASK: Compute PSNR
figure(1);
subplot(1,2,1);
imshow(iB);
subplot(1,2,2);
imshow(Y);
psnr(iB,Y)





