%% SGN-31007 Exercise 2
% Image Interpolation

%% Parameters to play with

d = 2; % downsampling rate (integer for simplicity)
n =1; % Order, try with 0: nearest, 1: linear, 3: cubic


%% Load high-resolution (ground truth) image
I_gt = im2double(imread('motorcycle.png'));

%% Down-sample
av = 1/d^2*ones(d); % average the pixels
I_gt = conv2(I_gt,av,'same'); 
I_low = I_gt(1:d:end,1:d:end); % Low resolution image


%% Up-sample with zeros
I_zeros = zeros(size(I_gt));
I_zeros(1:d:end,1:d:end) = I_low;

%% Spline Kernel
xk = -3:1/d:3; % Kernel grid

%%%% Write your own function for spline kernel
b_1d = bspline(xk,n);

% 2D kernel
b = b_1d'*b_1d;

%% Final reconstructed image
Iup = conv2(I_zeros,b,'same');

%% Reconstruction result
figure(1);
imshow(Iup);
PSNR = psnr(I_gt,Iup);
title(sprintf('Spline Order = %i, PSNR = %.2f', n, PSNR));

%% Frequency response of the spline kernels

figure(2);
plot(freq_response(n));
ylabel('magnitude')
xlabel('frequency')
title(sprintf('Frequency Response,Spline Order = %i', n));

