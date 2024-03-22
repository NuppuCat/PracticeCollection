%%
%task 1
disp('task1')
A = {8, 9, 9.7};

a = ones([3,3])* (-1);
a1 = a;
a2 = a;
a3 = a;
a1(2,2) = 8;
a2(2,2) = 9;
a3(2,2) = 9.7;

I = imread('cameraman.tif');
I1 = imfilter(I,a1);
I2 = imfilter(I,a2);
I3 = imfilter(I,a3);
figure;
subplot(221), imshow(uint8(I)), title('Original Image');
subplot(222), imshow(uint8(I+I1)), title('8');
subplot(223), imshow(uint8(I+I2)), title('9');
subplot(224), imshow(uint8(I+I3)), title('9.7');
%%
%task 2
disp('task2')
clear all;
close all;
I = imread('cameraman.tif');
I1 = I + uint8(10*randn(size(I)));
figure;
subplot(121), imshow(uint8(I)), title('Original Image');
subplot(122), imshow(uint8(I1)), title('noisy');
directional_filtering(I1, 3);
directional_filtering(I1, 5);
directional_filtering(I1, 7);
disp('Maybe I+I1+I2+I3+I4?')
%%
%task 3
disp('task3')
clear all;
close all;
I = imread('miranda1.tif');
%[m,n]=size(I);
I(150:250,150:250) = I(150:250,150:250)+uint8(10*randn(size(I(150:250,150:250))));
figure;
imshow(I)
size = 16;
med_filter(I, size)
med_filter_plus(I, size,4)
disp('if we still want some details we should use threshhod')








