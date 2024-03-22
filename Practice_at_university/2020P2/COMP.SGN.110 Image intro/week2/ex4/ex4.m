%%
%Task 1
disp('task1')
I = imread('university.png');

c = 0.21
S = c * log(1+double(I));
c2 = 0.17
r = 0.45
S2 = c2*(double(I).^r);
figure(1)
subplot(3,1,1);
imshow(I);
subplot(3,1,2);
imshow(S);
subplot(3,1,3);
imshow(S2);
disp('because their input/output grey level curves are different  ')

%%
%Task 2
disp('task2')
I1 = imread('moon.png');
I2 = imread('house.png');
I3 = imread('spine.jpg');
I4 = imread('church.png');
histequal(I1);
ContrastStretch(I1, 0, 255)
histequal(I2);
ContrastStretch(I2, 0, 255)
histequal(I3);
ContrastStretch(I3, 0, 255)
histequal(I4);
ContrastStretch(I4, 0, 255)
%%
%Task 3
clear all;
close all;
disp('task3')
I1 = imread('corel.png');
I2 = imread('church.png');
HistogramMatching(I1,I2);
disp('the hist of church distribut as equalized corel')
%%
%Task 4
clear all;
close all;
disp('task4')
disp('I think it is because after equalization, the cdf will as a straight line, it is balanced, so it will not change')
disp('I think it is because their hist have high level in narrow band.I found adapthisteq can solve it.')
disp('it uses histmatching to process different parts of the picture')
disp('Localized Histogram Equalization can also solve the problem,but I do not know how to code it.')
I1 = imread('moon.png');
I2 = imread('spine.jpg');
g1 = adapthisteq(I1);
g2 = adapthisteq(I2);
figure;
subplot(221), imshow(uint8(I1)), title('Original Image');
subplot(222), imhist(uint8(I1)), title('Histogram');
subplot(223), imshow(uint8(g1)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(g1)), title('Histogram');
figure;
subplot(221), imshow(uint8(I2)), title('Original Image');
subplot(222), imhist(uint8(I2)), title('Histogram');
subplot(223), imshow(uint8(g2)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(g2)), title('Histogram');




