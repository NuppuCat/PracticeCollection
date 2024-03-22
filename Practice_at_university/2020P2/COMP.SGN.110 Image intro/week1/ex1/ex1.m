%task1
disp('task 1')
I = imread('peppers.png');
figure(1);
imshow(I)
I1 = rgb2gray(I);
figure(2);
imshow(I1);
R1 = I;
R1 = R1(:,:,1); 
figure(3);
imshow(R1);
I2 = I;
I2(:,:,2) = I2(:,:,2)+50;
%is this correct? I2 = I(:,:,2)+50;
figure(4)
imshow(I2);
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
I3 = cat(3,R,G,B);
figure(5)
cla;
imshow(I3);

figure(6)
subplot(2,2,1);
imshow(I);
subplot(2,2,2);
imshow(I1);
subplot(2,2,3);
imshow(I2);
subplot(2,2,4);
imshow(I3);
%%
%task2
disp('task 2')
clear all;
close all;
Ex1_batch();
%%
%task3
disp('task 3')
m = [0 0 0 0 0 0 0 1 1 0;
     1 0 0 1 0 0 1 0 0 1;
     1 0 0 1 0 1 1 0 0 0;
     0 0 1 1 1 0 0 0 0 0;
     0 0 1 1 1 0 0 1 1 1;];
figure(7)
%SUIT SCREEN
imshow(m,'InitialMagnification','fit');
S1 = m(1:4,2:5);
S2 = m(1:4,6:9);
s1 = getnum(S1);
fprintf('%d\n',s1);

s2 = getnum(S2);
fprintf('%d\n',s2);

load S
s = getnum(S);
fprintf('%d\n',s);