%%
%task 1
close all;
clear all;
disp('task1')
z = zeros(128);
Ia = ones(128)*0.5*255;
Ib = z;
Ib(54:74,54:74) =255;
Ic = repmat(linspace(0,1,128)'*255,1,128);
Id = z;
Id(64,64) = 255;
x = cos(2*pi*4*linspace(0,1,128))*255;
y = cos(2*pi*4*linspace(0,1,128))*255;
[X,Y] = meshgrid(x,y);
Ie=X+Y;
figure;
subplot(321), imshow(uint8(Ia)), title('Ia');
subplot(322), imshow(uint8(Ib)), title('Ib');
subplot(323), imshow(uint8(Ic)), title('Ic');
subplot(324), imshow(uint8(Id)), title('Id');
subplot(325), imshow(uint8(Ie)), title('Ie');
%%
%task 2

disp('task2')
Fa = fft2(Ia);
Fb = fft2(Ib);
Fc = fft2(Ic);
Fd = fft2(Id);
Fe = fft2(Ie);
Fa = fftshift(Fa);
Fb = fftshift(Fb);
Fc = fftshift(Fc);
Fd = fftshift(Fd);
Fe = fftshift(Fe);

Fa_spec = log(abs(Fa)+0.0001);
Fb_spec = log(abs(Fb)+0.0001);
Fc_spec = log(abs(Fc)+0.0001);
Fd_spec = log(abs(Fd)+0.0001);
Fe_spec = log(abs(Fe)+0.0001);

figure;
subplot(321), imshow(Fa_spec), title('Fa');
subplot(322), imshow(Fb_spec), title('Fb');
subplot(323), imshow(Fc_spec), title('Fc');
subplot(324), imshow(Fd_spec), title('Fd');
subplot(325), imshow(Fe_spec), title('Fe');
disp('frequency distribution and directions')

%%
%task 3
disp('task3')
I = imread('cameraman.tif');
f = BWLPfilter(I, 20, 2);
f1 = 1-f;
F = fft2(I);
Fc = fftshift(F);
G1 = f.*Fc;

G2 = f1.*Fc;
gi = ifft2(ifftshift(G1));
gi2 = ifft2(ifftshift(G2));
g = real(gi);
g2 = real(gi2);
figure;
colormap('gray')
imagesc(g)
figure;
colormap('gray')
imagesc(g2)




