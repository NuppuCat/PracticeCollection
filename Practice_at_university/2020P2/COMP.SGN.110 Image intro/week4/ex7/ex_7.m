%%
disp('task 1')
I = imread('lena.jpg');
Ia = imnoise(I,'gaussian');
Ib = imnoise(I,'salt & pepper');
[height,width]=size(I);
noise = sqrt(-1500*log((1-rand(height,width))));
Ic = double(I)+noise;
figure;
subplot(221), imshow(uint8(I)), title('I');
subplot(222), imshow(uint8(Ia)), title('Ia');
subplot(223), imshow(uint8(Ib)), title('Ib');
subplot(224), imshow(uint8(Ic)), title('Ic');
%%
disp('task 2')
filter_img(Ia,5,5);
filter_img(Ib,5,5);
filter_img(Ic,5,5);
disp('mean filter is best to solve salt&pepper and gaussian noise,while Harmonic mean filter and Geometric mean filter perform better in rayleigh noise.')
%%
disp('task 3')
m = ones(5);
m(3,3) = 5;
Id = imnoise(I,'salt & pepper',0.02);
[Im1,Iwm1]= WMF(Id, 5,m);
figure;
subplot(221), imshow(uint8(I)), title('original');
subplot(222), imshow(uint8(Id)), title('noise image');
subplot(223), imshow(uint8(Im1)), title('median');
subplot(224), imshow(uint8(Iwm1)), title('WMF');

