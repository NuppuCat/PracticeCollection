%%
clear all;
close all;
disp('task1')
I = imread('cheetah.jpg');
I2 = imread('chameleon.jpg');
It = sliceCube(I,50);
It2 = sliceSphere(I,50);
I2t = sliceCube(I2,50);
I2t2 = sliceSphere(I2,50);
figure;
subplot(231), imagesc(I), title('I');
subplot(232), imagesc(It), title('sliceCube');
subplot(233), imagesc(It2), title('sliceSphere');
subplot(234), imagesc(I2), title('I2');
subplot(235), imagesc(I2t), title('sliceCube');
subplot(236), imagesc(I2t2), title('sliceSphere');
%%
disp('task2')
clear all;
close all;
%a)
I = imread('lena.tiff');
hI= rgb2hsi(I);
 r = I(:,:,1);
 g = I(:,:,2);
 b = I(:,:,3);
 h = hI(:,:,1);
 s = hI(:,:,2);
 i = hI(:,:,3);
figure;
colormap('gray')
subplot(231), imagesc(r), title('r');
subplot(232), imagesc(g), title('g');
subplot(233), imagesc(b), title('b');
subplot(234), imagesc(h), title('h');
subplot(235), imagesc(s), title('s');
subplot(236), imagesc(i), title('i');
%b)
gn = imnoise(g,'gaussian');
Ic = I;
Ic(:,:,2)=gn;
figure;
imagesc(Ic), title('noised g');

hIc= rgb2hsi(Ic);
 hc = hIc(:,:,1);
 sc = hIc(:,:,2);
 ic = hIc(:,:,3);
 figure;
 colormap('gray')
subplot(131), imagesc(hc), title('h');
subplot(132), imagesc(sc), title('s');
subplot(133), imagesc(ic), title('i');
disp('all are affected, especially h and i,because green noise will show in image,change the h and s a lot,i is gray level, just some noise.')
%c)
rn = imnoise(r,'gaussian');
bn = imnoise(b,'gaussian');
Ic2 = I;
Ic2(:,:,1)=rn;
Ic2(:,:,2)=gn;
Ic2(:,:,3)=bn;
figure;
imagesc(Ic2), title('noised all');

hIc2= rgb2hsi(Ic2);
 hc2 = hIc2(:,:,1);
 sc2 = hIc2(:,:,2);
 ic2 = hIc2(:,:,3);
 
figure;
 colormap('gray')
 
subplot(131), imagesc(hc2), title('h');
subplot(132), imagesc(sc2), title('s');
subplot(133), imagesc(ic2), title('i');
disp('too much noise in h and s, while i still looks good')
%d)
h = fspecial('average',3);

rf = imfilter(rn,h,'replicate');
gf = imfilter(gn,h,'replicate');
bf = imfilter(bn,h,'replicate');
hf = imfilter(hc2,h,'replicate');
sf = imfilter(sc2,h,'replicate');
iff = imfilter(ic2,h,'replicate');
Ifr = Ic2;
Ifr(:,:,1)=rf;
Ifg = Ic2;
Ifg(:,:,2)=gf;
Ifb = Ic2;
Ifb(:,:,3)=bf;
Ifh = hIc2;
Ifh(:,:,1)=hf;
Ifs = hIc2;
Ifs(:,:,2)=sf;
Ifi = hIc2;
Ifi(:,:,3)=iff;

 figure;
colormap('gray')
subplot(231), imagesc(Ifr), title('filtered r');
subplot(232), imagesc(Ifg), title('filtered g');
subplot(233), imagesc(Ifb), title('filtered b');
subplot(234), imagesc(his2rgb(Ifh)), title('filtered h');
subplot(235), imagesc(his2rgb(Ifs)), title('filtered s');
subplot(236), imagesc(his2rgb(Ifi)), title('filtered i');
disp('only the hue component cannot be filtered directly,it will change the color of image.')