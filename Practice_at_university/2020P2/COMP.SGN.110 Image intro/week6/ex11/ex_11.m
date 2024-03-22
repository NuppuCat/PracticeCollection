%%
disp('task1')
load('yuvdata.mat');
disp('a) yy is row*col,uu vv are row*col/4,maybe bcs use 4:1:1')
y = reshape(yy,[640,360]);

u = reshape(uu,[320,180]);
u=imresize(u,2);

v = reshape(vv,[320,180]);
v = imresize(v,2);
figure;
colormap('gray')
subplot(131), imagesc(uint8(y)), title('y');
subplot(132), imagesc(uint8(u)), title('u');
subplot(133), imagesc(uint8(v)), title('v');
u=u-127;
v = v-127;
YUV=cat(2,y(:),u(:),v(:));
YuvToRgb = [1 0 1.402;
    1 -0.34413 -0.71414;
    1 1.772 0];
RGB=YuvToRgb*YUV';
r = reshape(RGB(1,:),[640 360]);
g = reshape(RGB(2,:),[640 360]);
b = reshape(RGB(3,:),[640 360]);

rgb = zeros([360 640 3]);
% I think the rows and cols should transepose
rgb(:,:,1)=uint8(r');
rgb(:,:,2)=uint8(g');
rgb(:,:,3)=uint8(b');

figure;
imshow(uint8(rgb));

%%
disp('task2')
I = imread('lena.tiff');
It = rgb2ycbcr(I);
Y = It(:,:,1);
Cb = It(:,:,2);
Cr = It(:,:,3);
figure;
colormap('gray')
subplot(131), imagesc(uint8(Y)), title('Y');
subplot(132), imagesc(uint8(Cb)), title('Cb');
subplot(133), imagesc(uint8(Cr)), title('Cr');
[m,n]=size((Y));
i1 = (1:2:n);
i2 = (1:4:n);
Cbs1 = Cb(:,i1);
Cbs2 = Cb(:,i2);
Cbs3 = Cb(i1,i1);
Crs1 = Cr(:,i1);
Crs2 = Cr(:,i2);
Crs3 = Cr(i1,i1);
Ys = Y(i1,i1);

% remake
Cbs1 = imresize(Cbs1,[m,n]);
Cbs2 = imresize(Cbs2,[m,n]);
Cbs3 = imresize(Cbs3,[m,n]);
Crs1 = imresize(Crs1,[m,n]);
Crs2 = imresize(Crs2,[m,n]);
Crs3 = imresize(Crs3,[m,n]);
%Ys = imresize(Ys,[m,n]);
Ys = repelem(Ys,2,2);
YCC1 = I;
YCC2 = YCC1;
YCC3 = YCC1;
YCC4 = YCC1;

YCC1(:,:,1)=Y;
YCC1(:,:,2)=Cbs1;
YCC1(:,:,3)=Crs1;
YCC2(:,:,1)=Y;
YCC2(:,:,2)=Cbs2;
YCC2(:,:,3)=Crs2;
YCC3(:,:,1)=Y;
YCC3(:,:,2)=Cbs3;
YCC3(:,:,3)=Crs3;
YCC4(:,:,1)=Ys;
YCC4(:,:,2)=Cb;
YCC4(:,:,3)=Cr;

YCC1 = ycbcr2rgb(YCC1);
YCC2 = ycbcr2rgb(YCC2);
YCC3 = ycbcr2rgb(YCC3);
YCC4 = ycbcr2rgb(YCC4);



figure;

subplot(231), imagesc(I), title('original');
subplot(232), imagesc(YCC1), title('4:2:2');
subplot(233), imagesc(YCC2), title('4:1:1');
subplot(234), imagesc(YCC3), title('4:2:0');
subplot(235), imagesc(YCC4), title('Y 4:2:0');

disp('no perceptible difference')

err1 = immse(I,YCC1)
err2 = immse(I,YCC2)
err3 = immse(I,YCC3)
err4 = immse(I,YCC4)

disp('mse is big, just I cannot differ them :)')






