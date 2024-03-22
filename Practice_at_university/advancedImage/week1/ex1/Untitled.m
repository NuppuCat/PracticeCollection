clear all;
img = imread('apple.png');
img = rgb2gray(img);

[h,w]=size(img);
Alv2=img;
Alv1= [mean(mean(img(1:h/4,1:w/4))) mean(mean(img(1:h/4,w/4+1:w/2))) mean(mean(img(1:h/4,w/2+1:3*w/4))) mean(mean(img(1:h/4,3*w/4+1:w)))
    mean(mean(img(h/4+1:h/2,1:w/4))) mean(mean(img(h/4+1:h/2,w/4+1:w/2))) mean(mean(img(h/4+1:h/2,w/2+1:3*w/4))) mean(mean(img(h/4+1:h/2,3*w/4+1:w)))
    mean(mean(img(h/2+1:3*h/4,1:w/4))) mean(mean(img(h/2+1:3*h/4,w/4+1:w/2))) mean(mean(img(h/2+1:3*h/4,w/2+1:3*w/4))) mean(mean(img(h/2+1:3*h/4,3*w/4+1:w)))
    mean(mean(img(3*h/4+1:h,1:w/4))) mean(mean(img(3*h/4+1:h,w/4+1:w/2))) mean(mean(img(3*h/4+1:h,w/2+1:3*w/4))) mean(mean(img(3*h/4+1:h,3*w/4+1:w)))];
%figure()
imshow(uint8(Alv1))
Alv0 = mean(Alv1(:))
Alv1s = interp2(Alv1,3,'nearest');
Alv1s(9:16,:) = [];
Alv1s(:,9:16) = [];
Alv1s(17,:)=[];
Alv1s(:,17)=[];
Plv2= double(Alv2)-Alv1s
Alv0s = Alv0*ones(size(Alv1));
Plv1 = Alv1-Alv0s
Plv0 = Alv0
%% P2 a
clear all;
F = [8 4 
    2 1];
H =1/sqrt(2)*[1 1;1,-1];
T =H*F*H'
%b
Hi = inv(H)
F2=  Hi*T*H

