%%
disp('task1')
I = imread('fruits.jpg');
I2 = imread('festia.jpg');
[Ih1,Ihv1] =  intensityeq(I);
[Ih2,Ihv2] =  intensityeq(I2);

figure;
subplot(231), imagesc(I), title('I');
subplot(232), imagesc(Ih1), title('histeqrgb');
subplot(233), imagesc(Ihv1), title('intenstyeq');
subplot(234), imagesc(I2), title('I2');
subplot(235), imagesc(Ih2), title('histeqrgb');
subplot(236), imagesc(Ihv2), title('intenstyeq');
disp('rgb histeq changed color of the image, while hsv only change v the light(gray) level')
%%
disp('task2')
L = rgb2hsv(imread('lake.jpg'));
%choose a region from aim
c = (L(300:420,300:420,:));
a = mean(mean(c));
stda = std(std(double(c)));
I = L;
for i = 1:640
   for j = 1:640 
        o = (L(i,j,1));
        %compare hue
        t = abs(o-a(:,:,1));
        if t>stda(:,:,1)*19.5
            I(i,j,:)=0;
        end
            
   end
    
end

figure;
imagesc(hsv2rgb(I)),title('blue region');


%get threshhold
level = graythresh(I(:,:,1));
BW = im2bw(I(:,:,1),level);
L1 = bwlabel(BW);
L2 = L1;
%find 9 is the aim region
L2(L2~=9)=0;
L2(L2==9)=1;

I= L2.*I;
figure;
imshow(L2);
figure;
imagesc(hsv2rgb(L(300:420,300:420,:))),title('aim region');

figure;
imagesc(hsv2rgb(I)),title('biggest lake');

