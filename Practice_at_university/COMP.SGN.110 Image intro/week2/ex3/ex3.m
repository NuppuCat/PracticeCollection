%%
%Task 1
disp('task1')
I = imread('wom1.png');
figure(1);
imshow(I)
y = zeros(256,1);
for i = [0:1:255]
    
    b = (I==i);
    num = sum(b(:));
    y(i+1)= num/256/256;
    
end

figure(2)
%plot([0:255],y);
stem([0:255],y);
Ic = ContrastStretch(I);
figure(3);
imshow(Ic)
figure(4);
imhist(Ic)
I2 = imread('man8.png');
figure(5)
subplot(2,2,1);
imshow(I2);
subplot(2,2,2);
imhist(I2);
I2c = ContrastStretch(I2);
subplot(2,2,3);
imshow(I2c);
subplot(2,2,4);
imhist(I2c);
disp('after contraststretch, the picture become better to find some detail')
%%
%Task 2
disp('task2')
% blockproc('moon.tif',[32 32],myfun);
% replace var in () to ..
%f = @(block_struct) uint8(block_struct.data(2,2)*ones(size(block_struct.data)));
f = @(block_struct) uint8(block_struct.data(2,2));
mbaboon = imread('mbaboon.bmp');
figure()
imshow(mbaboon)
bmbaboon =  blockproc(mbaboon,[4 4],f);
figure(6)
imshow(bmbaboon)
f2 = @(block_struct) uint8(block_struct.data(1,1));
bmbaboon2 =  blockproc(mbaboon,[4 4],f2);
% 二维所以是mean2
f3 = @(block_struct) uint8(mean2(block_struct.data));
bmbaboon3 =  blockproc(mbaboon,[4 4],f3);
figure(7)
imshow(bmbaboon2)
figure(8)
imshow(bmbaboon3)
disp('It is amazing, a picture can contain a lot of infromation.With different sampling function, we can get different result.')

%%
%Task 3
disp('task3')
disp(' Brightness adaptation,The visual system changes(increase in dark) its overall sensitivity ')
%%
%Task 4
disp('task4')
disp('a) I think hf will pan right c units (if c < 0 will pan left)')
disp('b) I think hf will Stretch c times (if c<1 shrink 1/c times))')
disp('c) I think hf will not change, because the intensity in statistics(overall) not change )')









