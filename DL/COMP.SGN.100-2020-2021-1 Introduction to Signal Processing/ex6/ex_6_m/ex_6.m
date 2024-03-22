%%
%Task6
disp('task6')
clear;
load laughter
z = zeros(2*length(y),1);
z(1:2:end) = y;
%soundsc(z,8192*2)
%soundsc(y,8192)
a = fir1(100,0.5,'DC-1');
zf = filter(a,1,z);
%soundsc(zf,8192*2);
figure(1)
spectrogram(y)
figure(2)
spectrogram(z)
figure(3)
spectrogram(zf)
%%
%Task7
disp('task7')
b = fir1(100,1/3,'DC-1');
zf1 = filter(b,1,zf);
zr = zf1(1:3:end);
%soundsc(zr,5461);
figure(4)
spectrogram(zf1,2*8192)
figure(5)
spectrogram(zr,5461)
%%
%Task8
disp('task8')
clear;
load testdata_fisher.mat
figure(6)
clf
plot(X1(:,1),X1(:,2),'ro')
hold on
plot(X2(:,1),X2(:,2),'gx')
[w,c] = get_w(X1,X2);

line([0,w(1)],[0,w(2)]);
 axis equal
hold off
%%
%Task9
disp('task9')


r1 = X1*w;
r2 = X2*w;
p = (sum(r1>c)+sum(r2<c))/(length(X1)+length(X2));
disp('percentage of the samples are classified correctly is')
disp(p*100)
%%
%Task10
disp('task10')
clear;
imOrig = imread('canoe.jpg');
figure(7)
imshow (imOrig, []);
[x1,y1] = ginput(1);
[x2,y2] = ginput(1);
window1 = imOrig(y1-3:y1+3, x1-3:x1+3,:);
window2 = imOrig(y2-3:y2+3, x2-3:x2+3,:);
X1 = double(reshape(window1, 49, 3));
X2 = double(reshape(window2, 49, 3));
[w,c] = get_w(X1,X2);
imGray = w(1)*double(imOrig(:,:,1)) +...
w(2)*double(imOrig(:,:,2)) +...
w(3)*double(imOrig(:,:,3));
figure(8);
imshow(imGray, []);
figure(9)
imshow(imGray > c, []);