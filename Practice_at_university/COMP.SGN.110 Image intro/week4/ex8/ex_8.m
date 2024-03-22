%%
clear all;
close all;
disp('task1')
I = imread('DIP.jpg');
[row,col] = size(I);
[u, v] = meshgrid(-row/2:row/2-1, -col/2:col/2-1);
b = 0.1;
a = 0.1;
A = (a.*u+b.*v);
H = 1./pi./A.*sin(pi.*A).*exp(-1i*pi.*A);
H(A==0)=1;
F = fft2(I);
Fc = fftshift(F);
G = Fc.*H;
gi = ifft2(ifftshift(G));
g = real(gi);
figure;
colormap('gray')
imagesc(g)
%imshow(uint8(g))
t = G./H;
g2 = real(ifft2(ifftshift(t)));
figure;
colormap('gray')
imagesc(g2)
figure;
colormap('gray')
subplot(131), imagesc(I), title('Original Image');
subplot(132), imagesc(g), title('blurred');
subplot(133), imagesc(g2), title('restored');
err1 = immse(double(I),g)
err2 = immse(double(I),g2)


%%
disp('task2')
%J = imnoise(g,'gaussian',0,50);
y = (50^0.5).*randn(688,688);
J = g+y;

%N = fftshift(fft2(J - double(I)));
N = fftshift(fft2(y));
Fj = fft2(J);
Fcj = fftshift(Fj);
Gj = Fcj./H;
gij = ifft2(ifftshift(Gj));
gj = real(gij);

Fw = Fcj./H.*(abs(H).^2./(abs(H).^2+(abs(N).^2)./(abs(Fc).^2)));
gw = real(ifft2(ifftshift(Fw)));
figure;
colormap('gray')
subplot(131), imagesc(J), title('Original Image');
subplot(132), imagesc(gj), title('inverse');
subplot(133), imagesc(gw), title('wiener');
disp('because noise are added before inverse, as the transfer, noise will be strech')
k = [0.001 0.005 0.01 0.015 0.03];
for i = k
    
Fwk = Fcj./H.*(abs(H).^2./(abs(H).^2+i));
gwk = real(ifft2(ifftshift(Fwk)));
figure;
colormap('gray')
imagesc(gwk), title(i);
end





