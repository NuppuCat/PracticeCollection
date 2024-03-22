function HistogramMatching(A,B)
y = zeros(256,1);
s = zeros(256,1);
im1 = B;
im2 = A;
for i = 0:255
    
    b = (A==i);
    num = sum(b(:));
    y(i+1)= num/length(A(:));
    s(i+1)= sum(y(1:i+1));
    c= uint8((256-1)* s(i+1) + 0.5);
    im1(find(B==i)) = c;
    im2(find(A==i)) = c;
end
figure;
subplot(221), imshow(uint8(A)), title('Original Image');
subplot(222), imhist(uint8(A)), title('Histogram');
subplot(223), imshow(uint8(im2)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(im2)), title('Histogram');
figure;
subplot(221), imshow(uint8(B)), title('Original Image');
subplot(222), imhist(uint8(B)), title('Histogram');
subplot(223), imshow(uint8(im1)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(im1)), title('Histogram');
end