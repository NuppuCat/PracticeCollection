function histequal(im)
y = zeros(256,1);
s = zeros(256,1);
im1 = im;
for i = 0:255
    
    b = (im==i);
    num = sum(b(:));
    y(i+1)= num/length(im(:));
    s= sum(y(1:i+1));
    c= uint8((256-1)* s + 0.5);
    im1(find(im==i)) = c;
end


figure;
subplot(221), imshow(uint8(im)), title('Original Image');
subplot(222), imhist(uint8(im)), title('Histogram');
subplot(223), imshow(uint8(im1)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(im1)), title('Histogram');
end