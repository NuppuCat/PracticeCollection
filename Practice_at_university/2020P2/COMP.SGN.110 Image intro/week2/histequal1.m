function histequal1(image)


[height, width] = size(image); 
 
y = zeros(256,1);
s = zeros(256,1);

for i = 0:255
    
    b = (image==i);
    num = sum(b(:));
    y(i+1)= num/length(image(:));
   % s(i+1)= sum(y(1:i+1));
    %im1(find(im1==i))= uint8(round((256-1)* s(i+1) + 0.5));
   
end

CumPixel = cumsum(y);  
CumPixel = uint8((256-1) .* CumPixel + 0.5); 
 

outImage = uint8(zeros(height, width));  

for i = 0 : 255
   outImage(find(image==i)) = CumPixel(i+1);
end

CumPixel = cumsum(y);
CumPixel = uint8((256-1).* CumPixel + 0.5);


for i = 0 : 255
   im1(find(im==i)) = CumPixel(i+1);
end
figure;
subplot(221), imshow(uint8(image)), title('Original Image');
subplot(222), imhist(uint8(image)), title('Histogram');
subplot(223), imshow(uint8(outImage)), title('Cont. Stretched Image');
subplot(224), imhist(uint8(outImage)), title('Histogram');
end