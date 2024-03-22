function directional_filtering(I, size)
a = zeros(size);
a0 = a;
a0(int32(size/2),:)=1;
a0 = a0/size;
a1 = fliplr(eye(size))/size;
a2 = a;
a2(:,int32(size/2))=1;
a2 = a2/size;
a3 = eye(size)/size;
I0 = imfilter(I,a0);
I1 = imfilter(I,a1);
I2 = imfilter(I,a2);
I3 = imfilter(I,a3);

figure;
subplot(221), imshow(uint8(I+I0)), title('0');
subplot(222), imshow(uint8(I+I1)), title('45');
subplot(223), imshow(uint8(I+I2)), title('90');
subplot(224), imshow(uint8(I+I3)), title('135');
%figure;
%imshow(I+I0+I1+I2+I3)
end