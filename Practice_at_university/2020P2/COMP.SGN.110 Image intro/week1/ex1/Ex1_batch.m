function  Ex1_batch()

for i=[1:6]
im = imread(strcat('c_',num2str(i),'.jpg'));
pim = Process(im);
imwrite(pim,strcat('c_',num2str(i),'.bmp'));
end

end
function rfhB = Process(I)
B = imresize(I,0.75);
[m,n,c] = size(B);
hB = B(:,(ceil(n/2):n),:);
fhB = fliplr(hB);
rfhB = imrotate(fhB,90);
end
