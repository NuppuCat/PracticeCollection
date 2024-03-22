function [K1,img_result] =  WMF(I, S,m)
%扩增数列，每个元素重复权重个，以改变中位数
K1= medfilt2(I,[S,S]);


img = I;
%图片尺寸
[M , N] = size(img);
img_result = zeros(M, N);
expand_size = floor(S / 2);
%YEXT= wextend(TYPE,MODE,X,LEN)
expand_img = double(wextend('2D','zpd', img, expand_size));
m  = m(:);
for i=1:M
    for j=1:N
        mat = expand_img(i:i+S-1,j:j+S-1) ; 
        mat = mat(:);
        mat1 = [];
        for p  = 1:length(mat)
            ip = repmat(mat(p),1,m(p));
            mat1 = [mat1 ip];
        end
        %mat1
        img_result(i,j) = median(mat1);
        
    end
end

img_result = uint8(img_result);%转int8，图像





end