function med_filter_plus(I, S,alpha)
K1= medfilt2(I,[S,S]);
figure;
imshow(K1)

img = I;
%Í¼Æ¬³ß´ç
[M , N] = size(img);
img_result = zeros(M, N);


expand_size = floor(S / 2);


%YEXT= wextend(TYPE,MODE,X,LEN)
expand_img = double(wextend('2D','zpd', img, expand_size));

for i=1:M
    for j=1:N
        mat = expand_img(i:i+S-1,j:j+S-1) ; 
        mat = mat(:);
        if abs(median(mat)-img(i,j))>alpha
            img_result(i,j) = img(i,j);
        else
            img_result(i,j) = median(mat);
        end
    end
end

img_result = uint8(img_result);%×ªint8£¬Í¼Ïñ
figure;
subplot(1 ,2, 1);
title('original')
imshow(img)
subplot(1 ,2, 2);
imshow(img_result)
t = ['size' num2str(S) ',after filter'];
title(t)






end