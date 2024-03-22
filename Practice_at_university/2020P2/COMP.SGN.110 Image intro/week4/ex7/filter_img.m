function filter_img(g, m,n)
g = double(g);
f1 = imfilter(g,fspecial('average',[m n]));
f2 = exp(imfilter(log(g),ones(m,n),'replicate')).^(1/(m*n));
f3 = (m*n) ./ imfilter(1 ./ (g + eps), ones(m, n), 'replicate');
figure;
subplot(221), imshow(uint8(g)), title('original');
subplot(222), imshow(uint8(f1)), title('f1');
subplot(223), imshow(uint8(f2)), title('f2');
subplot(224), imshow(uint8(f3)), title('f3');
end