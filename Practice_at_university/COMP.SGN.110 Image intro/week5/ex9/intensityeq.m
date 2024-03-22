function [Ih2,Ih] =  intensityeq(I)
Ic = rgb2hsv(I);

v = Ic(:,:,3);

vh = histeq(v);
Ih= Ic;
Ih(:,:,3)=vh;
Ih = hsv2rgb(Ih);

r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
rh = histeq(r);
gh = histeq(g);
bh = histeq(b);
Ih2(:,:,1) = rh;
Ih2(:,:,2) = gh;
Ih2(:,:,3) = bh;

end