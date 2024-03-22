function I2 = ContrastStretch(I)
m = max(I(:));
mi = min(I(:));
I2 = (I-mi)*(255/(m-mi));

end