clear all;
clc;
x = [8, 16, 24, 32, 46, 50, 58, 64];
fx = fft(x);
cx = dct(x);
ifx = ifft(fx);
icx = idct(cx);