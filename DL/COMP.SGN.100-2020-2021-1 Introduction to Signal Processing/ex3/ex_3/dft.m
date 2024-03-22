function dft = dft(x)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
N = length(x);
F=exp(-2*pi*1i*(0:N-1)'*(0:N-1)/N); 
dft = F*x;
end

