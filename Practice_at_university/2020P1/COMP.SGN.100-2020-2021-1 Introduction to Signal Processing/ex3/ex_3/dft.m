function dft = dft(x)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
N = length(x);
F=exp(-2*pi*1i*(0:N-1)'*(0:N-1)/N); 
dft = F*x;
end

