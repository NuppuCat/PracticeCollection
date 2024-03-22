function [Aset] = pre(k,v,M)
%PRE 此处显示有关此函数的摘要
%   此处显示详细说明
N = length(v);
Aset=cell(k);
for i = 1:(k)
   v1 = v(1+(i-1)*N/k:i*N/k);
   temp = zeros(N/k,M);
   for j = 1:M
      temp(:,j)= [zeros(j-1,1);v1(1:N/k-j+1)] ;
   end
   Aset{i}= temp;
end
end

