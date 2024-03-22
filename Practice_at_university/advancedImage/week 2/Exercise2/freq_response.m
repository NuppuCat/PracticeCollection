function bn = freq_response(n)
bn=0;
% Write your own implementation of frequency response here
if n==1
    [h,w] = freqz(1);
    bn= [20*log10(abs(h))];
elseif n==3
    [h,w]=freqz([4/6 1/6 1/6]);
    bn= [20*log10(abs(h))];
end
end 


