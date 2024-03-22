function c = spearmanCorr(s1, s2)
% SPEARMANCORR - computes Spearman rank order correlation coefficient
%   (SROCC) between two series.
%
%   s1 - first series vector;
%   s2 - second series vector;
%   c - correlation coefficient (known as SROCC)
%

%%% start your code here
n=length(s1);
s1 =s1(:);
s2 = s2(:);
a =6*sum((s1-s2).^2)/(n*(n-1)*(n+1));
c = 1-a
%%% end your code here 

end
