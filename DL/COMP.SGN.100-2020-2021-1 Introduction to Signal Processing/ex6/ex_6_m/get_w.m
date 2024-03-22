function [w,c] = get_w(X1,X2)
c1 = cov(X1);
c2 = cov(X2);
u1 = mean(X1);
u2 = mean(X2);
w = inv(c1+c2)*(u1-u2)';
c = (w'*u1'+w'*u2')/2;

end