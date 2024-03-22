R = [2 1
    1 2] %autocorrelation matrix 
p = [6
    4] %cross correlation vector
mu = 0.1; % step size
N = 1000; %number of iterations
w = [0
    0]; %initial values of the filter coefficients

w_save = steepest_descent(R,p,w,mu,N); %performs N iterations of steepest descent

w_save(:,end) % result of the Nth iteration
%%
w_w = R\p;
wm=w_save(:,10)-w_w;