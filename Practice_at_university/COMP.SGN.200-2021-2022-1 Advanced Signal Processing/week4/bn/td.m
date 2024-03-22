N1  = [2*10^5 5*10^4 2*10^4];%  three values for the length of data
% a = 0.98  is high, but not extremely close to 1
a  = 1.3; % a is also called the correlation coefficient

max_tau = 2000; % we estimate r(0),r(1),...,r(max_tau)

% The analytic expression of the autocorrelation function
r_true(1+(0:max_tau)) = (a.^(0:max_tau))/(1-a^2);

colors  = ['r-';'g-';'m-';'b-']
figure(2), clf,plot(0:max_tau,r_true,'b-'), hold on

for len = 1:3 
    N = N1(len);        % Maximum value of discrete time
    % Nt = floor(N/4);  % The time series is cropped from Nt,Nt+1,..., N
    Nt = 2*10^4/4;      % Let's neglect the same initial segment, 1...Nt, for all three cases
    e  = randn(N,1);
    y1 = 0.;
    
    for ii = 2:N
        y1(ii) = a*y1(ii-1) + e(ii); % generate the AR(1) process
    end
    y = y1(Nt:N);
    Ny = length(y);
    %
    % Estimate the autocorrelation function from data, knowing that Ey = 0
    %
    r = zeros(max_tau+1,1);
    for tau = 0:max_tau
        % if(rem(tau,100)==0),[tau max_tau],end % show the iteration progressing
        time_segment = 1:(Ny-tau);
        r(tau+1) = sum(y(time_segment).* y(tau+time_segment))/(Ny-tau);
    end
    %
    plot(0:max_tau,r,colors(len,:))
end
title(['Autocorrelation function, a = ' num2str(a)],'Fontsize',14)
xlabel('lag \tau','Fontsize',12)
ylabel('r(\tau)','Fontsize',12)
legend('true r','N=2000000','N=200000','N=20000')  

%% Question
% Background
% We consider here the following helpful results, proven in the pdf for
% this task
% (1) a^50 = 0.36; a^225 = 0.01; a^2000 = 2.8324e-18; a^5000=1.3501e-44;
% (2) Ey_ny_{n-k} = a^k*(1-a^{2n-2k})/(1-a^2)
% (3) In the stationary regim r_k = a^k/(1-a^2), which  is called here true
%       autocorrelation
%
% Question 1: State which of the following statements are true:
% a) For the first 50 correlation values the estimates are  reasonably close to the
% true value, for all three cases of N
% b) for the case N = 2*10^4 the estimates for large lags tau are not
% reliable, because there are not enough data points 
% c) The efect of non-stationarity shown in (2) makes the estimates at low
% N unreliable

%% Part 3: Discuss the effect of accuracy of estimating the autocorrelation 
%% function for the adaptive noise cancellation (ANC) application
% In the ANC, the Wiener-Hopf optimal filter design requires the inversion
% of the correlation matrix. The needed order of the filter might be quite
% large for ANC (from hudreads to thousands, as you estimated it in Lecture 1,
% bonus assignment)
%
% Assume that in ANC the source of noise is the AR(1) process, from Part 1
% and the estimated autocorrelation was done as in Part 2, for N = 2*10^4
% Consider a filter of order 200.

%% Case 1: consider the true autocorrelation function r_true
% The autocorrelation matrix  R needed in Wiener-Hopf design w = R^(-1)p

R = toeplitz(r_true(1:200));
figure(3),imagesc(R), colormap(gray),title('Autocorrelation matrix R')
invR = inv(R);
figure(4),imagesc(invR), colormap(gray),title('Inverse of autocorrelation matrix R'),colorbar