%% Random variables
%% Part 1: (0.5 point) Generate independent random variables with uniform distribution
% Generate n=100 i.i.d (independent and identically distributed) random variables with distribution U(a,b)
n = 10000
u=4+6*rand(n,1);
figure(1)
plot(u,'o--')
title('100 uniform distributed i.i.d samples')
xlabel('index k')
ylabel('the value u(k)')
grid
% Estimate the mean of the generated variables 
estim_mean_u = sum(u)/n
% Estimate the standard deviation of the generated variables
estim_stand_dev_u = std(u)

% Q1: What is the range (a,b) of the generated (simulated) values u(1),...,u(1000)?  

% Q2: What is the mean value for data with distribution  U(a,b) (check the estimated value estim_mean_u)?

% Q3: What is the standard deviation of the distribution U(a,b) (check the estimated value estim_stand_dev_u)?

%% Part 2: (0.5 point) Repeat the above process for different values of n 
%% Study the new random variable RV = estim_mean_u = sum(u)/n from Part 1

% Several values for n used in the simulation study:
n1= [10 20 50 100 200 500 1000 2000 5000 10000 20000 50000]
std_dev_RV = []; true_std_dev_RV=[];
for i1 = 1:12
    n = n1(i1);
    % repeat for 10000 realizations
    RV = zeros( 10000, 1);
    for nrealiz = 1:10000
        % generate n independent values distributed as U(a,b)
        u = 4 + 6 * rand(n,1);
        % estimate the mean of the generated values
        RV(nrealiz) = sum(u)/n; % a new  value of the random variable
    end
    % check what was the estimated standard deviation of RV, based on data
    std_dev_RV(i1) = std(RV);
    % check some candidate theoretical formula
    std_dev_u = 6/sqrt(12);
    true_std_dev_RV(i1) = std_dev_u/sqrt(n);
    std_dev_RV(i1)/true_std_dev_RV(i1)
end

figure(2),plot(n1,std_dev_RV,'ob-'), grid on, xlabel('n'),ylabel('std.dev.(RV)')
figure(3),plot(log10(n1),log10(std_dev_RV),'ob-'),grid on, 
xlabel('log_1_0 n'),ylabel('log_1_0 std.dev.(RV)')

figure(4),plot(n1,std_dev_RV,'ob-', n1,true_std_dev_RV,'dr-'), grid on, xlabel('n'),ylabel('std.dev.(RV)')
figure(5),plot(log10(n1),log10(std_dev_RV),'ob-'),hold on
plot(log10(n1),log10(true_std_dev_RV),'dr-')
grid on, xlabel('log_1_0 n'),ylabel('log_1_0 std.dev.(RV)')
figure(6),plot(log10(n1),std_dev_RV./true_std_dev_RV,'ob-',0,0),grid on,  xlabel('log_1_0 n')

% Q4: In Figure 2 one can see the estimated standard deviation of the RV as
% a function of te number of datapoints, n
% In Figure 3 one can see the same, but with logarithmic scales. From
% Figure 3 one can infer that a) std_dev(RV) = const/n or b)std_dev(RV) = const*n
% or c) std_dev(RV) = const1*log10(n)+const2

% Q5: In Figure 4 and 5 one can check the matching of the formula derived
% using handling of the expectation operator with the simulated data
% presented in the pdf
% The precision of match between the simulated data and theoretical formula
% is:
% a) Better at low n, because n is on denominator b) Better at high n because we have more data for estimating parameters
% c) Anyway, quite the same precison for all n values, because the number of realizations are the same at each n

%% Part 3. (0.5 point) Generate a discrete random variable having values in the alphabet {1,2,3,4,5,6}
N= 1200000;
u=6*rand(N,1); % generate N = 1200000 uniform distributed values in [0,6)
ui = 1+floor(u); % quantize u to integer values
figure(4)
plot(ui(1:100),'or')
title('100 throwings of a dice')
xlabel('index k')
ylabel('the value u(k)')
% number of occurences of {1,2,3,4,5,6}
hc = hist(ui,min(ui):max(ui))
normalized_histogram = hc/sum(hc)
normalized_histogram-1/6
cumulative_histogram(1) = normalized_histogram(1);
for i = 2:6
    cumulative_histogram(i) = cumulative_histogram(i-1) + normalized_histogram(i);
end
figure(7)
plot(1:6,hc,'or',0,0,[0 6],N/6*[1 1],'b--')
xlabel('value on the dice')
ylabel('number of occurences')

%% Questions 2.1-2.3
% Q2.1. What is the ideal probability mass function of the random variable ui
%       {p(1),p(2),p(3),p(4),p(5),p(6)} ?
%
% Q2.2. What is the estimated  probability mass function of the random variable ui, when N = 1200000?
%    {hat_p(1),hat_p(2),hat_p(3),hat_p(4),hat_p(5),hat_p(6)} ?
%
% Q2.3 What is the ideal cumulative distribution function F(ui) = Prob(Ui<=ui)?

%% Part 4. (1.5 point) Compute the sum of two consecutive dice throwings
for i = 1:(N/2-1)
    uip(i) = ui(1+2*i) + ui(2+2*i); % this is the sum of two dice
    uip1(i) = ui(1+2*i);
    uip2(i) = ui(2+2*i);
end
figure(8)
plot(uip(1:100),'or')
title('100 throwings of a pair of dice')
xlabel('index k')
ylabel('the value u(k)')
% number of occurences of {2,3,4,5,6,7,8,9,10,11,12}
hc2 = hist(uip,min(uip):max(uip))
figure(9)
plot(min(uip):max(uip),36*hc2/sum(hc2),'or')
grid on
xlabel('sum of values on two dice')
ylabel('36*empirical probability')
 
% Variance of uip
N1 = length(uip);
muip = sum(uip)/N1
C1 = sum((uip-muip).^2)/length(uip)

% Correlation of uip with uip1, and uip with uip2

muip1 = sum(uip1)/N1
muip2 = sum(uip2)/N1
C1 = sum((uip1-muip1).*(uip-muip))/length(uip)
C2 = sum((uip2-muip2).*(uip-muip))/length(uip)

% Correlation of uip1 with uip2
C3 = sum((uip1-muip1).*(uip2-muip2))/length(uip)


%% Questions (1.5 points) to be answere as an essay
% 3.1. What is the ideal probability mass function
%       {p_uip(2),p_uip(3),...,p_uip(12)} ?
%
% 3.2. What is the estimated probability mass function
%       {hat_p_uip(2),hat_p_uip(3),...,hat_p_uip(12)} ?
%
% 3.3. What is the mean value of uip?
%       Hint: mean_val = p_uip(2)*uip(2) +... p_uip(12)*uip(12)?
%
% 3.4. What is the variance of uip?
%       Hint: var(uip) = p_uip(2)*(uip(2)-mean_val)^2 + ...  p_uip(12)*(uip(12)-mean_val)^2
%
% 3.5 Can you evaluate the ideal values of C1, C2 and C3?   


