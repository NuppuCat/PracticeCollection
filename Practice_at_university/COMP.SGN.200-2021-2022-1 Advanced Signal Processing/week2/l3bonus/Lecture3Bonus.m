%% Part 1. Experiment setup for Wiener filter example

% Generate all signals involved, estimate the autocorrelation values and
% cross-correlation values involved in the expression of the Wiener filter
% and then compute the filter parameters (pure simulation approach)
clear all;
close all;

N = 100000;
d = zeros(N,1);
x = zeros(N,1);

%% Generate independent random variable v_1

% Generate N i.i.d (independent and identically distributed) random variables with distribution N(0,0.27)

v1= sqrt(0.27)*randn(N,1);

%% Generate the variable d

a =  0.8458;
d(1) = 0;
for n= 2:N
    d(n) = -a*d(n-1) +v1(n);
end

%% Generate the variable x

b = -0.9458;
x(1) = 0;
for n= 2:N
    x(n) = -b*x(n-1) +d(n);
end

%% Generate the variable u

v2 = sqrt(0.1)*randn(N,1);
u = x(:) + v2(:);

%% Compute correlations
% autocorrelation values
k = (N/4):N;  % interval over which to compute the correlations (we leave out 
% the initial quarter values obtained by simulation, to avoid the transitory 
% regim induced by the particular initial values selected in the simulation)

for tau = 0:20
    r_u1(tau+1) = sum(u(k).*u(k-tau))/length(k);
end
r_u1
% cross-correlation values
for tau = 0:20
    r_ud1(tau+1) = sum(d(k).*u(k-tau))/length(k);
end
r_ud1

%% Wiener_Hopf design for a filter of order 2
% estimated autocovariance matrix
hatR = [r_u1(1) r_u1(2);
    r_u1(2) r_u1(1)]
w2 = hatR\[r_ud1(1); r_ud1(2)]

%% Wiener_Hopf design for a filter of order 3

R = [r_u1(1) r_u1(2) r_u1(3);
    r_u1(2) r_u1(1) r_u1(2);
    r_u1(3) r_u1(2) r_u1(1)]
w3 = R \[r_ud1(1); r_ud1(2); r_ud1(3)]

%% Wiener_Hopf design for a filter of order 4
R = [r_u1(1) r_u1(2) r_u1(3) r_u1(4);
    r_u1(2) r_u1(1) r_u1(2) r_u1(3) ;
    r_u1(3) r_u1(2) r_u1(1) r_u1(2) ;
    r_u1(4) r_u1(3) r_u1(2) r_u1(1)];
w4 = R \[r_ud1(1); r_ud1(2); r_ud1(3); r_ud1(4)]

%% Wiener_Hopf design for a filter of order 5
R = [r_u1(1) r_u1(2) r_u1(3) r_u1(4) r_u1(5);
    r_u1(2) r_u1(1) r_u1(2) r_u1(3) r_u1(4);
    r_u1(3) r_u1(2) r_u1(1) r_u1(2) r_u1(3) ;
    r_u1(4) r_u1(3) r_u1(2) r_u1(1) r_u1(2);
    r_u1(5) r_u1(4) r_u1(3) r_u1(2) r_u1(1)];
w5 = R \[r_ud1(1); r_ud1(2); r_ud1(3); r_ud1(4); r_ud1(5)]

%% Plot the waveforms

figure(1),plot(v1(1:200))
title('v1 are normal distributed i.i.d samples')
xlabel('index k')
ylabel('the value u(k)')
grid
figure(2),plot(d(1:200))
title('d is a first order AR(1) process')
xlabel('index k')
ylabel('the value d(k)')
grid
figure(3),plot(x(1:200))
title('x is a second order AR(2) process')
xlabel('index k')
ylabel('the value x(k)')
grid
figure(4),plot(u(1:200))
title('u is a second order AR(2) process plus noise')
xlabel('index k')
ylabel('the value u(k)')
grid
figure(5),plot(0:20, r_u1,'or-',0:19, r_u1,'ob-')
title('r_u is the correlation of a second order AR(2) process plus noise')
xlabel('index \tau')
ylabel('the value r_u(\tau)')
legend('r_u(\tau)','r_x(\tau)')
grid

%% Questions (1 point)
% Q1: The dependency of d(n) on v1(n) is through a 
% a) first order difference equation
% b) second order difference equation
% c) third order difference equation
% Q2: The dependency of x(n) on v1(n) is through a 
% a) first order difference equation
% b) second order difference equation
% c) third order difference equation
% Q3: The output chanel u(n) is
% a) only a linearly distorted version of d(n)
% b) a linearly distorted version of d(n) corrupted by the noise v1(n)
% c) a linearly distorted version of d(n) corrupted by the noise v2(n)


%% Part 2: Simulation of the filtering by Wiener-Hopf filter

%% Generate the output of  the filter y for parameters w2
y2 = zeros(N,1);
for n= 2:N
    y2(n) = w2(1)*u(n) + w2(2)*u(n-1);
end

% Compute the criterion obtained by Wiener-Hopf filter w2

k = (N/4):N;  % interval over which to compute the correlations
Jcrit(2) = sum((d(k)-y2(k)).^2 )/length(k)

%% Generate the output of  the filter y for parameters w5
y5 = zeros(N,1);
for n= 5:N
    y5(n) = w5(1)*u(n) + w5(2)*u(n-1)+ w5(3)*u(n-2)+ w5(4)*u(n-3)+ w5(5)*u(n-4);
end

% Compute the criterion obtained by Wiener-Hopf filter w2

k = (N/4):N;  % interval over which to compute the correlations
Jcrit(5) = sum((d(k)-y5(k)).^2 )/length(k)

%% Compare the output of the filter w2 and of the filter w5, both against d

k1 = (N-100):N;
figure(6), plot(k1,y2(k1),'or-',k1,d(k1),'db-'),xlabel('sample i'),ylabel('y_2(i),d(i)')
legend('y_2(i)','d(i)') 


figure(7), plot(k1,y2(k1),'or-',k1,y5(k1),'db-', k1,d(k1),'dg'),xlabel('sample i'),ylabel('y_2(i),y_5(i),d(i)')
legend('y_2(i)','y_5(i)','d(i)') 

%% Questions (1 point)
% Q4: Compare the values w2, w3,w4,w5 of the optimal Wiener-Hopf filters
% obtained for various filter orders filters  (M= 2,3,4,5). The longest (at
% M=5) is y5(n) = w5(1)*u(n) + w5(2)*u(n-1)+ w5(3)*u(n-2)+ w5(4)*u(n-3)+ w5(5)*u(n-4)
% and the shortest (at M=2) is y2(n) = w2(1)*u(n) + w2(2)*u(n-1);
% Which filter has a smaller criterion sum( (d(n)-y(n))^2) over the simulated data: 
% a) The filter of order 5, because it can be tuned better to the data. It has more degrees of freedom than the filter of order2, since the optimal filter of order 2 can be written as a particular 5th order filter
%  w = [w2(1);w2(2);0;0;0]
% b) The filter of order 5, because it combines linealrly more samples from the past
% u(n),u(n-1), ..u(n-4) as compared to the filter w2, and hence its output can become
% closer to d(n) 
% c) The filter of order 2, because has less parameters that need to be
% learned from the data, so they are more precise and give better
% performance sum( (d(n)-y(n))^2)

% Q5: Comparing w2 to w3, we see that the extra-parameter in w3, w3(3) is rather small. 
% Also, comparing w3 to w4, we see that the extra-parameter in w4, w4(4) is rather small. 
% This is true also between w4 and w5. That can be explained by the
% following arguments? (choose one or more true answers)
% a) The values of the input u(n),u(n-1), u(n-2) are strongly correlated
% (as proven by the autocorrelation coefficients of r_u1 = [1.1131
% 0.50542       0.8642      0.49141      0.74289 ...]). It means that using
% only  u(n),u(n-1) in the filter one obtains very significant filter
% coefficients w2(1), w2(2), but including additionally u(3) one cannot get a
% much different filter, so the term w2(3)*u(3) shouldn't be too large,
% which makes w2(3) small. However, it includes useful information, making
% the criterion smaller.
% b) The channel output u(n) is obtained in the simulations as  u(n) = x(n) +
% v2(n); x(n) = -b*x(n-1) +d(n); hence u(n) depends directly on d(n). 
% Any filter that computes y(n) = w(1)u(n) will be a function of u(n), 
% and thus of d(n) as well. So we can tune the parameter w(1) so that y(n) 
% is close to d(n), and we don't need any other parameeters w(2),w(3),w(4),etc. 
% All these parameters are just un-important random numbers depending on the noise realization used in simulation.
% c) The estimated correlations r_u1 = [1.1131; 0.50542; 0.8642; 0.49141; 0.74289 ...])
% are close to the theoretical ones found in the lecture ru = [ 1.1; 0.5;
% 0.85; 0.485; 0.7285], and hence they are not merely some random effect
% dependent on the current data. Also, one can show the same for the
% cros-correlations Eu(n-i)d(i). Hence all the coefficients in the Wiener
% filter are significant and don't depend on the simulated data too much. 

%% Questions (1 point)

% Q6: In Figure 6 the output y(n) of the filter w2 compares to the desired
% value d(n) in the following way:
% a) It cannot follow d(n) too closely. We have to run the algorithm for
% longer sequence of data to get a really significantly better  behavior
% b) It follows closely the values of d(n) when d(n) changes its sign , but
% it overshoots   when d(n) varies more smoothly 
% c) The errors between d(n) and y(n) are very large with respect to the
% true values of d(n)

% Q7: In Figure 7 we have the outputs of the filters w2 and w5, also the
% value of d(n). One can interpret the curves as:
% a) the outputs y_2(i) and y_5(i) are close one to another and they are alternating in winning the closest position to d(i)
% b) not all the time |y_5(i)-d(i)| is smaller than  |y_2(i)-d(i)|, although Jcrit(5) is smaller than Jcrit(2)
% c) Combining the filters y_2 and y_5 in a linear way as a*y_2+b*y_5 might improve the performance, picking all the time among y_2(n) or y_5(n) the one closer to d(n)

%% Part 3 Analytical Wiener_Hopf design for a filter of order 2 
% Correlations computed using Yule Walker (NOT based on simulated data)

%% Compute the correlation values from the Yule-Walker equation
% (See the matrix form at the bottom of page 19)

a1 = -0.1;
a2 = -0.8;
sigma2_v = 0.27;
sigma2_v2 = 0.1;
B=  [1 a1 a2
     a1 1+a2 0
     a2 a1 1];
C = [sigma2_v; 0; 0];
r_x = B\C

r_u(1:2,1) = r_x(1:2,1) + [sigma2_v2; 0];

p(1,1) = r_x(1) + b*r_x(2);
p(2,1) = r_x(2) + b*r_x(1)

%% Wiener_Hopf design for a filter of order 2

w = [r_u(1) r_u(2); r_u(2) r_u(1)]\[p(1); p(2)]

%% Question (2 points)
% Q8: Perform similarly to the above Part 3 all evaluations needed for the analytical
% Wiener_Hopf design  for a filter of order 5
% Hint: start again from the Yule Walker equations, but written for a 5x5
% matrix
% For checking the validity, compare your analytical auto- and cross-correlation results to the autocorrelation values and
% the crosscorrelation values obtained by simulation r_u1 and r_ud1 in Part 1
% Also check against w4 obtained in Part 1.
p5 =xcorr(d,u,4,'unbiased');
p5 = p5(5:end);
r5 = xcorr(u,u',4,'unbiased');
r5 =r5(5:end);
R5 = toeplitz(r5);
w55= R5\p5
w5

 