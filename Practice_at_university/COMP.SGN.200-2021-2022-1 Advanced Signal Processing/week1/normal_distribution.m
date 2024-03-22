%% Normal distribution.

% n1 is a vector containing 100000 pseudorandom variables from an normal
% distribution; the call is similar with that of rand:
n1 = randn(100000, 1);


% The function hist can be used to count the number of values that fit in
% M=50 bins.
M = 50;

figure(1),clf
hist(n1, M);
ylabel('Value count'); 
xlabel('Values');

% Question: Looking at the histogram (Figure 1), what might be the average value of
% 'n1' and its variance? 

%---------------------------------------
% n2 is given using randn function in the following way:
m = 4.23;
s = 3.123;
n2 = m + s * randn(100000, 1);

figure(2),clf
hist(n2, M);
ylabel('Value count');
xlabel('Values');

% Question: What are the values of 'm' and 's' such that the vector 'n2' contains
% values from a normal distribution with mean 2 and variance 3.
