%% Uniform distribution.

% u1 is a vector containing 100000 pseudorandom variables from an uniform
% distribution.
u1 = rand(100000, 1);

% The function hist can be used to count the number of values that fit in
% M=50 bins.
M = 50;

figure(1),clf 
hist(u1, M);
ylabel('Value count'); 
xlabel('Values');


%------------------------------------------------------------------------
% U is an 100x1000 matrix containing 100000 pseudorandom variables from an
% uniform distribution.
U = rand(100, 1000);

% We use the same hist function to count the values of the first row from U.
figure(2),clf
hist(U(1, :), M);
ylabel('Value count'); 
xlabel('Values');

% Question: Why does the histogram look different now? Why aren't the
% values distributed more or less uniformly in the bins?


%------------------------------------------------------------------------
% u2 is a vector containing 100000 pseudorandom variables from an uniform
% distribution.
u2 = -4 + 6 * rand(100000, 1);

% The function hist is used to count the number of values.
figure(3),clf
hist(u2, M);
ylabel('Value count'); 
xlabel('Values');

% Question: What interval corresponds to the distribution?

