% lms.m - Least mean squares algorithm
%
% Inputs:
% d       - the vector of desired signal samples of size Ns
% u       - the vector of input signal samples of size Ns
% mu      - the step-size parameter
% M       - the number of taps in filter
%
% Output:
% e - the output error vector of size Ns
% w - the last tap weights
% Wt - a matrix M x Ns containing the coefficients (their evolution)

function [e, w, Wt] = nlms(d, u, mu, M,a)

Ns = length(d);

if (Ns ~= length(u)) 
    return; 
end

% pad signal with zeros for first iterations
u = [zeros(M-1, 1); u];

w = zeros(M,1);

Wt = zeros(M, Ns);

y = zeros(Ns,1);
e = zeros(Ns,1);

for n = 1:Ns
    
    % get data
    uu = u(n+M-1:-1:n);
    %disp(size(uu))
    % filter (convolution)
    y(n) = w'*uu;
    
    % get error
    e(n) = d(n) - y(n);
    
    % use error to update filter coefficients
    u1 = uu'*uu;
    w = w + mu/(a+u1)*e(n)*uu;
    
    % store the evolution of the filter weights
    Wt(:, n) = w;
    
end