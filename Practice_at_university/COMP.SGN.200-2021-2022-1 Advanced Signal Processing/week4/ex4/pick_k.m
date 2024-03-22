function [FPE_order,AIC_order,MDL_order] = pick_k(N,NrOfTrials)

Nt  = 50000;

nK = 20;


FPE_val = zeros(1, nK);
AIC_val = zeros(1, nK);
MDL_val = zeros(1, nK);

FPE_order = zeros(NrOfTrials, 1);
AIC_order = zeros(NrOfTrials, 1);
MDL_order = zeros(NrOfTrials, 1);

% The AR model
omega = pi*[0.90 0.70 0.50 0.30 0.10];
rho   =    [0.75 0.95 0.85 0.80 0.90];
npairs = length(omega);
roots = zeros(2*npairs,1);
roots(1:npairs) = rho.*exp(1i*omega);
roots(npairs+1:end) = conj(roots(1:npairs));
AR = poly(roots);

for i=1:NrOfTrials
    
    % One realization of AR process of length Nt samples
    e = randn(Nt,1);
    [y,z] = filter(1, AR, e);
    e2 = randn(N,1);
    y = filter(1, AR, e2,z);
    y = filter(1, AR, e);
    y = filter(1, AR, e2);
    % Pick N samples from the realization
    y_1 = y(end-N+1:end);
    
    % Order estimation loop
    for k_estim=1:nK
        
        % We use Yule-Walker method to obtain an estimate of the all-pole
        % AR parameters
        [~, E] = aryule(y_1,k_estim);

        % Implement FPE, AIC, and MDL order selection criteria.
        % Use the variance E, obtained from aryule(), as the residual power
        % to drive each of the order selection criteria.
        % Store the result of each criteria in FPE_val(k_estim), AIC_val(k_estim), or
        % MDL_val(k_estim).
        FPE_val(k_estim) = (N+k_estim)*E/(N-k_estim);
        AIC_val(k_estim) = N*log(E)+2*k_estim;
        MDL_val(k_estim) = N*log(E)+k_estim*log(N);
    end
    
    % For each criteria, pick the value 'k*' which minimizes the criterion
    % and store the value of 'k*' in the corresponding vectors FPE_order(i),
    % AIC_order(i), or MDL_order(i).
    [m,FPE_order(i)] = min(FPE_val);
    [m,AIC_order(i)] = min(AIC_val);
    [m,MDL_order(i)] = min(MDL_val);
    
end

% From vectors FPE_order, AIC_order, MDL_order, obtain the histogram of the
% values of 'k*' and pick the one with the highest frequency as the order
% estimate for the given criteria.
figure;
subplot(3,1,1)
histogram(FPE_order)
title(sprintf( 'FPE_order N= %f NrOfTrials= %f', N, NrOfTrials))

subplot(3,1,2)
histogram(AIC_order)
title(sprintf( 'AIC_order N= %f NrOfTrials= %f', N, NrOfTrials))
subplot(3,1,3)
histogram(MDL_order)
title(sprintf( 'MDL_order N= %f NrOfTrials= %f', N, NrOfTrials))


end

