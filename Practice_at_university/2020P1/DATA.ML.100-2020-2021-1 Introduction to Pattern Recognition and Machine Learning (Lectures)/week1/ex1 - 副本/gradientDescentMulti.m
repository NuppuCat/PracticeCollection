function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp=theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
   % temp1 = theta(1)-alpha/m*((X(:,1)')*(X*theta-y));
   % temp2 = theta(2)-alpha/m*((X(:,2)')*(X*theta-y));
 
   % theta(1)=temp1;
    %theta(2)=temp2;
    
    
     for iter2 =  1 : size(X,2)
    temp(iter2) = theta(iter2)-alpha/m*((X(:,iter2))'*(X*theta-y));

     end
    theta = temp;

    
  

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
