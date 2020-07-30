function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% GRADIENTDESCENTMULTI implementation is identical to GRADIENTDESCENT i.e.
% for both univariate and multivariate linear regression this code is generic


% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % assumptions X is (m x n+1), theta is (n+1 x 1) and y is (mx1)
    % n+1 columns since first column of 1s is added to X matrix of n features
    
    
    % grad descent 
    % repeat until convergence 
    % theta_j = theta_j - alpha x 1/m x Sum i(1 to m) [(hypothesis_theta - y)] x X i  
    
    % hypothesis_theta
    % X is (m x n+1), theta is (n+1 x 1)
    H_theta = X * theta; % H_thetais (mx1)
    
    % say Delta is Sum i(1 to m) of [(hypothesis_theta - y)] x Xi 
    % take X' otherwise vector multiplication is not possibe
    % since X is (m x n+1) and [H_theta - y] is (m x 1), X' is (n+1 x m)
    % the vector multiplication involves sum of products hence no need to
    % use the sum function separately
    Delta = X' * [H_theta - y]; % result Delta is a (n+1 x 1) vector, like theta

    % simultaneously update all theta values by updating theta array elements
    theta = theta - alpha * 1/m * Delta;
 
    % Save the new cost J for new theta from every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

    % print out updated theta values and computed cost for upda theta
    % for every iteration  
    fprintf('Iteration number is: %d\n', iter);
    fprintf('Updated theta are: %f\n', theta);
    fprintf('Computed cost J_theta for updated theta is: %f\n', J_history(iter));



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
