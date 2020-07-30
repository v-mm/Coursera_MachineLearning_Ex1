function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % ============================================================

    % grad descent 
    % repeat until convergence 
    % theta_j = theta_j - alpha x 1/m x Sum i(1 to m) [(hypothesis_theta - y)] x X i  
    
    % hypothesis_theta
    H_theta = X * theta;
    
    % say Delta is Sum i(1 to m) of [(hypothesis_theta - y)] x Xi 
    % take X' otherwise vector multiplication is not possibe
    % since X is m by 2 and [H_theta - y] is m by 1
    % the vector multiplication involves sum of products hence no need to
    % use the sum function separately
    Delta = X' * [H_theta - y]; % result Delta is a 2 by 1 dimension vector

    % simultaneously update all theta values by updating theta array elements
    theta = theta - alpha * 1/m * Delta;
 
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % print out updated theta values and computed cost for upda theta
    % for every iteration  
    fprintf('Iteration number is: %d\n', iter);
    fprintf('Updated theta are: %f\n', theta);
    fprintf('Computed cost J_theta for updated theta is: %f\n', J_history(iter));
    
end

end
