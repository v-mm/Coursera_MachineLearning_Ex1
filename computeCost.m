function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% assumptions X is m by 2, y is m by 1 and theta is 2 by 1

H = X * theta; % hypothesis vector is X*theta instead of theta_transpose*X
               % considering the dimensions of X as m by 2 and theta as 2 by 1
               % H has dimensions m by 1 i.e. a column vector like y
err = (H - y); % computing the error between the hypothesis and y, 
               % note err is also a vector like H and y
               
err_sqr = err.^2; % squaring the difference of the elements in the two vectors
                  % err_sqr is also a vector

sum_err_sqr = sum(err_sqr); % since H is a m by 1 vector
                            % note sum_err_sqr is a scalar

J = sum_err_sqr / (2 * m); % Cost is (1/2m)*sum_err_sqr

% =========================================================================

end
