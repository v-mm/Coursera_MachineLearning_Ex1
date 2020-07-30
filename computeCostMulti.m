function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% COMPUTECOSTMULTI implementation is identical to COMPUTECOST i.e.
% for both univariate and multivariate linear regression this code is generic

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% assumptions X is (m x n+1), theta is (n+1 x 1) and y is (mx1)
% n+1 columns since first column of 1s is added to feature matrix of n features
% so total columns is n+1
 

H = X * theta; % hypothesis vector is X*theta instead of theta_transpose*X
               % considering the dimensions of X as (m x n+1) and theta as (n+1 x 1)
               % H has dimensions (mx1) i.e. a column vector like y
err = (H - y); % computing the error between the hypothesis  and y, 
               % note err is also a vector like H and y
               
err_sqr = err.^2; % squaring the difference of the elements in the two vectors
                  % err_sqr is also a vector

sum_err_sqr = sum(err_sqr); % since H is a m by 1 vector
                            % note sum_err_sqr is a scalar

J = sum_err_sqr / (2 * m); % Cost is (1/2m)*sum_err_sqr




% =========================================================================

end
