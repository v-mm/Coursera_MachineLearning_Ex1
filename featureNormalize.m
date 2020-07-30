function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);  % creates a row matrix with mean for every column 
sigma = std(X);% creates a row matrix with standard deviation for every column

% multiply 1s column matrix i.e. 1s(mx1) with row matrix of mu (1xn) to create a
% 2d matrix (mxn) of mu repeated for m rows. Similarly create sigma_matrix.
mu_matrix = ones(size(X,1),1) * mu;  
sigma_matrix = ones(size(X,1),1) * sigma;

% X, mu_matrix and sigma_matrix are same dimension mxn
% subract X - mu_matrix and elementwise divide with sigma_matrix
X_norm = (X-mu_matrix)./sigma_matrix;

% ============================================================

end
