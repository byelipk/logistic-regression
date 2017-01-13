function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Code to compute J
h = sigmoid(X * theta);
term1 = -y' * log(h);
term2 = (1 - y') * log(1 - h);
unregularized_cost = (1 / m) * (term1 - term2);

theta(1) = 0; % Regularization peanalizes theta from being too large.
lambda = 0;   % Default value for lambda
sum_of_sqaures = theta' * theta;
regularized_cost = (lambda / (2 * m)) * sum_of_sqaures;

J = (unregularized_cost + regularized_cost);

% Code to compute the gradient
alpha = 0.01;
diff  = h - y;
diffs = repmat(diff, 1, size(X, 2)); % Build an (m x n) matrix
delta = (1 / m) * sum(X.* diffs);    % Matrix multiplication on each X(i, j)

% NOTE
% Notice we're not computing reduced values for theta. That will be the job
% of `fminunc`.

grad = delta';





% =============================================================

end
