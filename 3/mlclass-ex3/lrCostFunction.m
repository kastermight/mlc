function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
g = sigmoid(X*theta);
J = sum(-y.*log(g) - (1-y).*log(1-g))/m + lambda*theta(2:end)'*theta(2:end)/(2*m);
grad = X'*(g-y)/m + [0; lambda*theta(2:end)/m];
end
