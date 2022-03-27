function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
pred = sigmoid(X*theta); % pred is a size(X,1) by 1 vector

J = (-y' * log(pred) - (1-y)' * log(1-pred))/m + lambda/2/m*theta(2:end)'*theta(2:end);
temp = theta;
temp(1) = 0;

grad = X'*(pred - y)/m + lambda*temp/m;













% =============================================================

grad = grad(:);

end
