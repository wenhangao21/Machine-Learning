function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

pred = sigmoid(X*theta);
J = (1 / m) * ( -y'*log(pred) - (1-y)'*log(1-pred)) + lambda/2/m*theta(2:end)'*theta(2:end);;
grad = (1/m) * (X'*(pred - y)) + lambda*theta/m;
grad(1) -= lambda*theta(1)/m;




end
