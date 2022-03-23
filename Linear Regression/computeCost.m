function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost function
m = length(y); % number of training examples
J = (sum((X*theta - y).^2))/(2*m);  




end
