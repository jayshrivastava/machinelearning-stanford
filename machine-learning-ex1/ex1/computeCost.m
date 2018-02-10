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

% Matrix x (m by 2) times theta (2 by 1) returns an m * 1 matrix. \
% Subtract all y values from this result and square y to  
a = X*theta - y;  
J = a'*a/(2*m);
  



% =========================================================================

end
