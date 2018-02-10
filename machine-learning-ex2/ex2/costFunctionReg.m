function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


hyp = sigmoid(X*theta);

a = -y'*log(hyp);
b = (1.-y)'*log(1-hyp);

theta2 = theta (2:size(theta),:); 
J = (a - b)/m + lambda*(theta2'*theta2)/(2*m);

theta3 = [0; theta2];
grad = (X'*(hyp-y))./m + lambda*theta3./m;



% =============================================================

end
