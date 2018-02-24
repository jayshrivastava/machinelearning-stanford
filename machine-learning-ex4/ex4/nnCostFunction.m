function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%FORWARD PROP - copied from previous assingmnet 
a1 = [ones(m,1) X]; %m x 401
z2 = a1*Theta1'; %m x 26
a2 = sigmoid(z2); %m x 26
a2 = [ones(m,1) a2]; %m x 27
z3 = a2*Theta2';%m x 10
hyp = sigmoid(z3)'; % 10*m matrix . each col is a result vector. (Ammended from prev. assignment)

%convert y values (1 to 10) into 10*1 vectors. put them into a matrix (10*m)
%Y is a 10 by m matrix
Y = zeros (num_labels,m);
for i = 1:m
  value = y(i); %class from digits 1 to 10
  Y(value,i) = 1; %vectorize
end

a = sum(sum((-Y).*log(hyp)-(1-Y).*log(1-hyp)));
J = a/m;
%TLDR: To sum an m * n matrix A, we sum twice: sum(sum(A));
%NOTE on the sum function: It returns the sum wrt the first dimention of the matrix that is not 1
%Notice that a is summed twice: 
%The first sum() returns a 1*m vector where each column m contains the sum of all its previous rows, 
%then the 2nd sum returns a 1*1 vector (summing all the column values)

%REGULARIZATION
t1 = Theta1 (:,2:size(Theta1,2)); %row: hidden layer size (25), col: input size (401 -> 400)
t2 = Theta2 (:,2:size(Theta2,2)); %Copied from prev. assignment

reg = (lambda/(2*m))*(sum(sum(t1.^2)) + sum(sum(t2.^2)));
J = J + reg;

%BACK PROP

a2t = a2'; %26xm
z2t = z2'; %25xm

for j=1:m
  
  %Follow steps in EX4.pdf
  A1 = a1(j,:);
  A2 = a2t(:,j);
  
  delta3 = hyp(:,j) - Y(:,j); % 10x1
 
  Z2 = z2t(:,j);
  Z2=[1; Z2]; % 26 * 1
  
  delta2 = (Theta2' * delta3) .* sigmoidGradient(Z2);
  
  delta2 = delta2(2:end); 

	Theta2_grad = Theta2_grad + delta3 * A2';
	Theta1_grad = Theta1_grad + delta2 * A1; 

end

Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
