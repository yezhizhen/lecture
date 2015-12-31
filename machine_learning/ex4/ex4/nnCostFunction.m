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

X = [ones(m,1), X];
%%p1
%%layer 1 becomes 5000  x (25+1)
z2 = X * Theta1';
layer2 = sigmoid(z2);
layer2 = [ones(m,1), layer2];

%layer 2 becomes 5000 x 10. 
z3 = layer2 * Theta2';
layer3 = sigmoid(z3);

%J +=  - tempy * log(layer2(i,j))  -  (1-tempy) * log(1- layer2(i,j));
%y is 5000 X 1.
%WANT TO MAKE IT 5000 x 10.
%running time 0.036814
%tic();

tempy = zeros(m,num_labels);

for i = 1:m,
	tempy(i, y(i)) = 1;
end;
%now becomes 5000X10
J = 1/m * sum(sum(   - tempy .* log(layer3) - (1-tempy) .* log(1-layer3)   ));


%runtime1 = toc()

%J = 0;
%row major running time 1.5827
%column major running time 1.7984

%if regardless of accessing of y
%row major access 1.55
%column major access 1.43
#{
%tic();
for i = 1:m,
for j=1:num_labels,
		if(y(i) == j)
			tempy = 1;
		else
			tempy = 0; 
		end;
	J +=  - tempy * log(layer3(i,j))  -  (1-tempy) * log(1- layer3(i,j));

end;
end;
J = 1/m * J;
%runtime2 = toc()
#}


regularterm = lambda/(2*m) * (  sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2))  );

J += regularterm;


%tempy = zeros(m,num_labels);
%for i = 1:m,
%	tempy(i, y(i)) = 1;
%end;

%compute gradient
%for example = 1:m,
delta3 = layer3 - tempy; 

%sigmoidGradient
%automatic broadcasting.
delta2 = (delta3 * Theta2)(:,2:end) .*sigmoidGradient (z2);
%layer1 = layer1(:,2:end);
%here, we are using the column of first matrix multiply row of second.
%By property, this is indeed pure matrix multiplication A*B.
%loop time: 0.4512 
%matrix time: 0.0165 

%tic();

Theta2_grad = delta3' * layer2;
Theta1_grad = delta2' * X;


#{
for i = 1 : m,
	Theta2_grad += delta3(i,:)' * layer2(i,:); 
	Theta1_grad += delta2(i,:)' * X(i,:);
end;
#}

%add regularized term for Theta(:,2:end)

Theta2_grad(:,2:end) += lambda * Theta2(:,2:end);
Theta1_grad(:,2:end) += lambda * Theta1(:,2:end);

%runtime3 = toc()



Theta2_grad /= m;
Theta1_grad /= m;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
