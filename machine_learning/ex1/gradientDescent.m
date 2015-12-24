function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
gradient = zeros(length(theta),1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
for iter = 1:num_iters
	gradient = (sum((X*theta -y)'*X, 1))';
	theta = theta - alpha/m*gradient;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

	% check the method is working 
	if(iter>1 && J_history(iter) > J_history(iter-1))
		fprintf('iteration:%d  The gradient decent fails\n', iter);
		return;
	end	
end

end
