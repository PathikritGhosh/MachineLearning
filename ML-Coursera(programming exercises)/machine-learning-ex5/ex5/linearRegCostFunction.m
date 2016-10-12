function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

	temp=X*theta;
	temp=temp-y;
	temp=temp.^2;
	temp=sum(sum(temp));
	temp=temp/(2*m);
	J=temp;
	
	temp=theta.^2;
	t=sum(sum(temp(2:end,1)));
	t=t*lambda;
	t=t/(2*m);
	
	J=J+t;
	
	temp=X*theta;
	temp=temp-y;
	temp=(X')*temp;
	temp=temp/m;
	grad=temp;
	temp=theta;
	temp=temp*lambda;
	temp=temp/m;
	grad(2:end,1)=grad(2:end,1)+temp(2:end,1);
	





% =========================================================================

grad = grad(:);

end
