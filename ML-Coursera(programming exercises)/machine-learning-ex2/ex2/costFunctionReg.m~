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

%compute cost
temp=X*theta;
temp=sigmoid(temp);
J=(y')*log(temp);
J=J+((1-y)')*log(1-temp);
J=J/m;
J=J*(-1);

temp=theta(2:size(theta,1));
temp=(temp')*(temp);
temp=temp*lambda;
temp=temp/m;
temp=temp/2;

J=J+temp;
%

% compute gradient
temp=X*theta;
temp=sigmoid(temp);
temp=temp-y;
temp=(X')*temp;
grad=temp/m;

temp=theta*lambda;
temp=temp/m;
temp=[0;temp(2:size(temp,1))];

grad=grad+temp;
%






% =============================================================

end
