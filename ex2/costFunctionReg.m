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
h = sigmoid(X*theta); %the hypothesis
first = sum(-(y'*log(h)));   %first sum term
second = sum((1-y)'*log(1-h)); %second sum term
unreg = (1/m).*(first-second);  %unregularized cost
theta(1) = 0;   %exclude the bias feature
ss = theta'*theta; %sum of squares of theta
reg1 = (lambda/(2*m))*ss; %regularization term
J = unreg + reg1 ;    %cost function

error = h-y;
s = error'*X;
theta(1) = 0;
reg2 = (lambda/m).*theta;

grad = (1/m).*s + reg2';    %gradient
    
    
% =============================================================

end
