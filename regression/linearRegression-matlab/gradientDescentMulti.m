% Regression task - gradient descent of linear regression
%
% @HYPJUDY 2017.6.21
% Details: https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/

function [theta, J_history] = ...
    gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates
%   theta by taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    newDecrement = alpha / m * (X * theta - y)' * X; 
    theta = theta - newDecrement';
    
    % Save the cost J in every iteration   
    % Compute cost for linear regression with multiple variables
    % J_history(iter) = 1 / (2 * m) * sum((X * theta - y).^2);
    J_history(iter) = sqrt(sum((X * theta - y).^2) / m); % RMSE
    fprintf('RMSE: %f \n', J_history(iter));
end

end
