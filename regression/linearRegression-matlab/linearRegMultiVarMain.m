% Regression task - 
% Main process for linear regression with multiple variables
%
% @HYPJUDY 2017.6.21
% Details: https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/

%% Clear and Close Figures
clear; close all; clc

%% Choose Method
method = 2;
% 1. Gradient descent - test with train dataset
% 2. Normal Equation - test with train dataset
% 3. Gradient Descent - Cross validation
% 4. Normal Equation - Cross validation

%% Load Data
fprintf('Loading data ...\n');
train_data = csvread('../data/save_train.csv', 1, 1);
X_train = train_data(:, 1:384);
y_train = train_data(:, 385);
m = length(y_train);

% Add intercept term to X
X_train = [ones(m, 1) X_train];

%% 1. Gradient descent - test with train dataset
if method == 1
% ----------- Output -------------
% Elapsed time: 9.203286 seconds
% RMSE on train dataset: 8.562877
% --------------------------------
% Init
alpha = 0.1 - 0.001;
num_iters = 500; %50000;
theta = ones(385, 1);
% Run
tic;
[theta, J_history] = ...
    gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);
toc;
% Plot
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2); % blue
xlabel('Number of iterations');
ylabel('Cost J');

%% 2. Normal Equation - test with train dataset
elseif method == 2
% ----------- Output -------------
% Elapsed time: 0.368472 seconds
% Normal equation RMSE on train dataset: 8.170871
% --------------------------------
tic
theta = pinv(X_train' * X_train) * X_train' * y_train;
toc
fprintf('Normal equation RMSE on train dataset: %f\n', ...
    sqrt(sum((X_train * theta - y_train).^2) / m));

%% 3. Gradient Descent - Cross validation
elseif method == 3
% ----------- Output -------------
% Elapsed time: 19.245382 seconds
% avrg_rmse: 10.150006
% --------------------------------
alpha = 0.1 - 0.001;
num_iters = 500;
tic
k = 3;
cv = cvpartition(y_train, 'Kfold', k);
mse = zeros(k, 1);
for k = 1 : k
    trainIdx = cv.training(k);
    testIdx = cv.test(k);
    theta = zeros(385, 1);
    [theta, J_history] = gradientDescentMulti(X_train(trainIdx, :), ...
        y_train(trainIdx), theta, alpha, num_iters);
     
    y_hat = X_train(testIdx, :) * theta;
    
    mse(k) = mean((y_train(testIdx) - y_hat).^2);
    fprintf('Fold %d mse: %f \n', k, mse(k));
end
toc
avrg_rmse = mean(sqrt(mse));
fprintf('avrg_rmse: %f \n', avrg_rmse);

%% 4. Normal Equation - Cross validation
else
% ----------- Output -------------
% Elapsed time: 1.136081 seconds
% avrg_rmse: 8.325153
% --------------------------------
tic
k = 3;
cv = cvpartition(y_train, 'Kfold', k);
mse = zeros(k, 1);
for k = 1 : k
    trainIdx = cv.training(k);
    testIdx = cv.test(k);
    theta = pinv(X_train(trainIdx, :)' * X_train(trainIdx, :)) ...
        * X_train(trainIdx, :)' * y_train(trainIdx);
    y_hat = X_train(testIdx, :) * theta;
    
    mse(k) = mean((y_train(testIdx) - y_hat).^2);
    fprintf('Fold %d mse: %f \n', k, mse(k));
end
toc
avrg_rmse = mean(sqrt(mse));
fprintf('avrg_rmse: %f \n', avrg_rmse);
end
%% Test and write to file
fprintf('Testing ...\n');
X_test = csvread('../data/save_test.csv', 1, 1);
X_test = [ones(m, 1) X_test];
predict_data = X_test * theta;
fprintf('Writing to file ...\n');
headers = {'id','reference'};
data = table((0:24999)', predict_data, 'VariableNames', headers);
writetable(data, 'result.csv')
