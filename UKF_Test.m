clc; close all; clearvars;

% Unscented Kalman Filter (UKF) for Economic Tracking
% Tracking GDP, Inflation, and Interest Rate

clear; clc;

% Define Initial Conditions
x = [1.0; 0.02; 0.05]; % Initial state: [GDP growth, Inflation, Interest rate]
P = eye(3); % Initial covariance matrix

% Process and measurement noise covariance matrices
Q = diag([0.01, 0.005, 0.002]); % Process noise
R = diag([0.02, 0.01, 0.005]); % Measurement noise

% UKF Parameters
alpha = 1e-3;  % Scaling parameter
kappa = 0;      % Secondary scaling
beta = 2;       % Optimal for Gaussian distributions
lambda = alpha^2 * (3 + kappa) - 3;

% Define Process Model
f = @(x) [ x(1) + 0.1*x(1)*x(2) - 0.05*x(3);  % GDP growth rate update
           x(2) + 0.01*x(1) - 0.02*x(2);      % Inflation update
           x(3) - 0.005*x(1) + 0.03*x(2)];    % Interest rate update

% Define Measurement Model
h = @(x) [x(1); x(2); x(3)]; % Direct measurement of state

% Simulation Parameters
N = 50; % Number of time steps
x_true = zeros(3, N);
x_est = zeros(3, N);
z_meas = zeros(3, N);
x_est(:,1) = x; % Initialize estimation state

for k = 2:N
    % Simulate true state
    x = f(x) + mvnrnd([0;0;0], Q)';
    x_true(:, k) = x;
    
    % Simulate measurement
    z = h(x) + mvnrnd([0;0;0], R)';
    z_meas(:, k) = z;
    
    % UKF Sigma Points Calculation
    n = length(x);
    sqrtP = chol(P, 'lower');
    X_sigma = [x, x + sqrt(n + lambda) * sqrtP, x - sqrt(n + lambda) * sqrtP];
    
    % Propagate Sigma Points
    X_pred = zeros(size(X_sigma));
    for i = 1:size(X_sigma,2)
        X_pred(:,i) = f(X_sigma(:,i));
    end
    
    % Compute Predicted Mean and Covariance
    Wm = [lambda / (n + lambda), repmat(1 / (2 * (n + lambda)), 1, 2 * n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
    x_pred = X_pred * Wm';
    P_pred = Q;
    for i = 1:size(X_pred,2)
        P_pred = P_pred + Wc(i) * (X_pred(:,i) - x_pred) * (X_pred(:,i) - x_pred)';
    end
    
    % Measurement Prediction
    Z_sigma = X_pred;
    z_pred = Z_sigma * Wm';
    P_zz = R;
    for i = 1:size(Z_sigma,2)
        P_zz = P_zz + Wc(i) * (Z_sigma(:,i) - z_pred) * (Z_sigma(:,i) - z_pred)';
    end
    P_xz = zeros(n);
    for i = 1:size(X_pred,2)
        P_xz = P_xz + Wc(i) * (X_pred(:,i) - x_pred) * (Z_sigma(:,i) - z_pred)';
    end
    
    % UKF Update Step
    K = P_xz / P_zz;
    x_est(:, k) = x_pred + K * (z - z_pred);
    P = P_pred - K * P_zz * K';
end

% Plot Results
figure;
subplot(3,1,1);
plot(1:N, x_true(1,:), 'b', 1:N, x_est(1,:), 'r--');
title('GDP Growth Rate Tracking'); legend('True', 'Estimated');

subplot(3,1,2);
plot(1:N, x_true(2,:), 'b', 1:N, x_est(2,:), 'r--');
title('Inflation Rate Tracking'); legend('True', 'Estimated');

subplot(3,1,3);
plot(1:N, x_true(3,:), 'b', 1:N, x_est(3,:), 'r--');
title('Interest Rate Tracking'); legend('True', 'Estimated');
