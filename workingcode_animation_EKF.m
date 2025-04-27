% Extended Kalman Filter (EKF) for Economic Tracking with Comparison Plots
clear; clc;close all;

% Read the CSV and Excel files
try
    % Import data using readtable for CSV files and readmatrix for Excel
    depreciation_data = readtable('Y0000C1Q027SBEA.csv');
    savings_data = readtable('PSAVERT.csv');
    labour_data = readtable('LFACTTTTUSM647S.csv');
    labour_growth_data = readtable('LABOUR_GROWTH_RATE.xlsx');
    output_data = readtable('GDPC1.csv');
    productivity_data = readtable('RTFPNAUSA632NRUG.csv');
    capital_data = readtable('RKNANPUSA666NRUG.csv');
    
    % Extract the last columns (values) - taking only the first 66 rows
    depreciation = depreciation_data{1:66, end};
    savings = savings_data{1:66, end};
    labour = labour_data{1:66, end};
    labour_growth = labour_growth_data{1:66, end};
    output = output_data{1:66, end};
    productivity = productivity_data{1:66, end};
    capital = capital_data{1:66, end};
    
    % Number of time steps
    N = 66;
    
    % Assemble all variables into a combined state vector
    % State vector: [output, labour, capital, depreciation, savings, productivity, labour_growth]
    x_true = [output, labour, capital, depreciation, savings, productivity, labour_growth]';
    
    % Create measurements with some noise
    rng(42); % Set random seed for reproducibility
    noise_level = 0.02; % 2% noise
    z_meas = x_true .* (1 + noise_level * randn(size(x_true)));
    
    disp('Successfully loaded CSV files.');
    
catch
    % If files not found or error reading, create simulated data
    disp('Error reading CSV files. Using simulated data instead.');
    
    % Simulation parameters
    N = 66;
    
    % Initial state values
    initial_state = [100; 50; 200; 0.1; 8; 1; 0.02]; 
    % [Output, Labour, Capital, Depreciation, Savings, Productivity, Labour Growth]
    
    % Create simulated data
    x_true = zeros(7, N);
    x_true(:,1) = initial_state;
    
    % Simple simulation model
    for k = 2:N
        % Simple economic model with some random growth/fluctuation
        growth_factors = [0.02; 0.01; 0.015; 0.001; 0.005; 0.01; 0.002];
        noise_factors = [0.01; 0.005; 0.008; 0.002; 0.01; 0.005; 0.001];
        
        growth_rate = noise_factors .* randn(7,1) + growth_factors;
        x_true(:,k) = x_true(:,k-1) .* (1 + growth_rate);
    end
    
    % Create noisy measurements
    z_meas = x_true .* (1 + 0.02 * randn(size(x_true)));
end

% Normalize data to avoid numerical issues
x_mean = mean(x_true, 2);
x_std = std(x_true, 0, 2);

x_true_norm = zeros(size(x_true));
z_meas_norm = zeros(size(z_meas));

for i = 1:size(x_true, 1)
    x_true_norm(i,:) = (x_true(i,:) - x_mean(i)) / x_std(i);
    z_meas_norm(i,:) = (z_meas(i,:) - x_mean(i)) / x_std(i);
end

% Initialize EKF for all variables
x_est_norm = zeros(size(x_true_norm));
x_est_norm(:,1) = z_meas_norm(:,1); % Initialize with first measurement
P = eye(size(x_true, 1)); % Initial covariance matrix

% Process and measurement noise covariance matrices
Q = diag(0.01 * ones(1, size(x_true, 1))); % Process noise
R = diag(0.02 * ones(1, size(x_true, 1))); % Measurement noise

% Define state transition function for normalized economic variables
% This is a simple model relating the variables
f = @(x) [
    x(1) + 0.05*x(3) - 0.02*x(2);                % Output: influenced by capital and labor
    x(2) + 0.01*x(1) - 0.01*x(7);                % Labour: influenced by output and labour growth
    x(3) + 0.03*x(1) - 0.01*x(4);                % Capital: influenced by output and depreciation
    x(4) + 0.01*x(3) - 0.005*x(1);               % Depreciation: influenced by capital and output
    x(5) + 0.02*x(1) - 0.01*x(2);                % Savings: influenced by output and labour
    x(6) + 0.01*x(1) + 0.01*x(3) - 0.02*x(2);    % Productivity: influenced by output, capital and labour
    x(7) + 0.01*x(6) - 0.005*x(5);               % Labour growth: influenced by productivity and savings
    ];

% Define direct measurement function
h = @(x) x; % Direct measurement of all state variables

% Run the EKF algorithm
for k = 2:N
    % Get current measurement
    z = z_meas_norm(:, k);
    
    % EKF Prediction Step
    x_pred = f(x_est_norm(:, k-1));
    
    % Calculate Jacobian of f (simplified, using constant values)
    % In a real application, this would be calculated based on the current state
    F = eye(size(x_true, 1)); % Initialize to identity
    
    % GDP/Output row
    F(1, 1) = 1;
    F(1, 2) = -0.02;
    F(1, 3) = 0.05;
    
    % Labour row
    F(2, 1) = 0.01;
    F(2, 2) = 1;
    F(2, 7) = -0.01;
    
    % Capital row
    F(3, 1) = 0.03;
    F(3, 3) = 1;
    F(3, 4) = -0.01;
    
    % Depreciation row
    F(4, 1) = -0.005;
    F(4, 3) = 0.01;
    F(4, 4) = 1;
    
    % Savings row
    F(5, 1) = 0.02;
    F(5, 2) = -0.01;
    F(5, 5) = 1;
    
    % Productivity row
    F(6, 1) = 0.01;
    F(6, 2) = -0.02;
    F(6, 3) = 0.01;
    F(6, 6) = 1;
    
    % Labour growth row
    F(7, 5) = -0.005;
    F(7, 6) = 0.01;
    F(7, 7) = 1;
    
    P_pred = F * P * F' + Q;
    
    % EKF Update Step
    H = eye(size(x_true, 1)); % Identity matrix for measurement Jacobian
    K = P_pred * H' / (H * P_pred * H' + R); % Kalman Gain
    x_est_norm(:, k) = x_pred + K * (z - h(x_pred));
    P = (eye(size(x_true, 1)) - K * H) * P_pred; % Updated covariance
end

% Denormalize the estimates
x_est = zeros(size(x_est_norm));
for i = 1:size(x_true, 1)
    x_est(i,:) = x_est_norm(i,:) * x_std(i) + x_mean(i);
end

% Create time vector
time = 1:N;

% Variable names for plots
variable_names = {'GDP/Output', 'Labour', 'Capital', 'Depreciation', 'Savings Rate', 'Productivity', 'Labour Growth Rate'};

% Create individual plots for each variable showing actual vs estimated values
for i = 1:size(x_true, 1)
    figure('Name', variable_names{i}, 'Position', [100*i, 100*i, 800, 400]);
    
    % Plot actual values (static line)
    plot(time, x_true(i,:), 'b-', 'LineWidth', 2);
    hold on;
    
    % Use comet for the Kalman Filter estimate (animated)
    comet(time, x_est(i,:));
    
    title([variable_names{i}, ': Actual vs Estimated']);
    xlabel('Time Step');
    ylabel('Value');
    legend('Actual Value', 'Kalman Filter Estimate');
    grid on;
    
    % Calculate and display RMSE
    rmse = sqrt(mean((x_true(i,:) - x_est(i,:)).^2));
    text(5, min(x_true(i,:)) + 0.9*(max(x_true(i,:)) - min(x_true(i,:))), ...
        ['RMSE: ', num2str(rmse, '%.4f')], 'FontSize', 12);
    
    % Improve formatting for readability
    ax = gca;
    ax.FontSize = 12;
    ax.LineWidth = 1.5;
end

% Calculate overall RMSE for each variable
rmse_values = zeros(size(x_true, 1), 1);
for i = 1:size(x_true, 1)
    rmse_values(i) = sqrt(mean((x_true(i,:) - x_est(i,:)).^2));
end

% Display RMSE values
fprintf('RMSE Values:\n');
for i = 1:size(x_true, 1)
    fprintf('%s: %.4f\n', variable_names{i}, rmse_values(i));
end

% Create a combined figure showing all normalized variables
figure('Name', 'All Variables Normalized', 'Position', [100, 100, 1000, 600]);

% Normalize data for plotting
norm_true = zeros(size(x_true));
norm_est = zeros(size(x_est));

for i = 1:size(x_true, 1)
    norm_true(i,:) = (x_true(i,:) - min(x_true(i,:))) / (max(x_true(i,:)) - min(x_true(i,:)));
    norm_est(i,:) = (x_est(i,:) - min(x_true(i,:))) / (max(x_true(i,:)) - min(x_true(i,:)));
end

% Create 2x4 subplot grid for all variables
for i = 1:size(x_true, 1)
    subplot(2, 4, i);
    
    % Plot actual values (static line)
    plot(time, norm_true(i,:), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Use comet for the Kalman Filter estimate (animated)
    comet(time, norm_est(i,:));
    
    title(variable_names{i});
    if i > 4 % Only add x-labels to bottom row
        xlabel('Time Step');
    end
    ylabel('Normalized Value');
    grid on;
    legend('Actual', 'Estimate');
end

% Adjust layout
set(gcf, 'Color', 'w');
sgtitle('Normalized Economic Variables: Actual vs Estimated', 'FontSize', 14);