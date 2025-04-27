% Unscented Kalman Filter (UKF) for Economic Tracking with Comet Plot Animation
% Modified to save animations using VideoWriter
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

% Initialize UKF for all variables
n = size(x_true, 1); % State dimension
x_est_norm = zeros(size(x_true_norm));
x_est_norm(:,1) = z_meas_norm(:,1); % Initialize with first measurement
P = eye(n); % Initial covariance matrix

% Process and measurement noise covariance matrices
Q = diag(0.01 * ones(1, n)); % Process noise
R = diag(0.02 * ones(1, n)); % Measurement noise

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

% UKF parameters
alpha = 1e-3;   % Determines spread of sigma points around mean
kappa = 0;      % Secondary scaling parameter
beta = 2;       % Incorporates prior knowledge of distribution (2 is optimal for Gaussian)
lambda = alpha^2 * (n + kappa) - n;
gamma = sqrt(n + lambda);

% Calculate weights for mean and covariance
Wm = zeros(2*n + 1, 1); % Weights for mean
Wc = zeros(2*n + 1, 1); % Weights for covariance

Wm(1) = lambda / (n + lambda);
Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);

for i = 2:(2*n + 1)
    Wm(i) = 1 / (2 * (n + lambda));
    Wc(i) = 1 / (2 * (n + lambda));
end

% Run the UKF algorithm
for k = 2:N
    % Get current measurement
    z = z_meas_norm(:, k);
    
    % Get previous state estimate and covariance
    x = x_est_norm(:, k-1);
    
    % Generate sigma points
    sigma_points = zeros(n, 2*n + 1);
    
    % Mean as the first sigma point
    sigma_points(:, 1) = x;
    
    % Calculate square root of P (using Cholesky decomposition)
    A = chol(P)';
    
    % Generate remaining sigma points
    for i = 1:n
        sigma_points(:, i+1) = x + gamma * A(:, i);
        sigma_points(:, i+1+n) = x - gamma * A(:, i);
    end
    
    % Propagate sigma points through state transition function
    sigma_points_pred = zeros(n, 2*n + 1);
    for i = 1:(2*n + 1)
        sigma_points_pred(:, i) = f(sigma_points(:, i));
    end
    
    % Calculate predicted mean
    x_pred = zeros(n, 1);
    for i = 1:(2*n + 1)
        x_pred = x_pred + Wm(i) * sigma_points_pred(:, i);
    end
    
    % Calculate predicted covariance
    P_pred = zeros(n, n);
    for i = 1:(2*n + 1)
        diff = sigma_points_pred(:, i) - x_pred;
        P_pred = P_pred + Wc(i) * (diff * diff');
    end
    P_pred = P_pred + Q;
    
    % Propagate sigma points through measurement function
    sigma_z = zeros(n, 2*n + 1);
    for i = 1:(2*n + 1)
        sigma_z(:, i) = h(sigma_points_pred(:, i));
    end
    
    % Calculate predicted measurement mean
    z_pred = zeros(n, 1);
    for i = 1:(2*n + 1)
        z_pred = z_pred + Wm(i) * sigma_z(:, i);
    end
    
    % Calculate innovation covariance
    S = zeros(n, n);
    for i = 1:(2*n + 1)
        diff = sigma_z(:, i) - z_pred;
        S = S + Wc(i) * (diff * diff');
    end
    S = S + R;
    
    % Calculate cross-correlation matrix
    P_xz = zeros(n, n);
    for i = 1:(2*n + 1)
        diff_x = sigma_points_pred(:, i) - x_pred;
        diff_z = sigma_z(:, i) - z_pred;
        P_xz = P_xz + Wc(i) * (diff_x * diff_z');
    end
    
    % Calculate Kalman gain
    K = P_xz / S;
    
    % Update state estimate and covariance
    x_est_norm(:, k) = x_pred + K * (z - z_pred);
    P = P_pred - K * S * K';
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

% Create individual plots with animation and save as videos
for i = 1:size(x_true, 1)
    % Create figure
    fig = figure('Name', variable_names{i}, 'Position', [100*i, 100*i, 800, 400]);
    
    % Set up the plot and axes
    hold on;
    plot(time, x_true(i,:), 'b-', 'LineWidth', 2);
    scatter(time, z_meas(i,:), 15, 'g', 'filled', 'MarkerFaceAlpha', 0.3);
    
    title([variable_names{i}, ': Actual vs Estimated']);
    xlabel('Time Step');
    ylabel('Value');
    grid on;
    
    % Calculate and display RMSE
    rmse = sqrt(mean((x_true(i,:) - x_est(i,:)).^2));
    text(5, min(x_true(i,:)) + 0.9*(max(x_true(i,:)) - min(x_true(i,:))), ...
        ['RMSE: ', num2str(rmse, '%.4f')], 'FontSize', 12);
    
    % Setup VideoWriter
    video_filename = [variable_names{i}, '_animation.mp4'];
    video_filename = strrep(video_filename, '/', '_'); % Replace any slashes in filenames
    vw = VideoWriter(video_filename, 'MPEG-4');
    vw.FrameRate = 15; % Frames per second
    vw.Quality = 95;   % Video quality (0-100)
    open(vw);
    
    % Create the animation and record it frame by frame
    h = animatedline('Color', 'r', 'LineWidth', 2);
    
    for k = 1:N
        addpoints(h, time(k), x_est(i,k));
        drawnow
        
        % Capture the frame
        frame = getframe(fig);
        writeVideo(vw, frame);
        
        % Pause briefly to slow down animation as needed
        pause(0.05);
    end
    
    % Close the video writer
    close(vw);
    fprintf('Saved video for %s to %s\n', variable_names{i}, video_filename);
    
    % Add legend after animation completes
    legend('Actual Value', 'Measurements', 'Kalman Filter Estimate', 'Location', 'best');
    
    % Improve formatting for readability
    ax = gca;
    ax.FontSize = 12;
    ax.LineWidth = 1.5;
end

% Create a combined figure with normalized data and animation
fig_combined = figure('Name', 'All Variables Normalized', 'Position', [100, 100, 1000, 600]);

% Normalize data for plotting
norm_true = zeros(size(x_true));
norm_est = zeros(size(x_est));
norm_meas = zeros(size(z_meas));

for i = 1:size(x_true, 1)
    norm_true(i,:) = (x_true(i,:) - min(x_true(i,:))) / (max(x_true(i,:)) - min(x_true(i,:)));
    norm_est(i,:) = (x_est(i,:) - min(x_true(i,:))) / (max(x_true(i,:)) - min(x_true(i,:)));
    norm_meas(i,:) = (z_meas(i,:) - min(x_true(i,:))) / (max(x_true(i,:)) - min(x_true(i,:)));
end

% Setup VideoWriter for combined animation
combined_video = VideoWriter('Combined_Economic_Variables_Animation.mp4', 'MPEG-4');
combined_video.FrameRate = 15;
combined_video.Quality = 95;
open(combined_video);

% Initialize animated lines
h_lines = cell(size(x_true, 1), 1);

% Create 2x4 subplot grid for all variables
for i = 1:size(x_true, 1)
    subplot(2, 4, i);
    
    % Plot true values first
    plot(time, norm_true(i,:), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Scatter plot for measurements
    scatter(time, norm_meas(i,:), 10, 'g', 'filled', 'MarkerFaceAlpha', 0.3);
    
    title(variable_names{i});
    if i > 4 % Only add x-labels to bottom row
        xlabel('Time Step');
    end
    ylabel('Normalized Value');
    grid on;
    
    % Create animated line for each subplot
    h_lines{i} = animatedline('Color', 'r', 'LineWidth', 1.5);
    
    % Only add legend to the first subplot
    if i == 1
        legend('Actual', 'Measurements', 'Estimate', 'Location', 'best');
    end
end

% Adjust layout
sgtitle('Normalized Economic Variables: Actual vs Estimated', 'FontSize', 14);

% Run animation for all subplots frame by frame
for k = 1:N
    for i = 1:size(x_true, 1)
        subplot(2, 4, i);
        addpoints(h_lines{i}, time(k), norm_est(i,k));
    end
    
    drawnow
    
    % Capture the frame
    frame = getframe(fig_combined);
    writeVideo(combined_video, frame);
    
    % Pause briefly to slow down animation
    pause(0.05);
end

% Close the video writer
close(combined_video);
fprintf('Saved combined animation to Combined_Economic_Variables_Animation.mp4\n');

% Display a summary of what was saved
fprintf('\nVideo files saved:\n');
for i = 1:size(x_true, 1)
    video_filename = [variable_names{i}, '_animation.mp4'];
    video_filename = strrep(video_filename, '/', '_');
    fprintf('- %s\n', video_filename);
end
fprintf('- Combined_Economic_Variables_Animation.mp4\n');