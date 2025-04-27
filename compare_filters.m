%% Function Definitions

function [rmse_ekf, rmse_ukf] = compare_filters(P0_scale, Q_scale, R_scale, x0_noise, f, h)
    % This function runs both EKF and UKF on the economic data and returns RMSE values
    
    % Load or simulate data (using the same data loading code from the provided scripts)
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
        
    catch
        % If files not found or error reading, create simulated data
        fprintf('Error reading CSV files. Using simulated data instead.\n');
        
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
    
    % Set up parameters according to sensitivity scale
    n = size(x_true, 1);
    
    % Apply scaled parameters
    P = P0_scale * eye(n);
    Q = Q_scale * diag(0.01 * ones(1, n));
    R = R_scale * diag(0.02 * ones(1, n));
    
    % Add noise to initial state estimate
    initial_x0 = z_meas_norm(:,1) + x0_noise * randn(n, 1);
    
    % ===== Run EKF =====
    x_est_norm_ekf = zeros(size(x_true_norm));
    x_est_norm_ekf(:,1) = initial_x0; % Initialize with noisy first measurement
    P_ekf = P; % Initial covariance matrix
    
    % Jacobian calculation function for EKF
    F_jacobian = eye(n);
    F_jacobian(1, 1) = 1; F_jacobian(1, 2) = -0.02; F_jacobian(1, 3) = 0.05;
    F_jacobian(2, 1) = 0.01; F_jacobian(2, 2) = 1; F_jacobian(2, 7) = -0.01;
    F_jacobian(3, 1) = 0.03; F_jacobian(3, 3) = 1; F_jacobian(3, 4) = -0.01;
    F_jacobian(4, 1) = -0.005; F_jacobian(4, 3) = 0.01; F_jacobian(4, 4) = 1;
    F_jacobian(5, 1) = 0.02; F_jacobian(5, 2) = -0.01; F_jacobian(5, 5) = 1;
    F_jacobian(6, 1) = 0.01; F_jacobian(6, 2) = -0.02; F_jacobian(6, 3) = 0.01; F_jacobian(6, 6) = 1;
    F_jacobian(7, 5) = -0.005; F_jacobian(7, 6) = 0.01; F_jacobian(7, 7) = 1;
    
    H = eye(n); % Direct measurement matrix
    
    % Run EKF
    for k = 2:N
        % EKF Prediction Step
        x_pred = f(x_est_norm_ekf(:, k-1));
        P_pred = F_jacobian * P_ekf * F_jacobian' + Q;
        
        % EKF Update Step
        K = P_pred * H' / (H * P_pred * H' + R); % Kalman Gain
        x_est_norm_ekf(:, k) = x_pred + K * (z_meas_norm(:, k) - h(x_pred));
        P_ekf = (eye(n) - K * H) * P_pred; % Updated covariance
    end
    
    % ===== Run UKF =====
    x_est_norm_ukf = zeros(size(x_true_norm));
    x_est_norm_ukf(:,1) = initial_x0; % Initialize with noisy first measurement
    P_ukf = P; % Initial covariance matrix
    
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
    
    % Run UKF
    for k = 2:N
        % Get previous state estimate and covariance
        x = x_est_norm_ukf(:, k-1);
        
        % Generate sigma points
        sigma_points = zeros(n, 2*n + 1);
        sigma_points(:, 1) = x;
        
        % Calculate square root of P (using Cholesky decomposition)
        try
            A = chol(P_ukf)';
        catch
            % If P is not positive definite, make it so
            [V, D] = eig(P_ukf);
            D = max(D, 1e-6 * eye(size(D)));
            P_ukf = V * D * V';
            A = chol(P_ukf)';
        end
        
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
        x_est_norm_ukf(:, k) = x_pred + K * (z_meas_norm(:, k) - z_pred);
        P_ukf = P_pred - K * S * K';
    end
    
    % Calculate overall RMSE for EKF and UKF
    rmse_ekf = sqrt(mean(mean((x_true_norm - x_est_norm_ekf).^2)));
    rmse_ukf = sqrt(mean(mean((x_true_norm - x_est_norm_ukf).^2)));
end