function [rmse_values, x_true, x_est] = UKF_economic_model(Q_diag, R_diag, alpha, beta, kappa)
    % Implementation of the Unscented Kalman Filter for economic variables
    % Load or generate data (same as original code)
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
    z_meas_norm = zeros(size(z_meas)); % Fixed: changed from z to z_meas
    
    for i = 1:size(x_true, 1)
        x_true_norm(i,:) = (x_true(i,:) - x_mean(i)) / x_std(i);
        z_meas_norm(i,:) = (z_meas(i,:) - x_mean(i)) / x_std(i);
    end
    
    % State and measurement dimensions
    n = size(x_true, 1); % State dimension
    m = size(z_meas, 1); % Measurement dimension
    
    % Set up system matrices
    % System model: x(k+1) = f(x(k)) + w(k)
    % Measurement model: z(k) = h(x(k)) + v(k)
    
    % Process and measurement noise covariances
    Q = diag(ones(n,1) * Q_diag); % Process noise covariance
    R = diag(ones(m,1) * R_diag); % Measurement noise covariance
    
    % Initial state estimate and covariance
    x_hat = zeros(n, 1); % Initial state estimate (normalized)
    P = eye(n); % Initial state covariance
    
    % UKF parameters
    lambda = alpha^2 * (n + kappa) - n;
    
    % Weights for sigma points
    Wm = zeros(2*n+1, 1); % Mean weights
    Wc = zeros(2*n+1, 1); % Covariance weights
    
    Wm(1) = lambda / (n + lambda);
    Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);
    
    for i = 2:2*n+1
        Wm(i) = 1 / (2*(n + lambda));
        Wc(i) = 1 / (2*(n + lambda));
    end
    
    % Storage for estimates
    x_est_norm = zeros(n, N);
    x_est = zeros(n, N);
    
    % UKF loop
    for k = 1:N
        % 1. Generate sigma points
        sigma_points = zeros(n, 2*n+1);
        
        % Mean as first sigma point
        sigma_points(:,1) = x_hat;
        
        % Calculate matrix square root
        S = chol((n + lambda) * P, 'lower');
        
        for i = 1:n
            sigma_points(:,i+1) = x_hat + S(:,i);
            sigma_points(:,i+1+n) = x_hat - S(:,i);
        end
        
        % 2. Predict step
        % a. Transform sigma points through process model
        sigma_points_pred = sigma_points; % Simple model for demonstration
        
        % b. Calculate predicted mean and covariance
        x_hat_pred = zeros(n, 1);
        for i = 1:2*n+1
            x_hat_pred = x_hat_pred + Wm(i) * sigma_points_pred(:,i);
        end
        
        P_pred = Q; % Initialize with process noise
        for i = 1:2*n+1
            diff = sigma_points_pred(:,i) - x_hat_pred;
            P_pred = P_pred + Wc(i) * (diff * diff');
        end
        
        % 3. Update step
        % a. Calculate predicted measurements from sigma points
        z_pred_sigma = sigma_points_pred; % Identity measurement model for simplicity
        
        % b. Calculate predicted measurement and innovation covariance
        z_hat_pred = zeros(m, 1);
        for i = 1:2*n+1
            z_hat_pred = z_hat_pred + Wm(i) * z_pred_sigma(:,i);
        end
        
        Pzz = R; % Initialize with measurement noise
        for i = 1:2*n+1
            diff = z_pred_sigma(:,i) - z_hat_pred;
            Pzz = Pzz + Wc(i) * (diff * diff');
        end
        
        % c. Calculate cross covariance
        Pxz = zeros(n, m);
        for i = 1:2*n+1
            diff_x = sigma_points_pred(:,i) - x_hat_pred;
            diff_z = z_pred_sigma(:,i) - z_hat_pred;
            Pxz = Pxz + Wc(i) * (diff_x * diff_z');
        end
        
        % d. Calculate Kalman gain and update state estimate
        K = Pxz / Pzz;
        
        x_hat = x_hat_pred + K * (z_meas_norm(:,k) - z_hat_pred);
        P = P_pred - K * Pzz * K';
        
        % Store the estimate (normalized)
        x_est_norm(:,k) = x_hat;
        
        % Denormalize for output
        for i = 1:n
            x_est(i,k) = x_hat(i) * x_std(i) + x_mean(i);
        end
    end
    
    % Calculate RMSE for each variable
    rmse_values = zeros(n, 1);
    for i = 1:n
        rmse_values(i) = sqrt(mean((x_true(i,:) - x_est(i,:)).^2));
    end
end