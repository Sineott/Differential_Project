
function x_est_norm = run_ukf(x0, P0, Q, R, z_meas_norm, N)
    % Runs UKF based on given noise levels and initial state
    n = length(x0);
    x_est_norm = zeros(n,N);
    x_est_norm(:,1) = x0;
    P = P0;

    f = @(x) [
        x(1) + 0.05*x(3) - 0.02*x(2);
        x(2) + 0.01*x(1) - 0.01*x(7);
        x(3) + 0.03*x(1) - 0.01*x(4);
        x(4) + 0.01*x(3) - 0.005*x(1);
        x(5) + 0.02*x(1) - 0.01*x(2);
        x(6) + 0.01*x(1) + 0.01*x(3) - 0.02*x(2);
        x(7) + 0.01*x(6) - 0.005*x(5);
    ];
    h = @(x) x;

    alpha = 1e-3; kappa = 0; beta = 2;
    lambda = alpha^2*(n+kappa) - n;
    gamma = sqrt(n + lambda);

    Wm = [lambda/(n+lambda); repmat(1/(2*(n+lambda)), 2*n, 1)];
    Wc = Wm; Wc(1) = Wc(1) + (1-alpha^2+beta);

    for k = 2:N
        x = x_est_norm(:,k-1);

        % Sigma Points
        A = chol(P,'lower');
        sigma_points = [x, x + gamma*A, x - gamma*A];

        % Predict sigma points
        sigma_pred = zeros(n,2*n+1);
        for i = 1:2*n+1
            sigma_pred(:,i) = f(sigma_points(:,i));
        end

        % Predict mean and covariance
        x_pred = sigma_pred * Wm;
        P_pred = Q;
        for i = 1:2*n+1
            diff = sigma_pred(:,i) - x_pred;
            P_pred = P_pred + Wc(i)*(diff*diff');
        end

        % Predict measurement
        sigma_z = sigma_pred;
        z_pred = sigma_z * Wm;
        S = R;
        for i = 1:2*n+1
            diff = sigma_z(:,i) - z_pred;
            S = S + Wc(i)*(diff*diff');
        end

        % Cross covariance
        P_xz = zeros(n);
        for i = 1:2*n+1
            diff_x = sigma_pred(:,i) - x_pred;
            diff_z = sigma_z(:,i) - z_pred;
            P_xz = P_xz + Wc(i)*(diff_x*diff_z');
        end

        % Kalman gain
        K = P_xz/S;

        % Update
        z = z_meas_norm(:,k);
        x_est_norm(:,k) = x_pred + K*(z - z_pred);
        P = P_pred - K*S*K';
    end
end
