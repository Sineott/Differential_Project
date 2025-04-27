% --- Supporting Functions ---
function x_est_norm = run_ekf(x0, P0, Q, R, z_meas_norm, N)
    % Runs EKF based on given noise levels and initial state
    n = length(x0);
    x_est_norm = zeros(n,N);
    x_est_norm(:,1) = x0;
    P = P0;

    % Define your model dynamics
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

    for k = 2:N
        % Prediction
        x_pred = f(x_est_norm(:,k-1));

        F = eye(n);
        F(1,2) = -0.02; F(1,3) = 0.05;
        F(2,1) = 0.01;  F(2,7) = -0.01;
        F(3,1) = 0.03;  F(3,4) = -0.01;
        F(4,1) = -0.005; F(4,3) = 0.01;
        F(5,1) = 0.02; F(5,2) = -0.01;
        F(6,1) = 0.01; F(6,2) = -0.02; F(6,3) = 0.01;
        F(7,5) = -0.005; F(7,6) = 0.01;

        P_pred = F*P*F' + Q;

        % Update
        H = eye(n);
        K = P_pred*H'/(H*P_pred*H' + R);
        z = z_meas_norm(:,k);

        x_est_norm(:,k) = x_pred + K*(z - h(x_pred));
        P = (eye(n) - K*H)*P_pred;
    end
end
