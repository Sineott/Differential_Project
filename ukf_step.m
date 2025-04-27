function [x_new, P_new] = ukf_step(x, P, y, Q, R, H, dt)
    L = numel(x);
    alpha = 1e-3;  kappa = 0;  beta = 2;
    lambda = alpha^2 * (L + kappa) - L;
    
    Wm = [lambda / (L + lambda), repmat(1/(2*(L + lambda)), 1, 2*L)];
    Wc = Wm; Wc(1) = Wc(1) + (1 - alpha^2 + beta);

    % Generate sigma points
    S = chol((L + lambda) * P, 'lower');
    sigma_pts = [x, x + S, x - S];

    % Predict sigma points
    for i = 1:(2*L + 1)
        sp = sigma_pts(:, i);
        theta = sp(4);
        sigma_pts(:, i) = [sp(1) + sp(3)*cos(theta)*dt;
                           sp(2) + sp(3)*sin(theta)*dt;
                           sp(3);
                           sp(4)];
    end

    % Predicted mean
    x_pred = sigma_pts * Wm';
    
    % Predicted covariance
    P_pred = Q;
    for i = 1:(2*L + 1)
        diff = sigma_pts(:, i) - x_pred;
        P_pred = P_pred + Wc(i) * (diff * diff');
    end

    % Predict measurements
    Z = H * sigma_pts;
    z_pred = Z * Wm';

    P_zz = R;
    P_xz = zeros(L, size(R,1));
    for i = 1:(2*L + 1)
        dz = Z(:, i) - z_pred;
        dx = sigma_pts(:, i) - x_pred;
        P_zz = P_zz + Wc(i) * (dz * dz');
        P_xz = P_xz + Wc(i) * (dx * dz');
    end

    K = P_xz / P_zz;
    x_new = x_pred + K * (y - z_pred);
    P_new = P_pred - K * P_zz * K';
end
