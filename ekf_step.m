function [x_new, P_new] = ekf_step(x, P, y, Q, R, H, dt)
    % Predict
    theta = x(4);
    F = [1 0 cos(theta)*dt  -x(3)*sin(theta)*dt;
         0 1 sin(theta)*dt   x(3)*cos(theta)*dt;
         0 0 1               0;
         0 0 0               1];

    x_pred = zeros(4,1);
    x_pred(1) = x(1) + x(3) * cos(theta) * dt;
    x_pred(2) = x(2) + x(3) * sin(theta) * dt;
    x_pred(3) = x(3);
    x_pred(4) = x(4);

    P_pred = F * P * F' + Q;

    % Update
    y_pred = H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_new = x_pred + K * (y - y_pred);
    P_new = (eye(4) - K * H) * P_pred;
end

