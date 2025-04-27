function H = compute_jacobian_h(x)
    % Numerical Jacobian calculation for measurement function
    nx = length(x);
    H = zeros(5, nx);
    epsilon = 1e-6;
    
    % Function value at current point
    h0 = [x(1); x(6); x(8); x(10); x(2)*x(1)^max(0,min(1,(1-x(4))))];
    
    % Compute Jacobian numerically
    for i = 1:nx
        x_perturbed = x;
        x_perturbed(i) = x_perturbed(i) + epsilon;
        
        h_perturbed = [x_perturbed(1); x_perturbed(6); x_perturbed(8); x_perturbed(10); x_perturbed(2)*x_perturbed(1)^max(0,min(1,(1-x_perturbed(4))))];
        
        H(:, i) = (h_perturbed - h0) / epsilon;
    end
end