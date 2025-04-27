% Jacobian Functions
function F = compute_jacobian_f(x)
    % Numerical Jacobian calculation instead of symbolic 
    % to avoid potential issues with symbolic differentiation
    nx = length(x);
    F = zeros(nx, nx);
    epsilon = 1e-6;
    
    % Function value at current point
    f0 = [
        x(6)*x(2)*x(1)^max(0,min(1,(1-x(4)))) - (x(8)+x(10))*x(1);
        x(3); 0; x(5); 0; x(7); 0; x(9); 0; x(11); 0
    ];
    
    % Compute Jacobian numerically
    for i = 1:nx
        x_perturbed = x;
        x_perturbed(i) = x_perturbed(i) + epsilon;
        
        f_perturbed = [
            x_perturbed(6)*x_perturbed(2)*x_perturbed(1)^max(0,min(1,(1-x_perturbed(4)))) - (x_perturbed(8)+x_perturbed(10))*x_perturbed(1);
            x_perturbed(3); 0; x_perturbed(5); 0; x_perturbed(7); 0; x_perturbed(9); 0; x_perturbed(11); 0
        ];
        
        F(:, i) = (f_perturbed - f0) / epsilon;
    end
end