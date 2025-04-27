function dx = dynamics(~, x, omega)
    dx = zeros(4,1);
    dx(1) = x(3) * cos(x(4));
    dx(2) = x(3) * sin(x(4));
    dx(3) = 0;
    dx(4) = omega;
end
