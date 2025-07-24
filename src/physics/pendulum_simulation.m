function [t, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan, varargin)
%PENDULUM_SIMULATION Simulates a simple pendulum using ODE45
%   [t, theta, omega] = PENDULUM_SIMULATION(L, theta0, omega0, tspan)
%   simulates a simple pendulum with length L, initial angle theta0,
%   initial angular velocity omega0, over time span tspan.
%
%   [t, theta, omega] = PENDULUM_SIMULATION(L, theta0, omega0, tspan, 'damping', b)
%   includes damping coefficient b.
%
%   Example:
%       [t, theta, omega] = pendulum_simulation(1, pi/4, 0, [0 10]);
%       plot(t, theta*180/pi);
%       xlabel('Time (s)'); ylabel('Angle (degrees)');

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'damping', 0, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'gravity', 9.81, @(x) isnumeric(x) && x > 0);
    parse(p, varargin{:});
    
    b = p.Results.damping;
    g = p.Results.gravity;
    
    % Validate inputs
    validateattributes(L, {'numeric'}, {'positive', 'scalar'}, 'pendulum_simulation', 'L');
    validateattributes(theta0, {'numeric'}, {'scalar'}, 'pendulum_simulation', 'theta0');
    validateattributes(omega0, {'numeric'}, {'scalar'}, 'pendulum_simulation', 'omega0');
    validateattributes(tspan, {'numeric'}, {'vector', 'numel', 2}, 'pendulum_simulation', 'tspan');
    
    % Define the ODE system
    function dydt = pendulum_ode(~, y)
        % y(1) = theta, y(2) = omega
        dydt = zeros(2,1);
        dydt(1) = y(2);  % dtheta/dt = omega
        dydt(2) = -(g/L)*sin(y(1)) - (b/L)*y(2);  % domega/dt
    end
    
    % Solve the ODE
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    [t, y] = ode45(@pendulum_ode, tspan, [theta0; omega0], options);
    
    theta = y(:,1);
    omega = y(:,2);
end