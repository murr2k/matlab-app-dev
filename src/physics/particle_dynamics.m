function [t, position, velocity] = particle_dynamics(mass, force_func, x0, v0, tspan, varargin)
%PARTICLE_DYNAMICS Simulates particle motion under external forces
%   [t, position, velocity] = PARTICLE_DYNAMICS(mass, force_func, x0, v0, tspan)
%   simulates a particle with given mass under external force defined by
%   force_func, starting at position x0 with velocity v0.
%
%   Inputs:
%       mass - particle mass (kg)
%       force_func - function handle @(t, x, v) returning force vector
%       x0 - initial position [x; y; z] (m)
%       v0 - initial velocity [vx; vy; vz] (m/s)
%       tspan - time span [t0 tf] (s)
%
%   Example:
%       % Particle in uniform gravity field
%       F = @(t, x, v) [0; 0; -9.81*mass];
%       [t, pos, vel] = particle_dynamics(1, F, [0;0;10], [5;0;5], [0 2]);

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'method', 'ode45', @(x) ismember(x, {'ode45', 'ode23', 'ode113'}));
    addParameter(p, 'tolerance', 1e-6, @(x) isnumeric(x) && x > 0);
    parse(p, varargin{:});
    
    % Validate inputs
    validateattributes(mass, {'numeric'}, {'positive', 'scalar'}, 'particle_dynamics', 'mass');
    validateattributes(x0, {'numeric'}, {'vector', 'numel', 3}, 'particle_dynamics', 'x0');
    validateattributes(v0, {'numeric'}, {'vector', 'numel', 3}, 'particle_dynamics', 'v0');
    validateattributes(tspan, {'numeric'}, {'vector', 'numel', 2}, 'particle_dynamics', 'tspan');
    
    % Reshape to column vectors
    x0 = x0(:);
    v0 = v0(:);
    
    % Define the ODE system
    function dydt = dynamics_ode(t, y)
        % y = [x; y; z; vx; vy; vz]
        position = y(1:3);
        velocity = y(4:6);
        
        % Calculate acceleration from force
        force = force_func(t, position, velocity);
        acceleration = force(:) / mass;
        
        % Return derivatives
        dydt = [velocity; acceleration];
    end
    
    % Set ODE options
    options = odeset('RelTol', p.Results.tolerance, 'AbsTol', p.Results.tolerance/100);
    
    % Select solver
    solver = str2func(p.Results.method);
    
    % Solve the ODE
    initial_state = [x0; v0];
    [t, y] = solver(@dynamics_ode, tspan, initial_state, options);
    
    % Extract position and velocity
    position = y(:, 1:3);
    velocity = y(:, 4:6);
end