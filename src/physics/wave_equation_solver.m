function [u, x, t] = wave_equation_solver(L, T, c, initial_u, initial_ut, varargin)
%WAVE_EQUATION_SOLVER Solves the 1D wave equation using finite differences
%   [u, x, t] = WAVE_EQUATION_SOLVER(L, T, c, initial_u, initial_ut)
%   solves the wave equation u_tt = c^2 * u_xx on domain [0, L] x [0, T]
%
%   Inputs:
%       L - spatial domain length
%       T - time duration
%       c - wave speed
%       initial_u - function handle for initial displacement u(x,0)
%       initial_ut - function handle for initial velocity u_t(x,0)
%
%   Optional parameters:
%       'nx' - number of spatial points (default: 100)
%       'nt' - number of time points (default: 200)
%       'boundary' - boundary conditions: 'dirichlet' or 'neumann' (default: 'dirichlet')

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'nx', 100, @(x) isnumeric(x) && x > 10);
    addParameter(p, 'nt', 200, @(x) isnumeric(x) && x > 10);
    addParameter(p, 'boundary', 'dirichlet', @(x) ismember(x, {'dirichlet', 'neumann'}));
    parse(p, varargin{:});
    
    nx = p.Results.nx;
    nt = p.Results.nt;
    boundary = p.Results.boundary;
    
    % Validate inputs
    validateattributes(L, {'numeric'}, {'positive', 'scalar'}, 'wave_equation_solver', 'L');
    validateattributes(T, {'numeric'}, {'positive', 'scalar'}, 'wave_equation_solver', 'T');
    validateattributes(c, {'numeric'}, {'positive', 'scalar'}, 'wave_equation_solver', 'c');
    validateattributes(initial_u, {'function_handle'}, {}, 'wave_equation_solver', 'initial_u');
    validateattributes(initial_ut, {'function_handle'}, {}, 'wave_equation_solver', 'initial_ut');
    
    % Create spatial and temporal grids
    dx = L / (nx - 1);
    dt = T / (nt - 1);
    x = linspace(0, L, nx)';
    t = linspace(0, T, nt);
    
    % Check CFL condition
    CFL = c * dt / dx;
    if CFL > 1
        warning('CFL condition violated: c*dt/dx = %.2f > 1. Solution may be unstable.', CFL);
    end
    
    % Initialize solution matrix
    u = zeros(nx, nt);
    
    % Set initial conditions
    u(:, 1) = initial_u(x);
    
    % Apply boundary conditions to initial condition
    if strcmp(boundary, 'dirichlet')
        u([1, end], 1) = 0;
    else % Neumann
        u(1, 1) = u(2, 1);
        u(end, 1) = u(end-1, 1);
    end
    
    % Use forward difference for first time step
    u(:, 2) = u(:, 1) + dt * initial_ut(x) + ...
              0.5 * (c * dt / dx)^2 * (circshift(u(:, 1), -1) - 2*u(:, 1) + circshift(u(:, 1), 1));
    
    % Apply boundary conditions for first time step
    if strcmp(boundary, 'dirichlet')
        u([1, end], 2) = 0;
    else % Neumann
        u(1, 2) = u(2, 2);
        u(end, 2) = u(end-1, 2);
    end
    
    % Time stepping using central differences
    r2 = (c * dt / dx)^2;
    
    for n = 2:nt-1
        % Interior points
        u(2:end-1, n+1) = 2*(1 - r2)*u(2:end-1, n) + r2*(u(3:end, n) + u(1:end-2, n)) - u(2:end-1, n-1);
        
        % Apply boundary conditions
        if strcmp(boundary, 'dirichlet')
            u(1, n+1) = 0;
            u(end, n+1) = 0;
        else % Neumann
            u(1, n+1) = u(2, n+1);
            u(end, n+1) = u(end-1, n+1);
        end
    end
end