%RUN_ALL_DEMOS Demonstrates all physics simulations in the project
%   This script runs examples of all the physics simulations available
%   in the project, creating visualizations for each.

fprintf('Running all physics simulation demos...\n\n');

%% Demo 1: Pendulum Simulation
fprintf('1. Pendulum Simulation Demo\n');
fprintf('   Comparing undamped vs damped pendulum\n');

% Parameters
L = 1;  % 1 meter pendulum
theta0 = pi/3;  % 60 degrees initial angle
omega0 = 0;  % Start from rest
tspan = [0 20];

% Run simulations
[t1, theta1, omega1] = pendulum_simulation(L, theta0, omega0, tspan);
[t2, theta2, omega2] = pendulum_simulation(L, theta0, omega0, tspan, 'damping', 0.5);

% Plot results
figure('Name', 'Pendulum Simulation');
subplot(2,2,1);
plot(t1, theta1*180/pi, 'b-', t2, theta2*180/pi, 'r--');
xlabel('Time (s)'); ylabel('Angle (degrees)');
title('Pendulum Angle vs Time');
legend('Undamped', 'Damped', 'Location', 'best');
grid on;

subplot(2,2,2);
plot(t1, omega1, 'b-', t2, omega2, 'r--');
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity vs Time');
legend('Undamped', 'Damped', 'Location', 'best');
grid on;

subplot(2,2,[3,4]);
plot_phase_space(t1, theta1, omega1, 'Title', 'Phase Space - Undamped Pendulum');

%% Demo 2: Particle Dynamics
fprintf('\n2. Particle Dynamics Demo\n');
fprintf('   Projectile motion with air resistance\n');

% Parameters
mass = 0.5;  % 0.5 kg
g = 9.81;
drag_coeff = 0.1;

% Force function with gravity and drag
force_with_drag = @(t, x, v) [0; 0; -mass*g] - drag_coeff * norm(v) * v;

% Initial conditions
x0 = [0; 0; 0];
v0 = [20; 0; 20];  % 45-degree launch at ~28 m/s
tspan = [0 4];

% Run simulations
[t_drag, pos_drag, vel_drag] = particle_dynamics(mass, force_with_drag, x0, v0, tspan);

% Compare with no drag
force_no_drag = @(t, x, v) [0; 0; -mass*g];
[t_no_drag, pos_no_drag, ~] = particle_dynamics(mass, force_no_drag, x0, v0, tspan);

% Plot trajectories
figure('Name', 'Particle Dynamics');
subplot(1,2,1);
plot3(pos_drag(:,1), pos_drag(:,2), pos_drag(:,3), 'r-', 'LineWidth', 2);
hold on;
plot3(pos_no_drag(:,1), pos_no_drag(:,2), pos_no_drag(:,3), 'b--', 'LineWidth', 2);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Projectile Motion');
legend('With air resistance', 'Without air resistance', 'Location', 'best');
grid on;
view(45, 30);

subplot(1,2,2);
plot(pos_drag(:,1), pos_drag(:,3), 'r-', 'LineWidth', 2);
hold on;
plot(pos_no_drag(:,1), pos_no_drag(:,3), 'b--', 'LineWidth', 2);
xlabel('Horizontal Distance (m)'); ylabel('Height (m)');
title('2D Trajectory View');
legend('With air resistance', 'Without air resistance', 'Location', 'best');
grid on;

%% Demo 3: Wave Equation
fprintf('\n3. Wave Equation Demo\n');
fprintf('   Standing wave and wave packet propagation\n');

% Standing wave parameters
L = 10;
T = 4;
c = 2;
initial_u_standing = @(x) sin(2*pi*x/L);
initial_ut_standing = @(x) zeros(size(x));

% Wave packet parameters
x0 = L/3;
sigma = 0.5;
k = 5;  % wave number
initial_u_packet = @(x) exp(-(x-x0).^2/(2*sigma^2)) .* cos(k*x);
initial_ut_packet = @(x) -c*k*exp(-(x-x0).^2/(2*sigma^2)) .* sin(k*x);

% Solve both cases
[u1, x, t] = wave_equation_solver(L, T, c, initial_u_standing, initial_ut_standing, ...
    'nx', 100, 'nt', 200);
[u2, ~, ~] = wave_equation_solver(L, T, c, initial_u_packet, initial_ut_packet, ...
    'nx', 100, 'nt', 200);

% Create animation figure
figure('Name', 'Wave Equation Solutions');

% Select time points to show
time_indices = round(linspace(1, length(t), 8));

for i = 1:8
    idx = time_indices(i);
    
    subplot(4,4,i);
    plot(x, u1(:,idx), 'b-', 'LineWidth', 1.5);
    ylim([-1.5 1.5]);
    xlabel('x'); ylabel('u');
    title(sprintf('Standing Wave: t=%.2f', t(idx)));
    grid on;
    
    subplot(4,4,i+8);
    plot(x, u2(:,idx), 'r-', 'LineWidth', 1.5);
    ylim([-1.5 1.5]);
    xlabel('x'); ylabel('u');
    title(sprintf('Wave Packet: t=%.2f', t(idx)));
    grid on;
end

%% Demo 4: Energy Analysis
fprintf('\n4. Energy Conservation Analysis\n');
fprintf('   Checking energy conservation in undamped systems\n');

% Pendulum energy
g = 9.81;
KE_pendulum = 0.5 * L^2 * omega1.^2;
PE_pendulum = g * L * (1 - cos(theta1));
E_pendulum = KE_pendulum + PE_pendulum;

figure('Name', 'Energy Conservation');
subplot(2,1,1);
plot(t1, KE_pendulum, 'b-', t1, PE_pendulum, 'r-', t1, E_pendulum, 'k--', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Energy per unit mass (J/kg)');
title('Pendulum Energy Conservation');
legend('Kinetic', 'Potential', 'Total', 'Location', 'best');
grid on;

% Wave energy density
dx = x(2) - x(1);
dt = t(2) - t(1);
wave_energy = zeros(1, length(t));

for i = 2:length(t)-1
    u_t = (u1(:, i+1) - u1(:, i-1)) / (2*dt);
    u_x = gradient(u1(:, i), dx);
    energy_density = 0.5 * (u_t.^2 + c^2 * u_x.^2);
    wave_energy(i) = trapz(x, energy_density);
end

subplot(2,1,2);
plot(t(2:end-1), wave_energy(2:end-1), 'g-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Total Energy');
title('Wave Energy Conservation');
grid on;

fprintf('\nAll demos completed successfully!\n');
fprintf('Figures created:\n');
fprintf('  - Figure 1: Pendulum Simulation\n');
fprintf('  - Figure 2: Particle Dynamics\n');
fprintf('  - Figure 3: Wave Equation Solutions\n');
fprintf('  - Figure 4: Energy Conservation\n');