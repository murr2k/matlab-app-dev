classdef test_particle_dynamics < matlab.unittest.TestCase
    %TEST_PARTICLE_DYNAMICS Unit tests for particle_dynamics function
    
    methods (Test)
        function test_free_fall(testCase)
            % Test particle in free fall
            mass = 1;
            g = 9.81;
            force_func = @(t, x, v) [0; 0; -mass*g];
            x0 = [0; 0; 100];
            v0 = [0; 0; 0];
            tspan = [0 4];
            
            [t, position, velocity] = particle_dynamics(mass, force_func, x0, v0, tspan);
            
            % Verify output dimensions
            testCase.verifySize(position, [length(t), 3], 'Position should be Nx3');
            testCase.verifySize(velocity, [length(t), 3], 'Velocity should be Nx3');
            
            % Verify initial conditions
            testCase.verifyEqual(position(1, :), x0', 'AbsTol', 1e-10);
            testCase.verifyEqual(velocity(1, :), v0', 'AbsTol', 1e-10);
            
            % Verify analytical solution for free fall
            z_analytical = x0(3) - 0.5*g*t.^2;
            vz_analytical = -g*t;
            
            testCase.verifyEqual(position(:, 3), z_analytical, 'RelTol', 1e-6, ...
                'Vertical position should match analytical solution');
            testCase.verifyEqual(velocity(:, 3), vz_analytical, 'RelTol', 1e-6, ...
                'Vertical velocity should match analytical solution');
        end
        
        function test_projectile_motion(testCase)
            % Test projectile motion
            mass = 0.5;
            g = 9.81;
            force_func = @(t, x, v) [0; 0; -mass*g];
            x0 = [0; 0; 0];
            v0 = [10; 0; 10];  % 45-degree launch
            tspan = [0 2];
            
            [t, position, ~] = particle_dynamics(mass, force_func, x0, v0, tspan);
            
            % Verify horizontal motion (constant velocity)
            x_analytical = v0(1) * t;
            testCase.verifyEqual(position(:, 1), x_analytical, 'RelTol', 1e-6, ...
                'Horizontal position should be linear');
            
            % Verify y-position remains zero (no y-velocity)
            testCase.verifyEqual(position(:, 2), zeros(size(t)), 'AbsTol', 1e-10, ...
                'Y-position should remain zero');
        end
        
        function test_harmonic_oscillator(testCase)
            % Test 1D harmonic oscillator
            mass = 1;
            k = 4;  % Spring constant
            force_func = @(t, x, v) [-k*x(1); 0; 0];
            x0 = [1; 0; 0];
            v0 = [0; 0; 0];
            tspan = [0 10];
            
            [t, position, ~] = particle_dynamics(mass, force_func, x0, v0, tspan);
            
            % Analytical solution
            omega = sqrt(k/mass);
            x_analytical = x0(1) * cos(omega * t);
            
            % Verify oscillatory behavior
            testCase.verifyEqual(position(:, 1), x_analytical, 'RelTol', 1e-4, ...
                'Should match harmonic oscillator solution');
        end
        
        function test_drag_force(testCase)
            % Test particle with drag force
            mass = 1;
            b = 0.1;  % Drag coefficient
            force_func = @(t, x, v) -b * v;
            x0 = [0; 0; 0];
            v0 = [10; 5; 0];
            tspan = [0 5];
            
            [~, ~, velocity] = particle_dynamics(mass, force_func, x0, v0, tspan);
            
            % Velocity should decay exponentially
            v_magnitude_initial = norm(v0);
            v_magnitude_final = norm(velocity(end, :));
            
            testCase.verifyLessThan(v_magnitude_final, v_magnitude_initial * 0.1, ...
                'Velocity should decay significantly due to drag');
        end
        
        function test_different_solvers(testCase)
            % Test different ODE solvers give similar results
            mass = 1;
            force_func = @(t, x, v) [0; 0; -9.81*mass];
            x0 = [0; 0; 10];
            v0 = [1; 0; 5];
            tspan = [0 1];
            
            [t45, pos45, ~] = particle_dynamics(mass, force_func, x0, v0, tspan, 'method', 'ode45');
            [t23, pos23, ~] = particle_dynamics(mass, force_func, x0, v0, tspan, 'method', 'ode23');
            
            % Interpolate to common time points
            t_common = linspace(0, 1, 50)';
            pos45_interp = interp1(t45, pos45, t_common);
            pos23_interp = interp1(t23, pos23, t_common);
            
            % Results should be similar
            testCase.verifyEqual(pos45_interp, pos23_interp, 'RelTol', 1e-3, ...
                'Different solvers should give similar results');
        end
        
        function test_input_validation(testCase)
            % Test input validation
            force_func = @(t, x, v) [0; 0; 0];
            
            testCase.verifyError(@() particle_dynamics(-1, force_func, [0;0;0], [0;0;0], [0 1]), ...
                'MATLAB:expectedPositive', 'Negative mass should error');
            
            testCase.verifyError(@() particle_dynamics(1, force_func, [0;0], [0;0;0], [0 1]), ...
                'MATLAB:expectedNumel', 'Wrong position dimension should error');
            
            testCase.verifyError(@() particle_dynamics(1, force_func, [0;0;0], [0;0], [0 1]), ...
                'MATLAB:expectedNumel', 'Wrong velocity dimension should error');
        end
    end
end