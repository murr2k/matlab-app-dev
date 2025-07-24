classdef test_wave_equation_solver < matlab.unittest.TestCase
    %TEST_WAVE_EQUATION_SOLVER Unit tests for wave_equation_solver function
    
    methods (Test)
        function test_basic_functionality(testCase)
            % Test basic wave equation solver
            L = 10;
            T = 5;
            c = 2;
            initial_u = @(x) sin(pi*x/L);
            initial_ut = @(x) zeros(size(x));
            
            [u, x, t] = wave_equation_solver(L, T, c, initial_u, initial_ut);
            
            % Verify output dimensions
            testCase.verifySize(u, [100, 200], 'Default grid should be 100x200');
            testCase.verifyEqual(x(1), 0, 'First x point should be 0');
            testCase.verifyEqual(x(end), L, 'Last x point should be L');
            testCase.verifyEqual(t(1), 0, 'First time point should be 0');
            testCase.verifyEqual(t(end), T, 'Last time point should be T');
        end
        
        function test_standing_wave(testCase)
            % Test standing wave solution
            L = 1;
            T = 2;
            c = 1;
            n = 1;  % Mode number
            initial_u = @(x) sin(n*pi*x/L);
            initial_ut = @(x) zeros(size(x));
            
            [u, x, t] = wave_equation_solver(L, T, c, initial_u, initial_ut, ...
                'nx', 50, 'nt', 100);
            
            % Analytical solution for standing wave
            omega = n*pi*c/L;
            u_analytical = zeros(length(x), length(t));
            for i = 1:length(t)
                u_analytical(:, i) = sin(n*pi*x/L) * cos(omega*t(i));
            end
            
            % Compare numerical and analytical solutions
            error = abs(u - u_analytical);
            testCase.verifyLessThan(max(error(:)), 0.1, ...
                'Standing wave solution should match analytical');
        end
        
        function test_wave_propagation(testCase)
            % Test wave propagation
            L = 20;
            T = 5;
            c = 2;
            
            % Gaussian pulse
            x0 = L/4;
            sigma = 0.5;
            initial_u = @(x) exp(-(x-x0).^2/(2*sigma^2));
            initial_ut = @(x) zeros(size(x));
            
            [u, x, ~] = wave_equation_solver(L, T, c, initial_u, initial_ut, ...
                'nx', 200, 'nt', 250);
            
            % Wave should split and propagate in both directions
            u_initial = u(:, 1);
            u_mid = u(:, 125);  % Middle time
            
            % Find peaks in middle time (simple alternative to findpeaks)
            threshold = 0.1;
            peaks = [];
            for i = 2:length(u_mid)-1
                if u_mid(i) > threshold && u_mid(i) > u_mid(i-1) && u_mid(i) > u_mid(i+1)
                    peaks = [peaks, i];
                end
            end
            
            testCase.verifyEqual(length(peaks), 2, ...
                'Should have two peaks (left and right propagating waves)');
        end
        
        function test_energy_conservation(testCase)
            % Test energy conservation for wave equation
            L = 10;
            T = 5;
            c = 1;
            initial_u = @(x) sin(2*pi*x/L);
            initial_ut = @(x) zeros(size(x));
            
            [u, x, t] = wave_equation_solver(L, T, c, initial_u, initial_ut, ...
                'nx', 100, 'nt', 200);
            
            dx = x(2) - x(1);
            
            % Calculate energy at each time step
            energy = zeros(1, length(t));
            for i = 1:length(t)
                % Kinetic energy (using time derivative)
                if i > 1 && i < length(t)
                    u_t = (u(:, i+1) - u(:, i-1)) / (2*(t(2)-t(1)));
                elseif i == 1
                    u_t = (u(:, 2) - u(:, 1)) / (t(2)-t(1));
                else
                    u_t = (u(:, end) - u(:, end-1)) / (t(2)-t(1));
                end
                
                % Potential energy (using spatial derivative)
                u_x = gradient(u(:, i), dx);
                
                % Total energy (integrate over space)
                energy(i) = 0.5 * trapz(x, u_t.^2 + c^2 * u_x.^2);
            end
            
            % Energy should be approximately conserved
            energy_variation = (energy - energy(1)) / energy(1);
            testCase.verifyLessThan(max(abs(energy_variation)), 0.1, ...
                'Energy should be conserved within 10%');
        end
        
        function test_boundary_conditions(testCase)
            % Test different boundary conditions
            L = 5;
            T = 2;
            c = 1;
            % Use initial condition that's zero at boundaries for cleaner test
            initial_u = @(x) sin(pi*x/L);
            initial_ut = @(x) zeros(size(x));
            
            % Dirichlet boundary conditions
            [u_dir, ~, ~] = wave_equation_solver(L, T, c, initial_u, initial_ut, ...
                'boundary', 'dirichlet', 'nx', 50, 'nt', 100);
            
            % Neumann boundary conditions
            [u_neu, ~, ~] = wave_equation_solver(L, T, c, initial_u, initial_ut, ...
                'boundary', 'neumann', 'nx', 50, 'nt', 100);
            
            % Check boundary conditions
            testCase.verifyEqual(u_dir([1, end], :), zeros(2, 100), 'AbsTol', 1e-10, ...
                'Dirichlet BC should have zero at boundaries');
            
            % For Neumann, check that the boundary implementation is working
            % by verifying the values match (zero derivative condition)
            for t = 1:size(u_neu, 2)
                testCase.verifyEqual(u_neu(1, t), u_neu(2, t), 'AbsTol', 1e-10, ...
                    sprintf('Neumann BC at left boundary, time step %d', t));
                testCase.verifyEqual(u_neu(end, t), u_neu(end-1, t), 'AbsTol', 1e-10, ...
                    sprintf('Neumann BC at right boundary, time step %d', t));
            end
        end
        
        function test_input_validation(testCase)
            % Test input validation
            initial_u = @(x) sin(x);
            initial_ut = @(x) zeros(size(x));
            
            testCase.verifyError(@() wave_equation_solver(-1, 1, 1, initial_u, initial_ut), ...
                'MATLAB:wave_equation_solver:expectedPositive', 'Negative length should error');
            
            testCase.verifyError(@() wave_equation_solver(1, -1, 1, initial_u, initial_ut), ...
                'MATLAB:wave_equation_solver:expectedPositive', 'Negative time should error');
            
            testCase.verifyError(@() wave_equation_solver(1, 1, -1, initial_u, initial_ut), ...
                'MATLAB:wave_equation_solver:expectedPositive', 'Negative wave speed should error');
        end
    end
end