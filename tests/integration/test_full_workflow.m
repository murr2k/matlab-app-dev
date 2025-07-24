classdef test_full_workflow < matlab.unittest.TestCase
    %TEST_FULL_WORKFLOW Integration tests for the complete workflow
    
    methods (Test)
        function test_pendulum_to_phase_space_visualization(testCase)
            % Test complete workflow from simulation to visualization
            
            % Run pendulum simulation
            L = 1;
            theta0 = pi/4;
            omega0 = 0;
            tspan = [0 10];
            
            [t, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan);
            
            % Create phase space plot
            fig = figure('Visible', 'off');
            plot_phase_space(t, theta, omega, 'Title', 'Test Phase Space');
            
            % Verify figure was created
            testCase.verifyNotEmpty(fig, 'Figure should be created');
            
            % Clean up
            close(fig);
        end
        
        function test_combined_simulations(testCase)
            % Test running multiple simulations together
            
            % Pendulum
            [t1, theta1, ~] = pendulum_simulation(1, pi/6, 0, [0 5]);
            
            % Particle
            force = @(t, x, v) [0; 0; -9.81];
            [t2, pos2, ~] = particle_dynamics(1, force, [0;0;10], [5;0;5], [0 2]);
            
            % Wave
            [u3, x3, t3] = wave_equation_solver(10, 2, 2, ...
                @(x) sin(pi*x/10), @(x) zeros(size(x)), 'nx', 50, 'nt', 100);
            
            % Verify all simulations produced output
            testCase.verifyNotEmpty(t1, 'Pendulum should produce output');
            testCase.verifyNotEmpty(t2, 'Particle dynamics should produce output');
            testCase.verifyNotEmpty(t3, 'Wave equation should produce output');
            
            % Verify dimensions are consistent
            testCase.verifyEqual(size(theta1, 1), size(t1, 1));
            testCase.verifyEqual(size(pos2, 1), size(t2, 1));
            testCase.verifyEqual(size(u3), [50, 100]);
        end
        
        function test_error_propagation(testCase)
            % Test that errors are properly handled through the workflow
            
            % Test invalid pendulum parameters
            testCase.verifyError(@() pendulum_simulation(-1, 0, 0, [0 1]), ...
                'MATLAB:expectedPositive');
            
            % Test invalid force function
            bad_force = @(t, x, v) [1; 2];  % Wrong dimension
            testCase.verifyError(@() particle_dynamics(1, bad_force, [0;0;0], [0;0;0], [0 1]), ...
                'MATLAB:dimagree');
        end
        
        function test_performance_benchmarks(testCase)
            % Test that simulations complete in reasonable time
            
            % Pendulum benchmark
            tic;
            pendulum_simulation(1, pi/4, 0, [0 10], 'damping', 0.1);
            pendulum_time = toc;
            
            testCase.verifyLessThan(pendulum_time, 1, ...
                'Pendulum simulation should complete in less than 1 second');
            
            % Particle dynamics benchmark
            tic;
            force = @(t, x, v) -0.1 * v;  % Simple drag
            particle_dynamics(1, force, [0;0;0], [10;10;10], [0 5]);
            particle_time = toc;
            
            testCase.verifyLessThan(particle_time, 1, ...
                'Particle dynamics should complete in less than 1 second');
            
            % Wave equation benchmark (smaller grid for speed)
            tic;
            wave_equation_solver(10, 2, 2, @(x) sin(pi*x/10), @(x) zeros(size(x)), ...
                'nx', 50, 'nt', 50);
            wave_time = toc;
            
            testCase.verifyLessThan(wave_time, 2, ...
                'Wave equation should complete in less than 2 seconds');
        end
        
        function test_file_structure(testCase)
            % Test that all expected files exist
            
            % Check source files
            testCase.verifyTrue(exist('src/physics/pendulum_simulation.m', 'file') == 2);
            testCase.verifyTrue(exist('src/physics/particle_dynamics.m', 'file') == 2);
            testCase.verifyTrue(exist('src/physics/wave_equation_solver.m', 'file') == 2);
            testCase.verifyTrue(exist('src/utilities/plot_phase_space.m', 'file') == 2);
            
            % Check test files
            testCase.verifyTrue(exist('tests/unit/test_pendulum_simulation.m', 'file') == 2);
            testCase.verifyTrue(exist('tests/unit/test_particle_dynamics.m', 'file') == 2);
            testCase.verifyTrue(exist('tests/unit/test_wave_equation_solver.m', 'file') == 2);
            
            % Check configuration files
            testCase.verifyTrue(exist('buildfile.m', 'file') == 2);
            testCase.verifyTrue(exist('startup.m', 'file') == 2);
            testCase.verifyTrue(exist('README.md', 'file') == 2);
        end
    end
end