classdef test_pendulum_simulation < matlab.unittest.TestCase
    %TEST_PENDULUM_SIMULATION Unit tests for pendulum_simulation function
    
    methods (Test)
        function test_basic_functionality(testCase)
            % Test basic pendulum simulation
            L = 1;
            theta0 = pi/6;
            omega0 = 0;
            tspan = [0 5];
            
            [t, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan);
            
            % Verify outputs
            testCase.verifyEqual(size(t, 2), 1, 'Time should be column vector');
            testCase.verifyEqual(size(theta), size(t), 'Theta should match time size');
            testCase.verifyEqual(size(omega), size(t), 'Omega should match time size');
            testCase.verifyEqual(t(1), 0, 'Start time should be 0');
            testCase.verifyGreaterThanOrEqual(t(end), 5, 'End time should be at least 5');
            testCase.verifyEqual(theta(1), theta0, 'AbsTol', 1e-10, 'Initial angle should match');
            testCase.verifyEqual(omega(1), omega0, 'AbsTol', 1e-10, 'Initial velocity should match');
        end
        
        function test_energy_conservation(testCase)
            % Test energy conservation for undamped pendulum
            L = 1;
            g = 9.81;
            theta0 = pi/4;
            omega0 = 0;
            tspan = [0 10];
            
            [~, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan, 'gravity', g);
            
            % Calculate total energy
            KE = 0.5 * L^2 * omega.^2;  % Kinetic energy (per unit mass)
            PE = g * L * (1 - cos(theta));  % Potential energy (per unit mass)
            E_total = KE + PE;
            
            % Energy should be conserved
            E_initial = E_total(1);
            E_variation = abs(E_total - E_initial) / E_initial;
            testCase.verifyLessThan(max(E_variation), 0.01, 'Energy should be conserved within 1%');
        end
        
        function test_damped_pendulum(testCase)
            % Test damped pendulum - energy should decrease
            L = 1;
            theta0 = pi/3;
            omega0 = 0;
            tspan = [0 20];
            damping = 0.5;
            
            [~, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan, 'damping', damping);
            
            % Calculate total energy
            g = 9.81;
            KE = 0.5 * L^2 * omega.^2;
            PE = g * L * (1 - cos(theta));
            E_total = KE + PE;
            
            % Energy should decrease monotonically (approximately)
            E_diff = diff(E_total);
            testCase.verifyLessThan(sum(E_diff > 0), length(E_diff) * 0.1, ...
                'Energy should mostly decrease for damped system');
        end
        
        function test_small_angle_approximation(testCase)
            % For small angles, pendulum should behave like harmonic oscillator
            L = 1;
            g = 9.81;
            theta0 = 0.1;  % Small angle
            omega0 = 0;
            tspan = [0 10];
            
            [t, theta, ~] = pendulum_simulation(L, theta0, omega0, tspan, 'gravity', g);
            
            % Theoretical solution for small angles
            omega_natural = sqrt(g/L);
            theta_theory = theta0 * cos(omega_natural * t);
            
            % Compare with numerical solution
            error = abs(theta - theta_theory);
            testCase.verifyLessThan(max(error), 0.01, ...
                'Small angle approximation should be accurate');
        end
        
        function test_input_validation(testCase)
            % Test input validation
            testCase.verifyError(@() pendulum_simulation(-1, 0, 0, [0 1]), ...
                'MATLAB:expectedPositive', 'Negative length should error');
            
            testCase.verifyError(@() pendulum_simulation(1, 0, 0, [1]), ...
                'MATLAB:expectedNumel', 'Single timespan should error');
            
            testCase.verifyError(@() pendulum_simulation(1, 0, 0, [0 1], 'damping', -1), ...
                'MATLAB:InputParser:ArgumentFailedValidation', 'Negative damping should error');
        end
    end
end