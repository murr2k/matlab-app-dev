# MATLAB Physics Simulations and Simulink Project

This project provides a MATLAB development environment for physics simulations and Simulink models, with automated CI/CD using GitHub Actions.

## Project Structure

```
matlab/
├── .github/
│   └── workflows/
│       └── matlab-ci.yml    # CI/CD pipeline configuration
├── src/
│   ├── physics/             # Physics simulation functions
│   │   ├── pendulum_simulation.m
│   │   ├── particle_dynamics.m
│   │   └── wave_equation_solver.m
│   ├── simulink/            # Simulink models and scripts
│   │   └── create_mass_spring_damper_model.m
│   └── utilities/           # Helper functions
├── tests/
│   ├── unit/               # Unit tests
│   │   ├── test_pendulum_simulation.m
│   │   ├── test_particle_dynamics.m
│   │   └── test_wave_equation_solver.m
│   └── integration/        # Integration tests
├── docs/                   # Documentation
└── README.md
```

## Features

### Physics Simulations

1. **Pendulum Simulation** (`pendulum_simulation.m`)
   - Simulates simple and damped pendulum motion
   - Uses ODE45 for numerical integration
   - Supports custom gravity and damping parameters

2. **Particle Dynamics** (`particle_dynamics.m`)
   - Simulates particle motion under external forces
   - Supports 3D motion with custom force functions
   - Multiple ODE solver options

3. **Wave Equation Solver** (`wave_equation_solver.m`)
   - Solves 1D wave equation using finite differences
   - Supports Dirichlet and Neumann boundary conditions
   - Customizable grid resolution

### Simulink Models

- **Mass-Spring-Damper System**: A configurable model for mechanical system simulation

## Getting Started

### Prerequisites

- MATLAB R2023b or later
- Simulink (for Simulink models)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd matlab
   ```

2. Add project to MATLAB path:
   ```matlab
   addpath(genpath('src'));
   ```

### Running Simulations

#### Example: Pendulum Simulation
```matlab
% Simulate a pendulum with L=1m, initial angle=45°
L = 1;
theta0 = pi/4;
omega0 = 0;
tspan = [0 10];

[t, theta, omega] = pendulum_simulation(L, theta0, omega0, tspan);

% Plot results
figure;
subplot(2,1,1);
plot(t, theta*180/pi);
xlabel('Time (s)'); ylabel('Angle (degrees)');
title('Pendulum Angle vs Time');

subplot(2,1,2);
plot(t, omega);
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity vs Time');
```

#### Example: Particle Dynamics
```matlab
% Simulate projectile motion
mass = 1;
g = 9.81;
force_func = @(t, x, v) [0; 0; -mass*g];
x0 = [0; 0; 0];
v0 = [10; 0; 10];  % 45-degree launch
tspan = [0 2];

[t, position, velocity] = particle_dynamics(mass, force_func, x0, v0, tspan);

% Plot trajectory
plot3(position(:,1), position(:,2), position(:,3));
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Projectile Trajectory');
grid on;
```

#### Example: Wave Equation
```matlab
% Solve wave equation with standing wave initial condition
L = 10;
T = 5;
c = 2;
initial_u = @(x) sin(pi*x/L);
initial_ut = @(x) zeros(size(x));

[u, x, t] = wave_equation_solver(L, T, c, initial_u, initial_ut);

% Animate the wave
figure;
for i = 1:5:length(t)
    plot(x, u(:,i));
    ylim([-1.5 1.5]);
    xlabel('Position (m)');
    ylabel('Displacement');
    title(sprintf('Wave at t = %.2f s', t(i)));
    drawnow;
    pause(0.05);
end
```

### Creating Simulink Models

To create the mass-spring-damper Simulink model:
```matlab
create_mass_spring_damper_model();
```

This will create a `.slx` file that you can open and simulate in Simulink.

## Testing

### Running Tests Locally

Run all tests:
```matlab
results = runtests('tests/unit');
```

Run specific test:
```matlab
results = runtests('tests/unit/test_pendulum_simulation.m');
```

### CI/CD Pipeline

The project includes GitHub Actions workflows that automatically:
1. Run all unit tests on push and pull requests
2. Check code quality using MATLAB's `checkcode`
3. Generate test results and code coverage reports
4. Build the project

## Development Guidelines

### Code Style
- Use descriptive variable names
- Add comprehensive help documentation for all functions
- Include input validation
- Follow MATLAB naming conventions

### Adding New Simulations
1. Create new function in appropriate `src/` subdirectory
2. Add corresponding unit tests in `tests/unit/`
3. Update documentation
4. Ensure CI/CD passes

### Testing Requirements
- All new functions must have unit tests
- Aim for >80% code coverage
- Tests should cover edge cases and error conditions

## GitHub Actions Setup

The CI/CD pipeline uses [matlab-actions](https://github.com/matlab-actions) for:
- Setting up MATLAB environment
- Running tests and generating reports
- Code quality checks

### Workflow Features
- Automatic testing on push to main/develop branches
- Pull request validation
- Test result artifacts
- Code coverage reports

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MATLAB Actions for GitHub Actions integration
- MATLAB documentation and examples