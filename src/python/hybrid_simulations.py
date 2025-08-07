"""
Hybrid Physics Simulations Integration Module
============================================

This module provides Python interfaces to MATLAB physics simulations,
enabling hybrid computational workflows that combine Python's data
processing capabilities with MATLAB's numerical simulation strengths.

Features:
- Python wrappers for all MATLAB physics simulations
- Data conversion and validation
- Performance optimization
- Visualization integration
- Batch processing capabilities

Author: Murray Kopit
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from pathlib import Path
from dataclasses import dataclass, field
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Import our modules
from matlab_engine_wrapper import MATLABEngineWrapper, MATLABSessionManager, MATLABConfig
from config_manager import get_current_config

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result container for physics simulations."""
    simulation_type: str
    parameters: Dict[str, Any]
    time: np.ndarray
    data: Dict[str, np.ndarray]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Configuration for physics simulations."""
    physics_path: Path
    enable_visualization: bool = True
    save_results: bool = False
    results_path: Optional[Path] = None
    validation_tolerance: float = 1e-12
    max_execution_time: float = 300.0  # 5 minutes


class PhysicsSimulationInterface:
    """
    Base interface for physics simulations with common functionality.
    """
    
    def __init__(self, engine: MATLABEngineWrapper, config: SimulationConfig):
        """
        Initialize physics simulation interface.
        
        Args:
            engine: Active MATLAB engine wrapper
            config: Simulation configuration
        """
        self.engine = engine
        self.config = config
        self._setup_matlab_environment()
    
    def _setup_matlab_environment(self):
        """Setup MATLAB environment for physics simulations."""
        try:
            # Add physics simulation path
            if self.config.physics_path.exists():
                self.engine.evaluate(f"addpath('{self.config.physics_path}')", convert_types=False)
                logger.info(f"Added physics path: {self.config.physics_path}")
            else:
                logger.warning(f"Physics path not found: {self.config.physics_path}")
            
            # Configure visualization
            if not self.config.enable_visualization:
                self.engine.evaluate("set(0, 'DefaultFigureVisible', 'off')", convert_types=False)
            
            # Set random seed for reproducibility
            self.engine.evaluate("rng(42)", convert_types=False)
            
        except Exception as e:
            logger.error(f"Failed to setup MATLAB environment: {e}")
            raise
    
    def _validate_parameters(self, params: Dict[str, Any], required_params: List[str]) -> bool:
        """Validate simulation parameters."""
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            logger.error(f"Missing required parameters: {missing_params}")
            return False
        
        return True
    
    def _convert_matlab_result(self, result: Any) -> np.ndarray:
        """Convert MATLAB result to numpy array."""
        if hasattr(result, '_data'):
            # Handle MATLAB arrays
            data = result._data
            if hasattr(result, '_size'):
                return np.array(data).reshape(result._size)
            else:
                return np.array(data)
        elif isinstance(result, (list, tuple)):
            return np.array(result)
        else:
            return np.array([result])
    
    def _save_result(self, result: SimulationResult):
        """Save simulation result to file."""
        if not self.config.save_results or not self.config.results_path:
            return
        
        self.config.results_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.simulation_type}_{timestamp}.json"
        filepath = self.config.results_path / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in result.data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        result_dict = {
            'simulation_type': result.simulation_type,
            'parameters': result.parameters,
            'time': result.time.tolist() if isinstance(result.time, np.ndarray) else result.time,
            'data': serializable_data,
            'execution_time': result.execution_time,
            'success': result.success,
            'error_message': result.error_message,
            'metadata': result.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Simulation result saved to {filepath}")


class PendulumSimulation(PhysicsSimulationInterface):
    """
    Python interface for MATLAB pendulum simulation.
    
    Wraps the MATLAB pendulum_simulation function with enhanced Python integration.
    """
    
    def simulate(self, length: float, initial_angle: float, initial_velocity: float,
                 time_span: Tuple[float, float], damping: float = 0.0, 
                 gravity: float = 9.81, **kwargs) -> SimulationResult:
        """
        Run pendulum simulation.
        
        Args:
            length: Pendulum length (m)
            initial_angle: Initial angle (rad)
            initial_velocity: Initial angular velocity (rad/s)
            time_span: Time span (start, end) in seconds
            damping: Damping coefficient
            gravity: Gravitational acceleration (m/s²)
            **kwargs: Additional parameters
            
        Returns:
            SimulationResult object
        """
        start_time = time.time()
        
        parameters = {
            'length': length,
            'initial_angle': initial_angle,
            'initial_velocity': initial_velocity,
            'time_span': time_span,
            'damping': damping,
            'gravity': gravity,
            **kwargs
        }
        
        try:
            # Validate parameters
            if not self._validate_parameters(parameters, ['length', 'initial_angle', 'initial_velocity', 'time_span']):
                raise ValueError("Invalid parameters")
            
            # Prepare MATLAB function call
            if damping > 0:
                matlab_call = (f"[t, theta, omega] = pendulum_simulation({length}, {initial_angle}, "
                             f"{initial_velocity}, [{time_span[0]}, {time_span[1]}], "
                             f"'damping', {damping}, 'gravity', {gravity})")
            else:
                matlab_call = (f"[t, theta, omega] = pendulum_simulation({length}, {initial_angle}, "
                             f"{initial_velocity}, [{time_span[0]}, {time_span[1]}], "
                             f"'gravity', {gravity})")
            
            # Execute simulation
            self.engine.evaluate(matlab_call, convert_types=False)
            
            # Retrieve results
            t = self._convert_matlab_result(self.engine.get_workspace_variable('t'))
            theta = self._convert_matlab_result(self.engine.get_workspace_variable('theta'))
            omega = self._convert_matlab_result(self.engine.get_workspace_variable('omega'))
            
            # Calculate derived quantities
            energy = self._calculate_pendulum_energy(theta, omega, length, gravity)
            period = self._estimate_period(t, theta)
            
            execution_time = time.time() - start_time
            
            result = SimulationResult(
                simulation_type='pendulum',
                parameters=parameters,
                time=t,
                data={
                    'theta': theta,
                    'omega': omega,
                    'energy': energy,
                    'position_x': length * np.sin(theta),
                    'position_y': -length * np.cos(theta)
                },
                execution_time=execution_time,
                success=True,
                metadata={
                    'estimated_period': period,
                    'max_angle': np.max(np.abs(theta)),
                    'final_energy': energy[-1] if len(energy) > 0 else 0
                }
            )
            
            # Clean up MATLAB workspace
            self.engine.evaluate("clear t theta omega", convert_types=False)
            
            # Save result if configured
            self._save_result(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pendulum simulation failed: {e}")
            
            return SimulationResult(
                simulation_type='pendulum',
                parameters=parameters,
                time=np.array([]),
                data={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_pendulum_energy(self, theta: np.ndarray, omega: np.ndarray, 
                                 length: float, gravity: float) -> np.ndarray:
        """Calculate total energy of pendulum system."""
        kinetic_energy = 0.5 * length**2 * omega**2
        potential_energy = gravity * length * (1 - np.cos(theta))
        return kinetic_energy + potential_energy
    
    def _estimate_period(self, time: np.ndarray, theta: np.ndarray) -> float:
        """Estimate pendulum period from simulation data."""
        try:
            # Find zero crossings
            zero_crossings = []
            for i in range(1, len(theta)):
                if theta[i-1] * theta[i] <= 0:  # Sign change
                    zero_crossings.append(time[i])
            
            if len(zero_crossings) >= 4:
                # Calculate period from multiple crossings
                periods = []
                for i in range(2, len(zero_crossings), 2):
                    period = zero_crossings[i] - zero_crossings[i-2]
                    periods.append(period)
                return np.mean(periods)
            else:
                # Fallback to theoretical period for small angles
                return 2 * np.pi * np.sqrt(length / 9.81)
                
        except Exception as e:
            logger.warning(f"Failed to estimate period: {e}")
            return 0.0
    
    def batch_simulate(self, parameter_sets: List[Dict[str, Any]]) -> List[SimulationResult]:
        """
        Run multiple pendulum simulations with different parameters.
        
        Args:
            parameter_sets: List of parameter dictionaries
            
        Returns:
            List of SimulationResult objects
        """
        results = []
        
        for i, params in enumerate(parameter_sets):
            logger.info(f"Running pendulum simulation {i+1}/{len(parameter_sets)}")
            result = self.simulate(**params)
            results.append(result)
        
        return results


class ParticleDynamicsSimulation(PhysicsSimulationInterface):
    """
    Python interface for MATLAB particle dynamics simulation.
    """
    
    def simulate(self, mass: float, force_function: str, initial_position: List[float],
                 initial_velocity: List[float], time_span: Tuple[float, float],
                 solver: str = 'ode45', tolerance: float = 1e-6, **kwargs) -> SimulationResult:
        """
        Run particle dynamics simulation.
        
        Args:
            mass: Particle mass (kg)
            force_function: MATLAB force function string
            initial_position: Initial position [x, y, z] (m)
            initial_velocity: Initial velocity [vx, vy, vz] (m/s)
            time_span: Time span (start, end) in seconds
            solver: ODE solver ('ode45', 'ode23', 'ode113')
            tolerance: Numerical tolerance
            **kwargs: Additional parameters
            
        Returns:
            SimulationResult object
        """
        start_time = time.time()
        
        parameters = {
            'mass': mass,
            'force_function': force_function,
            'initial_position': initial_position,
            'initial_velocity': initial_velocity,
            'time_span': time_span,
            'solver': solver,
            'tolerance': tolerance,
            **kwargs
        }
        
        try:
            # Validate parameters
            required_params = ['mass', 'force_function', 'initial_position', 'initial_velocity', 'time_span']
            if not self._validate_parameters(parameters, required_params):
                raise ValueError("Invalid parameters")
            
            # Define force function in MATLAB
            self.engine.evaluate(f"force_func = {force_function}", convert_types=False)
            
            # Set initial conditions
            self.engine.set_workspace_variable('x0', np.array(initial_position))
            self.engine.set_workspace_variable('v0', np.array(initial_velocity))
            
            # Prepare MATLAB function call
            matlab_call = (f"[t, position, velocity] = particle_dynamics({mass}, force_func, "
                         f"x0, v0, [{time_span[0]}, {time_span[1]}], "
                         f"'method', '{solver}', 'tolerance', {tolerance})")
            
            # Execute simulation
            self.engine.evaluate(matlab_call, convert_types=False)
            
            # Retrieve results
            t = self._convert_matlab_result(self.engine.get_workspace_variable('t'))
            position = self._convert_matlab_result(self.engine.get_workspace_variable('position'))
            velocity = self._convert_matlab_result(self.engine.get_workspace_variable('velocity'))
            
            # Calculate derived quantities
            speed = np.linalg.norm(velocity, axis=1)
            acceleration = self._calculate_acceleration(t, velocity)
            kinetic_energy = 0.5 * mass * speed**2
            
            execution_time = time.time() - start_time
            
            result = SimulationResult(
                simulation_type='particle_dynamics',
                parameters=parameters,
                time=t,
                data={
                    'position': position,
                    'velocity': velocity,
                    'speed': speed,
                    'acceleration': acceleration,
                    'kinetic_energy': kinetic_energy
                },
                execution_time=execution_time,
                success=True,
                metadata={
                    'max_speed': np.max(speed),
                    'final_position': position[-1] if len(position) > 0 else [0, 0, 0],
                    'total_distance': self._calculate_total_distance(position)
                }
            )
            
            # Clean up MATLAB workspace
            self.engine.evaluate("clear t position velocity force_func x0 v0", convert_types=False)
            
            self._save_result(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Particle dynamics simulation failed: {e}")
            
            return SimulationResult(
                simulation_type='particle_dynamics',
                parameters=parameters,
                time=np.array([]),
                data={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_acceleration(self, time: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate acceleration from velocity using numerical differentiation."""
        if len(time) < 2:
            return np.zeros_like(velocity)
        
        acceleration = np.zeros_like(velocity)
        
        # Forward difference for first point
        acceleration[0] = (velocity[1] - velocity[0]) / (time[1] - time[0])
        
        # Central difference for interior points
        for i in range(1, len(time) - 1):
            acceleration[i] = (velocity[i+1] - velocity[i-1]) / (time[i+1] - time[i-1])
        
        # Backward difference for last point
        acceleration[-1] = (velocity[-1] - velocity[-2]) / (time[-1] - time[-2])
        
        return acceleration
    
    def _calculate_total_distance(self, position: np.ndarray) -> float:
        """Calculate total distance traveled."""
        if len(position) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(position, axis=0), axis=1)
        return np.sum(distances)
    
    def simulate_projectile(self, mass: float, initial_position: List[float],
                          initial_velocity: List[float], gravity: float = 9.81,
                          air_resistance: bool = False, drag_coefficient: float = 0.1,
                          **kwargs) -> SimulationResult:
        """
        Convenient method for projectile motion simulation.
        
        Args:
            mass: Projectile mass (kg)
            initial_position: Initial position [x, y, z] (m)
            initial_velocity: Initial velocity [vx, vy, vz] (m/s)
            gravity: Gravitational acceleration (m/s²)
            air_resistance: Whether to include air resistance
            drag_coefficient: Drag coefficient for air resistance
            **kwargs: Additional parameters
            
        Returns:
            SimulationResult object
        """
        if air_resistance:
            force_function = (f"@(t, x, v) [0; 0; -{gravity} * {mass}] - "
                            f"{drag_coefficient} * norm(v) * v")
        else:
            force_function = f"@(t, x, v) [0; 0; -{gravity} * {mass}]"
        
        # Calculate flight time for automatic time span
        if 'time_span' not in kwargs:
            # Estimate flight time assuming projectile motion
            vy0 = initial_velocity[2] if len(initial_velocity) > 2 else 0
            y0 = initial_position[2] if len(initial_position) > 2 else 0
            
            if vy0 > 0:
                flight_time = 2 * vy0 / gravity + np.sqrt(2 * y0 / gravity)
            else:
                flight_time = np.sqrt(2 * y0 / gravity)
            
            kwargs['time_span'] = (0, max(flight_time, 1.0))
        
        return self.simulate(
            mass=mass,
            force_function=force_function,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            **kwargs
        )


class WaveEquationSolver(PhysicsSimulationInterface):
    """
    Python interface for MATLAB wave equation solver.
    """
    
    def solve(self, domain_length: float, time_duration: float, wave_speed: float,
              initial_displacement: str, initial_velocity: str, 
              nx: int = 100, nt: int = 200, boundary: str = 'dirichlet',
              **kwargs) -> SimulationResult:
        """
        Solve 1D wave equation.
        
        Args:
            domain_length: Spatial domain length
            time_duration: Time duration
            wave_speed: Wave speed
            initial_displacement: MATLAB function string for initial displacement
            initial_velocity: MATLAB function string for initial velocity
            nx: Number of spatial points
            nt: Number of time points
            boundary: Boundary conditions ('dirichlet' or 'neumann')
            **kwargs: Additional parameters
            
        Returns:
            SimulationResult object
        """
        start_time = time.time()
        
        parameters = {
            'domain_length': domain_length,
            'time_duration': time_duration,
            'wave_speed': wave_speed,
            'initial_displacement': initial_displacement,
            'initial_velocity': initial_velocity,
            'nx': nx,
            'nt': nt,
            'boundary': boundary,
            **kwargs
        }
        
        try:
            # Validate parameters
            required_params = ['domain_length', 'time_duration', 'wave_speed', 
                             'initial_displacement', 'initial_velocity']
            if not self._validate_parameters(parameters, required_params):
                raise ValueError("Invalid parameters")
            
            # Define initial condition functions in MATLAB
            self.engine.evaluate(f"initial_u = {initial_displacement}", convert_types=False)
            self.engine.evaluate(f"initial_ut = {initial_velocity}", convert_types=False)
            
            # Prepare MATLAB function call
            matlab_call = (f"[u, x, t] = wave_equation_solver({domain_length}, {time_duration}, "
                         f"{wave_speed}, initial_u, initial_ut, 'nx', {nx}, 'nt', {nt}, "
                         f"'boundary', '{boundary}')")
            
            # Execute simulation
            self.engine.evaluate(matlab_call, convert_types=False)
            
            # Retrieve results
            u = self._convert_matlab_result(self.engine.get_workspace_variable('u'))
            x = self._convert_matlab_result(self.engine.get_workspace_variable('x'))
            t = self._convert_matlab_result(self.engine.get_workspace_variable('t'))
            
            # Calculate wave properties
            wave_energy = self._calculate_wave_energy(u, x, t, wave_speed)
            max_amplitude = np.max(np.abs(u))
            
            execution_time = time.time() - start_time
            
            result = SimulationResult(
                simulation_type='wave_equation',
                parameters=parameters,
                time=t,
                data={
                    'u': u,
                    'x': x,
                    'wave_energy': wave_energy,
                    'amplitude_envelope': np.max(np.abs(u), axis=0)
                },
                execution_time=execution_time,
                success=True,
                metadata={
                    'max_amplitude': max_amplitude,
                    'spatial_resolution': domain_length / (nx - 1),
                    'temporal_resolution': time_duration / (nt - 1),
                    'cfl_number': wave_speed * (time_duration / (nt - 1)) / (domain_length / (nx - 1))
                }
            )
            
            # Clean up MATLAB workspace
            self.engine.evaluate("clear u x t initial_u initial_ut", convert_types=False)
            
            self._save_result(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Wave equation solver failed: {e}")
            
            return SimulationResult(
                simulation_type='wave_equation',
                parameters=parameters,
                time=np.array([]),
                data={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_wave_energy(self, u: np.ndarray, x: np.ndarray, t: np.ndarray, 
                             wave_speed: float) -> np.ndarray:
        """Calculate total wave energy over time."""
        if u.ndim != 2 or len(x) < 2 or len(t) < 2:
            return np.array([])
        
        dx = x[1] - x[0]
        energy = np.zeros(len(t))
        
        for i in range(len(t)):
            # Spatial derivative (kinetic energy term)
            if i > 0:
                u_t = (u[:, i] - u[:, i-1]) / (t[i] - t[i-1])
            else:
                u_t = np.zeros_like(u[:, i])
            
            # Spatial derivative (potential energy term)
            u_x = np.gradient(u[:, i], dx)
            
            # Total energy density
            energy_density = 0.5 * (u_t**2 + wave_speed**2 * u_x**2)
            energy[i] = np.trapz(energy_density, x)
        
        return energy
    
    def solve_gaussian_pulse(self, domain_length: float = 10.0, time_duration: float = 5.0,
                           wave_speed: float = 1.0, pulse_center: float = None,
                           pulse_width: float = 1.0, **kwargs) -> SimulationResult:
        """
        Convenient method for Gaussian pulse wave simulation.
        
        Args:
            domain_length: Spatial domain length
            time_duration: Time duration
            wave_speed: Wave speed
            pulse_center: Center of Gaussian pulse (default: domain_length/4)
            pulse_width: Width of Gaussian pulse
            **kwargs: Additional parameters
            
        Returns:
            SimulationResult object
        """
        if pulse_center is None:
            pulse_center = domain_length / 4
        
        initial_displacement = f"@(x) exp(-((x - {pulse_center})/{pulse_width}).^2)"
        initial_velocity = "@(x) zeros(size(x))"
        
        return self.solve(
            domain_length=domain_length,
            time_duration=time_duration,
            wave_speed=wave_speed,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
            **kwargs
        )


class HybridSimulationManager:
    """
    Manager for hybrid Python-MATLAB physics simulations.
    
    Provides high-level interface for running and analyzing physics simulations.
    """
    
    def __init__(self, session_manager: Optional[MATLABSessionManager] = None,
                 physics_path: Optional[Path] = None):
        """
        Initialize hybrid simulation manager.
        
        Args:
            session_manager: MATLAB session manager (creates one if None)
            physics_path: Path to physics simulation files
        """
        if session_manager is None:
            config = MATLABConfig(
                startup_options=['-nojvm', '-nodisplay'],
                headless_mode=True,
                performance_monitoring=True,
                max_sessions=3
            )
            self.session_manager = MATLABSessionManager(config=config)
            self._should_close_manager = True
        else:
            self.session_manager = session_manager
            self._should_close_manager = False
        
        if physics_path is None:
            physics_path = Path(__file__).parent.parent / "physics"
        
        self.config = SimulationConfig(
            physics_path=physics_path,
            enable_visualization=False,  # Disable for batch processing
            save_results=True,
            results_path=Path(__file__).parent / "simulation_results"
        )
        
        # Initialize simulation interfaces
        self.engine = self.session_manager.get_or_create_session("hybrid_simulations")
        self.pendulum = PendulumSimulation(self.engine, self.config)
        self.particle = ParticleDynamicsSimulation(self.engine, self.config)
        self.wave = WaveEquationSolver(self.engine, self.config)
    
    def run_pendulum_parameter_study(self, length_range: Tuple[float, float],
                                   angle_range: Tuple[float, float],
                                   n_samples: int = 10) -> List[SimulationResult]:
        """
        Run parameter study for pendulum simulations.
        
        Args:
            length_range: Range of pendulum lengths (min, max)
            angle_range: Range of initial angles (min, max)
            n_samples: Number of samples for each parameter
            
        Returns:
            List of simulation results
        """
        logger.info(f"Running pendulum parameter study with {n_samples} samples")
        
        lengths = np.linspace(length_range[0], length_range[1], n_samples)
        angles = np.linspace(angle_range[0], angle_range[1], n_samples)
        
        parameter_sets = []
        for L in lengths:
            for theta0 in angles:
                parameter_sets.append({
                    'length': L,
                    'initial_angle': theta0,
                    'initial_velocity': 0.0,
                    'time_span': (0, 4 * np.pi * np.sqrt(L / 9.81))  # ~4 periods
                })
        
        return self.pendulum.batch_simulate(parameter_sets)
    
    def run_projectile_trajectories(self, launch_angles: List[float],
                                  initial_speed: float = 20.0,
                                  height: float = 0.0) -> List[SimulationResult]:
        """
        Run projectile trajectory simulations for different launch angles.
        
        Args:
            launch_angles: List of launch angles in degrees
            initial_speed: Initial speed (m/s)
            height: Launch height (m)
            
        Returns:
            List of simulation results
        """
        logger.info(f"Running projectile trajectories for {len(launch_angles)} angles")
        
        results = []
        for angle_deg in launch_angles:
            angle_rad = np.radians(angle_deg)
            vx0 = initial_speed * np.cos(angle_rad)
            vz0 = initial_speed * np.sin(angle_rad)  # z is vertical
            
            result = self.particle.simulate_projectile(
                mass=1.0,
                initial_position=[0, 0, height],
                initial_velocity=[vx0, 0, vz0],
                gravity=9.81,
                air_resistance=False
            )
            results.append(result)
        
        return results
    
    def run_wave_interference_study(self, frequencies: List[float],
                                  amplitudes: List[float]) -> List[SimulationResult]:
        """
        Run wave interference study with multiple frequencies and amplitudes.
        
        Args:
            frequencies: List of wave frequencies
            amplitudes: List of wave amplitudes
            
        Returns:
            List of simulation results
        """
        logger.info(f"Running wave interference study")
        
        results = []
        for freq in frequencies:
            for amp in amplitudes:
                # Create sinusoidal initial condition
                initial_disp = f"@(x) {amp} * sin(2*pi*{freq}*x/10)"
                initial_vel = "@(x) zeros(size(x))"
                
                result = self.wave.solve(
                    domain_length=10.0,
                    time_duration=5.0,
                    wave_speed=2.0,
                    initial_displacement=initial_disp,
                    initial_velocity=initial_vel,
                    nx=200,
                    nt=500
                )
                results.append(result)
        
        return results
    
    def validate_simulations(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Validate simulation results for physical consistency.
        
        Args:
            results: List of simulation results to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_simulations': len(results),
            'successful_simulations': 0,
            'failed_simulations': 0,
            'validation_errors': [],
            'physics_violations': []
        }
        
        for result in results:
            if result.success:
                validation_report['successful_simulations'] += 1
                
                # Perform physics-specific validation
                if result.simulation_type == 'pendulum':
                    self._validate_pendulum_physics(result, validation_report)
                elif result.simulation_type == 'particle_dynamics':
                    self._validate_particle_physics(result, validation_report)
                elif result.simulation_type == 'wave_equation':
                    self._validate_wave_physics(result, validation_report)
            else:
                validation_report['failed_simulations'] += 1
                validation_report['validation_errors'].append({
                    'simulation': result.simulation_type,
                    'error': result.error_message
                })
        
        return validation_report
    
    def _validate_pendulum_physics(self, result: SimulationResult, report: Dict[str, Any]):
        """Validate pendulum physics."""
        try:
            # Energy conservation check (for undamped pendulum)
            if result.parameters.get('damping', 0) == 0:
                energy = result.data.get('energy', np.array([]))
                if len(energy) > 1:
                    energy_variation = (np.max(energy) - np.min(energy)) / np.mean(energy)
                    if energy_variation > 0.01:  # 1% tolerance
                        report['physics_violations'].append({
                            'simulation': 'pendulum',
                            'violation': 'energy_conservation',
                            'variation': energy_variation
                        })
        except Exception as e:
            logger.warning(f"Pendulum validation error: {e}")
    
    def _validate_particle_physics(self, result: SimulationResult, report: Dict[str, Any]):
        """Validate particle physics."""
        try:
            # Check if final position is reasonable for projectile motion
            if 'position' in result.data:
                position = result.data['position']
                if len(position) > 0:
                    final_height = position[-1, 2] if position.shape[1] > 2 else 0
                    if final_height < -1.0:  # Below ground level
                        report['physics_violations'].append({
                            'simulation': 'particle_dynamics',
                            'violation': 'negative_height',
                            'final_height': final_height
                        })
        except Exception as e:
            logger.warning(f"Particle validation error: {e}")
    
    def _validate_wave_physics(self, result: SimulationResult, report: Dict[str, Any]):
        """Validate wave physics."""
        try:
            # Check wave energy conservation
            if 'wave_energy' in result.data:
                energy = result.data['wave_energy']
                if len(energy) > 1:
                    energy_variation = (np.max(energy) - np.min(energy)) / np.mean(energy)
                    if energy_variation > 0.05:  # 5% tolerance for numerical errors
                        report['physics_violations'].append({
                            'simulation': 'wave_equation',
                            'violation': 'energy_conservation',
                            'variation': energy_variation
                        })
        except Exception as e:
            logger.warning(f"Wave validation error: {e}")
    
    def generate_simulation_summary(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Generate comprehensive simulation summary."""
        summary = {
            'total_simulations': len(results),
            'simulation_types': {},
            'execution_times': {
                'total': sum(r.execution_time for r in results),
                'average': np.mean([r.execution_time for r in results]),
                'min': min(r.execution_time for r in results) if results else 0,
                'max': max(r.execution_time for r in results) if results else 0
            },
            'success_rate': sum(r.success for r in results) / len(results) * 100 if results else 0,
            'physics_summary': {}
        }
        
        # Group by simulation type
        for result in results:
            sim_type = result.simulation_type
            if sim_type not in summary['simulation_types']:
                summary['simulation_types'][sim_type] = {
                    'count': 0,
                    'successful': 0,
                    'average_time': 0
                }
            
            summary['simulation_types'][sim_type]['count'] += 1
            if result.success:
                summary['simulation_types'][sim_type]['successful'] += 1
        
        # Calculate success rates and average times
        for sim_type, data in summary['simulation_types'].items():
            type_results = [r for r in results if r.simulation_type == sim_type]
            data['success_rate'] = data['successful'] / data['count'] * 100
            data['average_time'] = np.mean([r.execution_time for r in type_results])
        
        return summary
    
    def close(self):
        """Clean up resources."""
        if self._should_close_manager:
            self.session_manager.close_all_sessions()


# Example usage and demonstration
def run_hybrid_simulation_demo():
    """Demonstrate hybrid simulation capabilities."""
    print("Hybrid Physics Simulations Demo")
    print("=" * 50)
    
    # Create simulation manager
    manager = HybridSimulationManager()
    
    try:
        # 1. Pendulum parameter study
        print("\n1. Running pendulum parameter study...")
        pendulum_results = manager.run_pendulum_parameter_study(
            length_range=(0.5, 2.0),
            angle_range=(0.1, 1.0),
            n_samples=3
        )
        print(f"Completed {len(pendulum_results)} pendulum simulations")
        
        # 2. Projectile trajectories
        print("\n2. Running projectile trajectory analysis...")
        trajectory_results = manager.run_projectile_trajectories(
            launch_angles=[15, 30, 45, 60, 75],
            initial_speed=25.0,
            height=1.0
        )
        print(f"Completed {len(trajectory_results)} trajectory simulations")
        
        # 3. Wave interference study
        print("\n3. Running wave interference study...")
        wave_results = manager.run_wave_interference_study(
            frequencies=[1, 2],
            amplitudes=[1, 2]
        )
        print(f"Completed {len(wave_results)} wave simulations")
        
        # 4. Validation
        print("\n4. Validating simulation results...")
        all_results = pendulum_results + trajectory_results + wave_results
        validation_report = manager.validate_simulations(all_results)
        print(f"Validation: {validation_report['successful_simulations']}/{validation_report['total_simulations']} successful")
        
        # 5. Summary
        print("\n5. Generating simulation summary...")
        summary = manager.generate_simulation_summary(all_results)
        print(f"Overall success rate: {summary['success_rate']:.1f}%")
        print(f"Total execution time: {summary['execution_times']['total']:.2f}s")
        
        return all_results, validation_report, summary
        
    finally:
        manager.close()


if __name__ == "__main__":
    # Run demonstration
    results, validation, summary = run_hybrid_simulation_demo()
    
    print("\nHybrid simulation demonstration completed!")
    print(f"Results: {len(results)} simulations")
    print(f"Validation: {validation['successful_simulations']} successful")
    print(f"Summary: {summary['success_rate']:.1f}% overall success rate")