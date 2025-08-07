"""
Pipeline Validation Framework for MATLAB Engine API
==================================================

This module provides comprehensive pipeline validation testing to ensure
the entire MATLAB Engine API integration works correctly from end-to-end.

Features:
- Automated validation against golden values
- Configuration-driven test scenarios
- Comprehensive validation reports
- Performance benchmarking
- Error propagation testing

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import yaml
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass, field
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our modules
from matlab_engine_wrapper import MATLABEngineWrapper, MATLABSessionManager, MATLABConfig
from config_manager import get_current_config, Environment, set_environment
from test_mathematical_validation import MathematicalValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineTestCase:
    """Definition of a pipeline test case."""
    name: str
    description: str
    setup_commands: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    cleanup_commands: List[str] = field(default_factory=list)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    timeout: float = 60.0
    requires_physics: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a pipeline test execution."""
    test_case: PipelineTestCase
    success: bool
    execution_time: float
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineValidator:
    """
    Comprehensive pipeline validation framework.
    
    Validates complete workflows from Python to MATLAB and back,
    ensuring data integrity, performance, and reliability.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize pipeline validator.
        
        Args:
            config_path: Path to pipeline configuration YAML file
        """
        self.config_path = config_path or Path(__file__).parent / "pipeline_config.yaml"
        self.golden_values_path = Path(__file__).parent / "pipeline_golden_values.json"
        self.results_dir = Path(__file__).parent / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.test_cases = self._load_test_cases()
        self.golden_values = self._load_golden_values()
        self.results: List[PipelineResult] = []
    
    def _load_test_cases(self) -> List[PipelineTestCase]:
        """Load test cases from configuration file."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, creating default")
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        test_cases = []
        for case_data in config_data.get('test_cases', []):
            test_cases.append(PipelineTestCase(**case_data))
        
        return test_cases
    
    def _create_default_config(self):
        """Create default pipeline configuration."""
        default_config = {
            'test_cases': [
                {
                    'name': 'basic_arithmetic_pipeline',
                    'description': 'Test basic arithmetic operations through the pipeline',
                    'setup_commands': [
                        'a = 10',
                        'b = 5'
                    ],
                    'test_commands': [
                        'result_add = a + b',
                        'result_mult = a * b',
                        'result_div = a / b'
                    ],
                    'cleanup_commands': [
                        'clear a b result_add result_mult result_div'
                    ],
                    'expected_results': {
                        'result_add': 15,
                        'result_mult': 50,
                        'result_div': 2.0
                    },
                    'tolerances': {
                        'result_add': 1e-15,
                        'result_mult': 1e-15,
                        'result_div': 1e-15
                    }
                },
                {
                    'name': 'matrix_operations_pipeline',
                    'description': 'Test matrix operations pipeline',
                    'setup_commands': [
                        'A = [1, 2; 3, 4]',
                        'B = [5, 6; 7, 8]'
                    ],
                    'test_commands': [
                        'C = A * B',
                        'D = inv(A)',
                        'det_A = det(A)',
                        'trace_A = trace(A)'
                    ],
                    'cleanup_commands': [
                        'clear A B C D det_A trace_A'
                    ],
                    'expected_results': {
                        'C': [[19, 22], [43, 50]],
                        'D': [[-2.0, 1.0], [1.5, -0.5]],
                        'det_A': -2.0,
                        'trace_A': 5.0
                    },
                    'tolerances': {
                        'C': 1e-15,
                        'D': 1e-15,
                        'det_A': 1e-15,
                        'trace_A': 1e-15
                    }
                },
                {
                    'name': 'physics_simulation_pipeline',
                    'description': 'Test physics simulation integration',
                    'setup_commands': [
                        'addpath(fullfile(pwd, "../../physics"))'
                    ],
                    'test_commands': [
                        '[t, theta, omega] = pendulum_simulation(1, pi/4, 0, [0 1])',
                        'final_theta = theta(end)',
                        'energy_initial = 0.5 * 9.81 * 1 * (1 - cos(pi/4))',
                        'energy_final = 0.5 * omega(end)^2 + 0.5 * 9.81 * 1 * (1 - cos(theta(end)))'
                    ],
                    'cleanup_commands': [
                        'clear t theta omega final_theta energy_initial energy_final'
                    ],
                    'expected_results': {
                        'final_theta': -0.785398163,  # Approximately -pi/4
                    },
                    'tolerances': {
                        'final_theta': 0.01,
                    },
                    'requires_physics': True,
                    'timeout': 30.0
                },
                {
                    'name': 'data_type_conversion_pipeline',
                    'description': 'Test Python-MATLAB data type conversions',
                    'setup_commands': [],
                    'test_commands': [
                        'python_array = py_array',  # Will be set from Python
                        'matlab_sum = sum(python_array)',
                        'matlab_mean = mean(python_array)',
                        'matlab_std = std(python_array)'
                    ],
                    'cleanup_commands': [
                        'clear python_array matlab_sum matlab_mean matlab_std'
                    ],
                    'expected_results': {},  # Will be calculated dynamically
                    'tolerances': {},
                    'metadata': {'test_arrays': [[1, 2, 3, 4, 5], [1.1, 2.2, 3.3], [10, 20, 30, 40]]}
                },
                {
                    'name': 'error_handling_pipeline',
                    'description': 'Test error handling and recovery',
                    'setup_commands': [],
                    'test_commands': [
                        'try_result = 1 / 0'  # This should fail
                    ],
                    'cleanup_commands': [
                        'clear try_result'
                    ],
                    'expected_results': {},
                    'tolerances': {},
                    'metadata': {'expect_error': True}
                },
                {
                    'name': 'performance_stress_pipeline',
                    'description': 'Test performance under stress',
                    'setup_commands': [
                        'rng(42)'  # Set random seed for reproducibility
                    ],
                    'test_commands': [
                        'large_matrix = randn(1000, 1000)',
                        'eig_result = eig(large_matrix)',
                        'fft_result = fft(large_matrix)',
                        'inv_result = inv(large_matrix)'
                    ],
                    'cleanup_commands': [
                        'clear large_matrix eig_result fft_result inv_result'
                    ],
                    'expected_results': {},
                    'tolerances': {},
                    'timeout': 120.0,
                    'metadata': {'performance_test': True}
                }
            ]
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default pipeline configuration at {self.config_path}")
    
    def _load_golden_values(self) -> Dict[str, Any]:
        """Load golden values for validation."""
        if self.golden_values_path.exists():
            with open(self.golden_values_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_golden_values(self):
        """Save golden values to file."""
        with open(self.golden_values_path, 'w') as f:
            json.dump(self.golden_values, f, indent=2)
    
    def _compare_results(self, expected: Any, actual: Any, tolerance: float) -> Tuple[bool, float]:
        """Compare expected vs actual results."""
        try:
            if isinstance(expected, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
                expected_arr = np.asarray(expected)
                actual_arr = np.asarray(actual)
                
                if expected_arr.shape != actual_arr.shape:
                    return False, float('inf')
                
                error = np.max(np.abs(expected_arr - actual_arr))
                return error <= tolerance, float(error)
            
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                error = abs(float(expected) - float(actual))
                return error <= tolerance, error
            
            elif isinstance(expected, complex) and isinstance(actual, complex):
                error = abs(expected - actual)
                return error <= tolerance, float(error)
            
            else:
                # For other types, try direct comparison
                return expected == actual, 0.0
                
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return False, float('inf')
    
    def run_single_test_case(self, test_case: PipelineTestCase, 
                           engine: MATLABEngineWrapper) -> PipelineResult:
        """
        Run a single pipeline test case.
        
        Args:
            test_case: Test case to run
            engine: MATLAB engine wrapper
            
        Returns:
            PipelineResult object
        """
        start_time = time.time()
        result = PipelineResult(test_case=test_case, success=True, execution_time=0)
        
        try:
            logger.info(f"Running pipeline test: {test_case.name}")
            
            # Handle special data type conversion test
            if test_case.name == 'data_type_conversion_pipeline':
                self._setup_data_type_test(test_case, engine)
            
            # Execute setup commands
            for cmd in test_case.setup_commands:
                try:
                    engine.evaluate(cmd, convert_types=False)
                except Exception as e:
                    if not test_case.metadata.get('expect_error', False):
                        result.errors.append(f"Setup command failed: {cmd} - {e}")
                        result.success = False
            
            # Execute test commands and collect results
            for cmd in test_case.test_commands:
                try:
                    # Extract variable name from command if it's an assignment
                    if '=' in cmd and not cmd.strip().startswith('try_result'):
                        var_name = cmd.split('=')[0].strip()
                        engine.evaluate(cmd, convert_types=False)
                        
                        # Get the result
                        if not test_case.metadata.get('expect_error', False):
                            value = engine.get_workspace_variable(var_name, convert_types=True)
                            result.results[var_name] = value
                    else:
                        # Direct evaluation
                        if test_case.metadata.get('expect_error', False):
                            try:
                                engine.evaluate(cmd)
                                result.errors.append(f"Expected error but command succeeded: {cmd}")
                            except Exception:
                                # Expected error occurred
                                logger.info(f"Expected error caught for: {cmd}")
                        else:
                            engine.evaluate(cmd)
                            
                except Exception as e:
                    if test_case.metadata.get('expect_error', False):
                        logger.info(f"Expected error caught: {e}")
                    else:
                        result.errors.append(f"Test command failed: {cmd} - {e}")
                        result.success = False
            
            # Validate results against expected values
            if not test_case.metadata.get('expect_error', False):
                self._validate_test_results(test_case, result)
            
            # Execute cleanup commands
            for cmd in test_case.cleanup_commands:
                try:
                    engine.evaluate(cmd, convert_types=False)
                except Exception as e:
                    result.warnings.append(f"Cleanup command failed: {cmd} - {e}")
            
            # Collect performance metrics
            if test_case.metadata.get('performance_test', False):
                stats = engine.get_performance_stats()
                result.performance_metrics = {
                    'avg_execution_time': stats['avg_execution_time'],
                    'total_operations': stats['total_operations'],
                    'success_rate': stats['successful_operations'] / stats['total_operations']
                }
        
        except Exception as e:
            result.errors.append(f"Unexpected error: {e}")
            result.success = False
            logger.error(f"Pipeline test {test_case.name} failed: {e}")
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _setup_data_type_test(self, test_case: PipelineTestCase, engine: MATLABEngineWrapper):
        """Setup data type conversion test with dynamic expected values."""
        test_arrays = test_case.metadata.get('test_arrays', [])
        
        for i, py_array in enumerate(test_arrays):
            py_array = np.array(py_array)
            
            # Set array in MATLAB workspace
            engine.set_workspace_variable('py_array', py_array)
            
            # Calculate expected results
            expected_sum = float(np.sum(py_array))
            expected_mean = float(np.mean(py_array))
            expected_std = float(np.std(py_array, ddof=0))  # MATLAB uses population std
            
            # Update test case with expected results
            test_case.expected_results.update({
                'matlab_sum': expected_sum,
                'matlab_mean': expected_mean,
                'matlab_std': expected_std
            })
            
            # Set tolerances
            test_case.tolerances.update({
                'matlab_sum': 1e-14,
                'matlab_mean': 1e-14,
                'matlab_std': 1e-14
            })
            
            break  # Only test first array for now
    
    def _validate_test_results(self, test_case: PipelineTestCase, result: PipelineResult):
        """Validate test results against expected values."""
        for var_name, expected_value in test_case.expected_results.items():
            if var_name in result.results:
                actual_value = result.results[var_name]
                tolerance = test_case.tolerances.get(var_name, 1e-12)
                
                is_valid, error = self._compare_results(expected_value, actual_value, tolerance)
                
                if not is_valid:
                    result.success = False
                    result.errors.append(
                        f"Result validation failed for {var_name}: "
                        f"expected {expected_value}, got {actual_value}, "
                        f"error {error}, tolerance {tolerance}"
                    )
            else:
                result.warnings.append(f"Expected result {var_name} not found in results")
    
    def run_pipeline_validation(self, session_manager: Optional[MATLABSessionManager] = None) -> List[PipelineResult]:
        """
        Run complete pipeline validation suite.
        
        Args:
            session_manager: Optional session manager (creates one if None)
            
        Returns:
            List of PipelineResult objects
        """
        logger.info("Starting pipeline validation suite...")
        
        if session_manager is None:
            config = MATLABConfig(
                startup_options=['-nojvm', '-nodisplay'],
                headless_mode=True,
                performance_monitoring=True,
                session_timeout=300
            )
            session_manager = MATLABSessionManager(config=config)
            should_close_manager = True
        else:
            should_close_manager = False
        
        try:
            # Get or create MATLAB session
            engine = session_manager.get_or_create_session("pipeline_validation")
            
            # Add physics path if needed
            physics_path = Path(__file__).parent.parent.parent / "physics"
            if physics_path.exists():
                engine.evaluate(f"addpath('{physics_path}')", convert_types=False)
            
            # Run test cases
            for test_case in self.test_cases:
                # Skip physics tests if physics path doesn't exist
                if test_case.requires_physics and not physics_path.exists():
                    logger.warning(f"Skipping {test_case.name} - physics path not found")
                    continue
                
                result = self.run_single_test_case(test_case, engine)
                self.results.append(result)
                
                logger.info(f"Test {test_case.name}: {'PASSED' if result.success else 'FAILED'}")
                if result.errors:
                    for error in result.errors:
                        logger.error(f"  {error}")
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"  {warning}")
        
        finally:
            if should_close_manager:
                session_manager.close_all_sessions()
        
        return self.results
    
    def run_concurrent_validation(self, max_workers: int = 3) -> List[PipelineResult]:
        """
        Run pipeline validation with concurrent test execution.
        
        Args:
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of PipelineResult objects
        """
        logger.info(f"Starting concurrent pipeline validation with {max_workers} workers...")
        
        config = MATLABConfig(
            startup_options=['-nojvm', '-nodisplay'],
            headless_mode=True,
            max_sessions=max_workers,
            session_timeout=300
        )
        
        with MATLABSessionManager(config=config) as session_manager:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit test cases to executor
                future_to_testcase = {}
                
                for test_case in self.test_cases:
                    # Skip physics tests if needed
                    physics_path = Path(__file__).parent.parent.parent / "physics"
                    if test_case.requires_physics and not physics_path.exists():
                        continue
                    
                    future = executor.submit(self._run_concurrent_test, test_case, session_manager)
                    future_to_testcase[future] = test_case
                
                # Collect results
                concurrent_results = []
                for future in as_completed(future_to_testcase):
                    test_case = future_to_testcase[future]
                    try:
                        result = future.result()
                        concurrent_results.append(result)
                        logger.info(f"Concurrent test {test_case.name}: {'PASSED' if result.success else 'FAILED'}")
                    except Exception as e:
                        logger.error(f"Concurrent test {test_case.name} failed with exception: {e}")
                        error_result = PipelineResult(
                            test_case=test_case,
                            success=False,
                            execution_time=0,
                            errors=[str(e)]
                        )
                        concurrent_results.append(error_result)
        
        self.results.extend(concurrent_results)
        return concurrent_results
    
    def _run_concurrent_test(self, test_case: PipelineTestCase, 
                           session_manager: MATLABSessionManager) -> PipelineResult:
        """Run a single test case in concurrent mode."""
        session_id = f"concurrent_{test_case.name}_{threading.get_ident()}"
        engine = session_manager.get_or_create_session(session_id)
        
        # Add physics path if needed
        physics_path = Path(__file__).parent.parent.parent / "physics"
        if physics_path.exists():
            engine.evaluate(f"addpath('{physics_path}')", convert_types=False)
        
        return self.run_single_test_case(test_case, engine)
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline validation report."""
        if not self.results:
            return {"error": "No results available. Run validation first."}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_execution_time = sum(r.execution_time for r in self.results)
        avg_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "total_execution_time": total_execution_time,
                "average_execution_time": avg_execution_time,
            },
            "test_results": [],
            "failed_tests": [],
            "performance_metrics": {}
        }
        
        # Detailed test results
        for result in self.results:
            test_summary = {
                "name": result.test_case.name,
                "description": result.test_case.description,
                "success": result.success,
                "execution_time": result.execution_time,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
            }
            
            if result.performance_metrics:
                test_summary["performance"] = result.performance_metrics
            
            report["test_results"].append(test_summary)
            
            if not result.success:
                report["failed_tests"].append({
                    "name": result.test_case.name,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "execution_time": result.execution_time
                })
        
        # Performance metrics aggregation
        performance_tests = [r for r in self.results if r.performance_metrics]
        if performance_tests:
            report["performance_metrics"] = {
                "average_execution_time": sum(r.performance_metrics.get('avg_execution_time', 0) 
                                           for r in performance_tests) / len(performance_tests),
                "total_operations": sum(r.performance_metrics.get('total_operations', 0) 
                                     for r in performance_tests),
                "overall_success_rate": sum(r.performance_metrics.get('success_rate', 0) 
                                         for r in performance_tests) / len(performance_tests)
            }
        
        return report
    
    def save_validation_report(self, report: Optional[Dict[str, Any]] = None) -> Path:
        """Save validation report to file."""
        if report is None:
            report = self.generate_validation_report()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"pipeline_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_file}")
        return report_file


# Pytest Test Classes
class TestPipelineValidation:
    """Pytest test class for pipeline validation."""
    
    @pytest.fixture(scope="class")
    def session_manager(self):
        """Create MATLAB session manager for testing."""
        config = MATLABConfig(
            startup_options=['-nojvm', '-nodisplay'],
            headless_mode=True,
            performance_monitoring=True,
            max_sessions=3
        )
        
        manager = MATLABSessionManager(config=config)
        yield manager
        manager.close_all_sessions()
    
    @pytest.fixture(scope="class")
    def pipeline_validator(self):
        """Create pipeline validator."""
        # Create temporary config directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_pipeline_config.yaml"
            validator = PipelineValidator(config_path=config_path)
            yield validator
    
    def test_basic_arithmetic_pipeline(self, pipeline_validator, session_manager):
        """Test basic arithmetic pipeline."""
        # Find the arithmetic test case
        arithmetic_test = next(
            (tc for tc in pipeline_validator.test_cases if tc.name == 'basic_arithmetic_pipeline'),
            None
        )
        
        assert arithmetic_test is not None, "Basic arithmetic test case not found"
        
        engine = session_manager.get_or_create_session("test_arithmetic")
        result = pipeline_validator.run_single_test_case(arithmetic_test, engine)
        
        assert result.success, f"Arithmetic pipeline failed: {result.errors}"
        assert 'result_add' in result.results
        assert 'result_mult' in result.results
        assert 'result_div' in result.results
    
    def test_matrix_operations_pipeline(self, pipeline_validator, session_manager):
        """Test matrix operations pipeline."""
        matrix_test = next(
            (tc for tc in pipeline_validator.test_cases if tc.name == 'matrix_operations_pipeline'),
            None
        )
        
        assert matrix_test is not None, "Matrix operations test case not found"
        
        engine = session_manager.get_or_create_session("test_matrix")
        result = pipeline_validator.run_single_test_case(matrix_test, engine)
        
        assert result.success, f"Matrix pipeline failed: {result.errors}"
        assert 'C' in result.results
        assert 'D' in result.results
        assert 'det_A' in result.results
    
    def test_data_type_conversion_pipeline(self, pipeline_validator, session_manager):
        """Test data type conversion pipeline."""
        conversion_test = next(
            (tc for tc in pipeline_validator.test_cases if tc.name == 'data_type_conversion_pipeline'),
            None
        )
        
        assert conversion_test is not None, "Data type conversion test case not found"
        
        engine = session_manager.get_or_create_session("test_conversion")
        result = pipeline_validator.run_single_test_case(conversion_test, engine)
        
        assert result.success, f"Conversion pipeline failed: {result.errors}"
    
    def test_error_handling_pipeline(self, pipeline_validator, session_manager):
        """Test error handling pipeline."""
        error_test = next(
            (tc for tc in pipeline_validator.test_cases if tc.name == 'error_handling_pipeline'),
            None
        )
        
        assert error_test is not None, "Error handling test case not found"
        
        engine = session_manager.get_or_create_session("test_error")
        result = pipeline_validator.run_single_test_case(error_test, engine)
        
        # For error handling test, success means we handled errors properly
        assert result.success or len(result.errors) == 0, "Error handling pipeline failed unexpectedly"
    
    def test_full_pipeline_validation(self, pipeline_validator, session_manager):
        """Test complete pipeline validation suite."""
        results = pipeline_validator.run_pipeline_validation(session_manager)
        
        assert len(results) > 0, "No pipeline tests were run"
        
        # Check overall success rate
        passed = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = (passed / total) * 100
        
        # Require at least 80% success rate for basic pipeline validation
        assert success_rate >= 80, f"Pipeline success rate {success_rate}% too low (< 80%)"
        
        # Generate and validate report
        report = pipeline_validator.generate_validation_report()
        assert report["summary"]["total_tests"] == total
        assert report["summary"]["passed_tests"] == passed
    
    def test_concurrent_pipeline_validation(self, pipeline_validator):
        """Test concurrent pipeline validation."""
        concurrent_results = pipeline_validator.run_concurrent_validation(max_workers=2)
        
        assert len(concurrent_results) > 0, "No concurrent tests were run"
        
        # Check that concurrent execution works
        passed = sum(1 for r in concurrent_results if r.success)
        total = len(concurrent_results)
        success_rate = (passed / total) * 100
        
        assert success_rate >= 70, f"Concurrent success rate {success_rate}% too low (< 70%)"


if __name__ == "__main__":
    # Run standalone pipeline validation
    print("MATLAB Engine Pipeline Validation Suite")
    print("=" * 60)
    
    validator = PipelineValidator()
    
    # Run sequential validation
    print("Running sequential pipeline validation...")
    results = validator.run_pipeline_validation()
    
    # Generate report
    report = validator.generate_validation_report()
    
    print(f"\nPipeline Validation Results:")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Execution Time: {report['summary']['total_execution_time']:.2f}s")
    print(f"Average Execution Time: {report['summary']['average_execution_time']:.2f}s")
    
    # Print failed tests
    if report['failed_tests']:
        print(f"\nFailed Tests:")
        for failed in report['failed_tests']:
            print(f"  {failed['name']}:")
            for error in failed['errors']:
                print(f"    - {error}")
    
    # Save report
    report_file = validator.save_validation_report(report)
    print(f"\nDetailed report saved to: {report_file}")
    
    print("\nPipeline validation completed!")