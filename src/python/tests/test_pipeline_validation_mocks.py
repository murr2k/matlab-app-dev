"""
Mock-based Pipeline Validation Test Suite for MATLAB Engine API
===============================================================

This module provides comprehensive mock-based pipeline validation tests 
that can run without MATLAB installed. These tests validate the Python code
structure and logic for the pipeline validation framework.

Part of Issue #1: MATLAB Engine API Integration testing framework.

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import yaml
import json
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class MockMATLABEngine:
    """Enhanced mock MATLAB engine for pipeline operations."""
    
    def __init__(self):
        self.workspace = {}
        self._closed = False
        self._call_history = []
        self.variables = {}
    
    def eval(self, expression, nargout=0, **kwargs):
        """Mock MATLAB eval with pipeline-specific operations."""
        self._call_history.append(('eval', expression, kwargs))
        
        if self._closed:
            raise RuntimeError("MATLAB engine is closed")
        
        # Handle variable assignments
        if '=' in expression and not expression.strip().startswith('['):
            var_name, value_expr = expression.split('=', 1)
            var_name = var_name.strip()
            value_expr = value_expr.strip()
            
            # Evaluate the right-hand side
            if value_expr == "10":
                self.variables[var_name] = 10
            elif value_expr == "5":
                self.variables[var_name] = 5
            elif value_expr == "a + b":
                self.variables[var_name] = self.variables.get('a', 0) + self.variables.get('b', 0)
            elif value_expr == "a * b":
                self.variables[var_name] = self.variables.get('a', 0) * self.variables.get('b', 0)
            elif value_expr == "a / b":
                self.variables[var_name] = self.variables.get('a', 0) / self.variables.get('b', 0)
            elif value_expr == "[1, 2; 3, 4]":
                self.variables[var_name] = np.array([[1, 2], [3, 4]])
            elif value_expr == "[5, 6; 7, 8]":
                self.variables[var_name] = np.array([[5, 6], [7, 8]])
            elif value_expr == "A * B":
                A = self.variables.get('A', np.array([[1, 2], [3, 4]]))
                B = self.variables.get('B', np.array([[5, 6], [7, 8]]))
                self.variables[var_name] = np.array([[19, 22], [43, 50]])
            elif value_expr == "inv(A)":
                self.variables[var_name] = np.array([[-2.0, 1.0], [1.5, -0.5]])
            elif value_expr == "det(A)":
                self.variables[var_name] = -2.0
            elif value_expr == "trace(A)":
                self.variables[var_name] = 5.0
            elif value_expr == "py_array":
                self.variables[var_name] = self.variables.get('py_array', np.array([1, 2, 3, 4, 5]))
            elif value_expr == "sum(python_array)":
                arr = self.variables.get('python_array', np.array([1, 2, 3, 4, 5]))
                self.variables[var_name] = np.sum(arr)
            elif value_expr == "mean(python_array)":
                arr = self.variables.get('python_array', np.array([1, 2, 3, 4, 5]))
                self.variables[var_name] = np.mean(arr)
            elif value_expr == "std(python_array)":
                arr = self.variables.get('python_array', np.array([1, 2, 3, 4, 5]))
                self.variables[var_name] = np.std(arr, ddof=0)
            elif value_expr == "sin(pi/2)":
                self.variables[var_name] = 1.0
            elif value_expr == "cos(pi)":
                self.variables[var_name] = -1.0
            elif value_expr == "tan(pi/4)":
                self.variables[var_name] = 1.0
            elif value_expr == "exp(1i * pi)":
                self.variables[var_name] = [-1.0, 0.0]  # Complex as real array
            elif value_expr == "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]":
                self.variables[var_name] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            elif value_expr == "mean(data)":
                self.variables[var_name] = 5.5
            elif value_expr == "std(data)":
                self.variables[var_name] = 3.02765
            elif value_expr == "var(data)":
                self.variables[var_name] = 9.16667
            elif value_expr == "median(data)":
                self.variables[var_name] = 5.5
            elif value_expr == "corrcoef(data, data.^2)":
                self.variables[var_name] = np.array([[1, 1], [1, 1]])  # Simplified
            elif "randn(1000, 1000)" in value_expr:
                self.variables[var_name] = np.random.randn(1000, 1000)
            elif "eig(" in value_expr:
                self.variables[var_name] = np.random.randn(1000) + 1j * np.random.randn(1000)
            elif "fft(" in value_expr:
                self.variables[var_name] = np.random.randn(1000, 1000) + 1j * np.random.randn(1000, 1000)
            elif "inv(" in value_expr:
                self.variables[var_name] = np.random.randn(1000, 1000)
            elif "1 / 0" in value_expr:
                raise RuntimeError("Division by zero")
            elif value_expr == "1000":
                self.variables[var_name] = 1000
            elif value_expr == "0:1/fs:1-1/fs":
                fs = self.variables.get('fs', 1000)
                self.variables[var_name] = np.arange(0, 1, 1/fs)
            elif value_expr == "50":
                self.variables[var_name] = 50
            elif value_expr == "120":
                self.variables[var_name] = 120
            elif "sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t)" in value_expr:
                t = self.variables.get('t', np.arange(0, 1, 0.001))
                f1 = self.variables.get('f1', 50)
                f2 = self.variables.get('f2', 120)
                signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
                self.variables[var_name] = signal
            elif value_expr == "fft(signal)":
                signal = self.variables.get('signal', np.ones(1000))
                self.variables[var_name] = np.fft.fft(signal)
            elif value_expr == "abs(signal_fft).^2":
                signal_fft = self.variables.get('signal_fft', np.ones(1000))
                self.variables[var_name] = np.abs(signal_fft)**2
            elif value_expr == "real(ifft(signal_fft))":
                signal_fft = self.variables.get('signal_fft', np.ones(1000))
                self.variables[var_name] = np.real(np.fft.ifft(signal_fft))
            elif value_expr == "max(abs(signal - signal_reconstructed))":
                signal = self.variables.get('signal', np.ones(1000))
                signal_reconstructed = self.variables.get('signal_reconstructed', np.ones(1000))
                self.variables[var_name] = np.max(np.abs(signal - signal_reconstructed))
            elif value_expr == "[1, 1; 2, -1]":
                self.variables[var_name] = np.array([[1, 1], [2, -1]])
            elif value_expr == "[3; 0]":
                self.variables[var_name] = np.array([3, 0])
            elif value_expr == "A_linear \\ b_linear":
                A = self.variables.get('A_linear', np.array([[1, 1], [2, -1]]))
                b = self.variables.get('b_linear', np.array([3, 0]))
                self.variables[var_name] = np.linalg.solve(A, b)
            elif value_expr == "roots([1, -5, 6])":
                self.variables[var_name] = np.array([3.0, 2.0])
            elif value_expr == "fzero(@(x) x^2 - 2, 1)":
                self.variables[var_name] = np.sqrt(2)
            elif "1 / 0" in value_expr:
                raise RuntimeError("Division by zero")
            else:
                self.variables[var_name] = 1.0
            
            return None
        
        # Handle matrix/array operations
        elif expression.startswith('[') and ']' in expression:
            if expression == "[t, theta, omega] = pendulum_simulation(1, pi/4, 0, [0 1])":
                # Mock physics simulation
                t = np.linspace(0, 1, 100)
                theta = -np.pi/4 * np.cos(np.sqrt(9.81) * t)
                omega = np.pi/4 * np.sqrt(9.81) * np.sin(np.sqrt(9.81) * t)
                self.variables['t'] = t
                self.variables['theta'] = theta
                self.variables['omega'] = omega
                return None
            elif "[Q,R] = qr(A); norm(Q'*Q - eye(2))" in expression:
                return 0.0
            elif "[t,y] = ode45(@(t,y) y, [0 1], 1); y(end)" in expression:
                return np.e
            elif "[dx,dy] = gradient([1,4,9;2,5,10;3,6,11]); dx(2,2)" in expression:
                return 3.0
        
        # Handle clear commands
        elif "clear" in expression:
            # Clear specified variables
            if "clear all" in expression:
                self.variables.clear()
            else:
                # Parse variable names to clear
                parts = expression.replace("clear", "").strip().split()
                for var in parts:
                    self.variables.pop(var, None)
            return None
        
        # Handle path additions
        elif "addpath" in expression:
            return None
        
        # Handle setup commands
        elif "rng(" in expression:
            return None
        elif "tic" in expression:
            return None
        elif "set(0" in expression:
            return None
        
        # Default expressions
        elif expression == "1+1":
            return 2
        else:
            return 1.0
    
    def quit(self):
        """Mock quit function."""
        self._closed = True


@pytest.fixture
def mock_matlab_engine():
    """Fixture providing a mock MATLAB engine for pipeline operations."""
    return MockMATLABEngine()


@pytest.fixture
def mock_matlab_module():
    """Fixture that patches the matlab module."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngine())
        yield mock_matlab


class TestPipelineValidationMocks:
    """Test pipeline validation framework using mocks."""
    
    def test_pipeline_validator_creation(self, mock_matlab_module):
        """Test creating pipeline validator with mocks."""
        from test_pipeline_validation import PipelineValidator
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            validator = PipelineValidator(config_path=config_path)
            
            assert validator.config_path == config_path
            assert len(validator.test_cases) > 0  # Should load default test cases
            assert validator.results == []
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_test_case_loading(self, mock_matlab_module):
        """Test loading test cases from configuration."""
        from test_pipeline_validation import PipelineValidator, PipelineTestCase
        
        # Create test configuration
        config_data = {
            'test_cases': [
                {
                    'name': 'test_case_1',
                    'description': 'Test case 1',
                    'setup_commands': ['a = 1'],
                    'test_commands': ['b = a + 1'],
                    'cleanup_commands': ['clear a b'],
                    'expected_results': {'b': 2},
                    'tolerances': {'b': 1e-15}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            validator = PipelineValidator(config_path=config_path)
            
            assert len(validator.test_cases) == 1
            assert validator.test_cases[0].name == 'test_case_1'
            assert validator.test_cases[0].description == 'Test case 1'
            assert validator.test_cases[0].expected_results == {'b': 2}
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_basic_arithmetic_pipeline(self, mock_matlab_module):
        """Test basic arithmetic pipeline with mocks."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Find the arithmetic test case
        arithmetic_test = next(
            (tc for tc in validator.test_cases if tc.name == 'basic_arithmetic_pipeline'),
            None
        )
        
        assert arithmetic_test is not None
        
        result = validator.run_single_test_case(arithmetic_test, engine)
        
        assert result.success is True
        assert 'result_add' in result.results
        assert 'result_mult' in result.results
        assert 'result_div' in result.results
        assert result.results['result_add'] == 15
        assert result.results['result_mult'] == 50
        assert result.results['result_div'] == 2.0
    
    def test_matrix_operations_pipeline(self, mock_matlab_module):
        """Test matrix operations pipeline with mocks."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Find the matrix test case
        matrix_test = next(
            (tc for tc in validator.test_cases if tc.name == 'matrix_operations_pipeline'),
            None
        )
        
        assert matrix_test is not None
        
        result = validator.run_single_test_case(matrix_test, engine)
        
        assert result.success is True
        assert 'C' in result.results
        assert 'D' in result.results
        assert 'det_A' in result.results
        assert 'trace_A' in result.results
    
    def test_data_type_conversion_pipeline(self, mock_matlab_module):
        """Test data type conversion pipeline with mocks."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Find the data type conversion test case
        conversion_test = next(
            (tc for tc in validator.test_cases if tc.name == 'data_type_conversion_pipeline'),
            None
        )
        
        assert conversion_test is not None
        
        result = validator.run_single_test_case(conversion_test, engine)
        
        # Should succeed with proper data type handling
        assert result.success is True
        assert 'matlab_sum' in result.results
        assert 'matlab_mean' in result.results
        assert 'matlab_std' in result.results
    
    def test_error_handling_pipeline(self, mock_matlab_module):
        """Test error handling pipeline with mocks."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Find the error handling test case
        error_test = next(
            (tc for tc in validator.test_cases if tc.name == 'error_handling_pipeline'),
            None
        )
        
        assert error_test is not None
        
        result = validator.run_single_test_case(error_test, engine)
        
        # Should handle errors gracefully (success means proper error handling)
        assert result.success is True or len(result.errors) == 0
    
    def test_performance_stress_pipeline(self, mock_matlab_module):
        """Test performance stress pipeline with mocks."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Find the performance test case
        performance_test = next(
            (tc for tc in validator.test_cases if tc.name == 'performance_stress_pipeline'),
            None
        )
        
        assert performance_test is not None
        
        result = validator.run_single_test_case(performance_test, engine)
        
        # Performance test should complete (may not validate specific results)
        assert result.execution_time > 0
        assert 'performance_test' in performance_test.metadata
    
    def test_full_pipeline_validation(self, mock_matlab_module):
        """Test complete pipeline validation suite with mocks."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True, max_sessions=2)
        
        with MATLABSessionManager(config=config) as manager:
            validator = PipelineValidator()
            results = validator.run_pipeline_validation(manager)
            
            assert len(results) > 0
            
            # Check that we have basic test results
            test_names = [result.test_case.name for result in results]
            assert 'basic_arithmetic_pipeline' in test_names
            assert 'matrix_operations_pipeline' in test_names
            
            # Generate report
            report = validator.generate_validation_report()
            
            assert report["summary"]["total_tests"] > 0
            assert "test_results" in report
            assert "performance_metrics" in report
    
    def test_concurrent_pipeline_validation(self, mock_matlab_module):
        """Test concurrent pipeline validation with mocks."""
        from test_pipeline_validation import PipelineValidator
        
        validator = PipelineValidator()
        
        # Run with limited concurrency to avoid resource issues in test environment
        results = validator.run_concurrent_validation(max_workers=2)
        
        assert len(results) > 0
        
        # Check that concurrent execution worked
        test_names = [result.test_case.name for result in results]
        assert len(set(test_names)) > 1  # Multiple different tests ran
    
    def test_pipeline_result_structure(self, mock_matlab_module):
        """Test pipeline result data structure."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator, PipelineTestCase, PipelineResult
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Create simple test case
        test_case = PipelineTestCase(
            name="test_result_structure",
            description="Test result structure",
            setup_commands=["a = 1"],
            test_commands=["b = a + 1"],
            cleanup_commands=["clear a b"],
            expected_results={"b": 2},
            tolerances={"b": 1e-15}
        )
        
        result = validator.run_single_test_case(test_case, engine)
        
        assert isinstance(result, PipelineResult)
        assert result.test_case == test_case
        assert isinstance(result.success, bool)
        assert result.execution_time >= 0
        assert isinstance(result.results, dict)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
    
    def test_result_comparison_methods(self, mock_matlab_module):
        """Test result comparison methods."""
        from test_pipeline_validation import PipelineValidator
        
        validator = PipelineValidator()
        
        # Test scalar comparison
        is_valid, error = validator._compare_results(5.0, 5.0, 1e-15)
        assert is_valid is True
        assert error == 0.0
        
        is_valid, error = validator._compare_results(5.0, 5.1, 0.01)
        assert is_valid is False
        assert error == 0.1
        
        # Test array comparison
        array1 = np.array([1, 2, 3])
        array2 = np.array([1, 2, 3])
        is_valid, error = validator._compare_results(array1, array2, 1e-15)
        assert is_valid is True
        assert error == 0.0
        
        # Test shape mismatch
        array3 = np.array([1, 2])
        is_valid, error = validator._compare_results(array1, array3, 1e-15)
        assert is_valid is False
        assert error == float('inf')
    
    def test_report_generation(self, mock_matlab_module):
        """Test pipeline validation report generation."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Run a few test cases
        arithmetic_test = next(
            (tc for tc in validator.test_cases if tc.name == 'basic_arithmetic_pipeline'),
            None
        )
        
        if arithmetic_test:
            result = validator.run_single_test_case(arithmetic_test, engine)
            validator.results.append(result)
        
        report = validator.generate_validation_report()
        
        assert "summary" in report
        assert report["summary"]["total_tests"] >= 1
        assert "test_results" in report
        assert "failed_tests" in report
        assert "performance_metrics" in report
        
        # Check summary fields
        summary = report["summary"]
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "success_rate" in summary
        assert "total_execution_time" in summary
        assert "average_execution_time" in summary
    
    def test_configuration_file_handling(self, mock_matlab_module):
        """Test configuration file handling."""
        from test_pipeline_validation import PipelineValidator
        
        # Test with non-existent config file
        non_existent_path = Path("/tmp/non_existent_config.yaml")
        validator = PipelineValidator(config_path=non_existent_path)
        
        # Should create default configuration
        assert len(validator.test_cases) > 0
        assert non_existent_path.exists()
        
        # Clean up
        non_existent_path.unlink(missing_ok=True)


class TestPipelineValidationEdgeCases:
    """Test edge cases and error conditions for pipeline validation."""
    
    def test_empty_test_case_list(self, mock_matlab_module):
        """Test handling of empty test case list."""
        from test_pipeline_validation import PipelineValidator
        
        # Create config with no test cases
        config_data = {'test_cases': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            validator = PipelineValidator(config_path=config_path)
            
            assert len(validator.test_cases) == 0
            
            # Generate report with no results
            report = validator.generate_validation_report()
            assert report["summary"]["total_tests"] == 0
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_malformed_configuration(self, mock_matlab_module):
        """Test handling of malformed configuration."""
        from test_pipeline_validation import PipelineValidator
        
        # Create malformed YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)
        
        try:
            # Should handle malformed YAML gracefully
            with pytest.raises((yaml.YAMLError, Exception)):
                validator = PipelineValidator(config_path=config_path)
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_test_case_timeout_handling(self, mock_matlab_module):
        """Test test case timeout handling."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator, PipelineTestCase
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Create test case with very short timeout
        test_case = PipelineTestCase(
            name="timeout_test",
            description="Test timeout handling",
            setup_commands=[],
            test_commands=["result = 1 + 1"],  # Simple, should not timeout
            cleanup_commands=[],
            expected_results={"result": 2},
            tolerances={"result": 1e-15},
            timeout=0.1  # Very short timeout
        )
        
        result = validator.run_single_test_case(test_case, engine)
        
        # Should complete quickly with mock
        assert result.execution_time <= 1.0  # Should be much faster with mocks
    
    def test_missing_expected_results(self, mock_matlab_module):
        """Test handling of missing expected results."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        from test_pipeline_validation import PipelineValidator, PipelineTestCase
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        validator = PipelineValidator()
        
        # Create test case that produces result not in expected_results
        test_case = PipelineTestCase(
            name="missing_results_test",
            description="Test missing expected results",
            setup_commands=[],
            test_commands=["unexpected_result = 42"],
            cleanup_commands=[],
            expected_results={"expected_result": 24},  # Different from what's produced
            tolerances={"expected_result": 1e-15}
        )
        
        result = validator.run_single_test_case(test_case, engine)
        
        # Should have warnings about missing expected results
        assert len(result.warnings) > 0
    
    def test_concurrent_execution_thread_safety(self, mock_matlab_module):
        """Test thread safety of concurrent execution."""
        from test_pipeline_validation import PipelineValidator
        
        validator = PipelineValidator()
        
        # Create multiple validator instances to test thread safety
        def run_validation():
            return validator.run_concurrent_validation(max_workers=1)
        
        # Run multiple validations concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_validation) for _ in range(2)]
            results_list = [future.result() for future in futures]
        
        # Both should complete successfully
        assert all(len(results) > 0 for results in results_list)


if __name__ == "__main__":
    # Run standalone tests
    print("Running Mock-based Pipeline Validation Tests...")
    print("=" * 60)
    
    # Mock the MATLAB module
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngine())
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        try:
            from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionManager
            from test_pipeline_validation import PipelineValidator
            
            print("✓ Successfully imported pipeline validation modules with mocks")
            
            # Test pipeline validator creation
            validator = PipelineValidator()
            print(f"✓ Created pipeline validator with {len(validator.test_cases)} test cases")
            
            # Test basic arithmetic pipeline
            config = MATLABConfig(headless_mode=True)
            engine = MATLABEngineWrapper(config=config)
            engine.start()
            
            arithmetic_test = next(
                (tc for tc in validator.test_cases if tc.name == 'basic_arithmetic_pipeline'),
                None
            )
            
            if arithmetic_test:
                result = validator.run_single_test_case(arithmetic_test, engine)
                print(f"✓ Basic arithmetic pipeline: {'PASSED' if result.success else 'FAILED'}")
            
            # Test full pipeline validation
            with MATLABSessionManager(config=config) as manager:
                results = validator.run_pipeline_validation(manager)
                report = validator.generate_validation_report()
                
                print(f"✓ Full pipeline validation: {report['summary']['success_rate']:.1f}% success rate")
                print(f"  Total tests: {report['summary']['total_tests']}")
                print(f"  Execution time: {report['summary']['total_execution_time']:.3f}s")
            
            # Test concurrent validation
            concurrent_results = validator.run_concurrent_validation(max_workers=2)
            print(f"✓ Concurrent validation: {len(concurrent_results)} tests completed")
            
            print("\n" + "=" * 60)
            print("All mock pipeline validation tests completed successfully!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise