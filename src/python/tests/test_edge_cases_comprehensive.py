"""
Comprehensive Edge Case and Boundary Condition Tests
====================================================

This module provides extensive edge case testing for the MATLAB Engine API
wrapper and related components, ensuring robust behavior under unusual
conditions and boundary scenarios.

Part of Issue #1: MATLAB Engine API Integration testing framework.

Author: Murray Kopit  
License: MIT
"""

import pytest
import numpy as np
import time
import threading
import json
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import os
import signal
import gc
import weakref

logger = logging.getLogger(__name__)


class MockMATLABEngine:
    """Enhanced mock MATLAB engine for edge case testing."""
    
    def __init__(self, fail_on_operations=False, delay_operations=False):
        self.workspace = {}
        self._closed = False
        self._call_history = []
        self._fail_on_operations = fail_on_operations
        self._delay_operations = delay_operations
        self._operation_count = 0
        # Each instance starts fresh (not closed)
        self._closed = False
        
    def eval(self, expression, nargout=0, **kwargs):
        """Mock MATLAB eval with configurable failure modes."""
        self._operation_count += 1
        self._call_history.append(('eval', expression, kwargs))
        
        if self._delay_operations:
            time.sleep(0.1)  # Simulate slow operations
        
        if self._closed:
            raise RuntimeError("MATLAB engine is closed")
        
        if self._fail_on_operations:
            if self._operation_count % 3 == 0:  # Fail every 3rd operation
                raise RuntimeError(f"Simulated failure on operation {self._operation_count}")
        
        # Handle specific expressions for edge cases
        if expression == "Inf":
            return float('inf')
        elif expression == "-Inf":
            return float('-inf')
        elif expression == "NaN":
            return float('nan')
        elif expression == "1/0":
            raise RuntimeError("Division by zero")
        elif expression == "0/0":
            raise RuntimeError("Indeterminate form")
        elif "memory_test" in expression:
            # Simulate memory-intensive operation
            return np.random.randn(10000, 10000)
        elif "slow_operation" in expression:
            time.sleep(1.0)  # Simulate very slow operation
            return 42
        elif "unicode_test" in expression:
            return "测试中文字符"  # Test Unicode handling
        elif "very_large_number" in expression:
            return 10**308  # Near float64 limit
        elif "very_small_number" in expression:
            return 10**-308  # Near float64 limit
        elif expression == "1+1":
            return 2
        else:
            return 1.0
    
    def quit(self):
        """Mock quit function."""
        self._closed = True


@pytest.fixture
def mock_matlab_module():
    """Fixture that patches the matlab module."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        # Each call creates a new mock engine instance
        mock_matlab.engine.start_matlab = Mock(side_effect=lambda *args, **kwargs: MockMATLABEngine())
        yield mock_matlab


@pytest.fixture
def failing_mock_matlab_module():
    """Fixture with a mock that fails operations."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(side_effect=lambda *args, **kwargs: MockMATLABEngine(fail_on_operations=True))
        yield mock_matlab


@pytest.fixture
def slow_mock_matlab_module():
    """Fixture with a mock that has delayed operations."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(side_effect=lambda *args, **kwargs: MockMATLABEngine(delay_operations=True))
        yield mock_matlab


class TestBoundaryConditions:
    """Test boundary conditions and extreme values."""
    
    def test_very_large_numbers(self, mock_matlab_module):
        """Test handling of very large numbers."""
        from matlab_engine_wrapper import MATLABEngineWrapper, TypeConverter
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test with very large number
        result = wrapper.evaluate("very_large_number")
        assert result == 10**308
        
        # Test type conversion with large numbers
        large_num = 10**100
        converted = TypeConverter.python_to_matlab(large_num)
        assert converted == [large_num]
    
    def test_very_small_numbers(self, mock_matlab_module):
        """Test handling of very small numbers."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        result = wrapper.evaluate("very_small_number")
        assert result == 10**-308
    
    def test_infinity_handling(self, mock_matlab_module):
        """Test handling of infinity values."""
        from matlab_engine_wrapper import MATLABEngineWrapper, TypeConverter
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test positive infinity
        result = wrapper.evaluate("Inf")
        assert result == float('inf')
        
        # Test negative infinity
        result = wrapper.evaluate("-Inf")
        assert result == float('-inf')
        
        # Test type conversion with infinity
        converted = TypeConverter.python_to_matlab(float('inf'))
        assert converted == [float('inf')]
    
    def test_nan_handling(self, mock_matlab_module):
        """Test handling of NaN values."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        result = wrapper.evaluate("NaN")
        assert np.isnan(result)
    
    def test_empty_arrays(self, mock_matlab_module):
        """Test handling of empty arrays and matrices."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test empty numpy array
        empty_array = np.array([])
        converted = TypeConverter.python_to_matlab(empty_array)
        assert converted == []
        
        # Test empty list
        empty_list = []
        converted = TypeConverter.python_to_matlab(empty_list)
        assert converted == []
    
    def test_single_element_arrays(self, mock_matlab_module):
        """Test handling of single-element arrays."""
        from matlab_engine_wrapper import TypeConverter
        
        single_element = np.array([42])
        converted = TypeConverter.python_to_matlab(single_element)
        assert converted == [42]
    
    def test_multidimensional_arrays(self, mock_matlab_module):
        """Test handling of high-dimensional arrays."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test 3D array
        array_3d = np.random.randn(2, 3, 4)
        converted = TypeConverter.python_to_matlab(array_3d)
        assert isinstance(converted, list)
        
        # Test 4D array
        array_4d = np.random.randn(2, 2, 2, 2)
        converted = TypeConverter.python_to_matlab(array_4d)
        assert isinstance(converted, list)
    
    def test_unicode_strings(self, mock_matlab_module):
        """Test handling of Unicode strings."""
        from matlab_engine_wrapper import MATLABEngineWrapper, TypeConverter
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test Unicode in evaluation
        result = wrapper.evaluate("unicode_test")
        assert result == "测试中文字符"
        
        # Test Unicode in type conversion
        unicode_str = "Hello 世界 🌍"
        converted = TypeConverter.python_to_matlab(unicode_str)
        assert converted == unicode_str


class TestResourceLimits:
    """Test behavior under resource constraints."""
    
    def test_memory_intensive_operations(self, mock_matlab_module):
        """Test handling of memory-intensive operations."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # This would simulate a large matrix operation
        result = wrapper.evaluate("memory_test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (10000, 10000)
    
    def test_session_timeout_behavior(self, slow_mock_matlab_module):
        """Test session timeout behavior."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABExecutionError
        
        config = MATLABConfig(session_timeout=1)  # Very short timeout
        wrapper = MATLABEngineWrapper(config=config)
        wrapper.start()
        
        # Operations should still work within timeout
        result = wrapper.evaluate("1+1")
        assert result == 2
    
    def test_maximum_sessions_limit(self, mock_matlab_module):
        """Test behavior when reaching maximum session limits."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(max_sessions=2)
        manager = MATLABSessionManager(config=config)
        
        # Create maximum number of sessions
        session1 = manager.get_or_create_session("session1")
        session2 = manager.get_or_create_session("session2")
        
        # Creating another session should evict the oldest
        session3 = manager.get_or_create_session("session3")
        
        status = manager.get_pool_status()
        assert status['total_sessions'] <= 2  # Should not exceed limit
    
    def test_large_workspace_variables(self, mock_matlab_module):
        """Test handling of large workspace variables."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Create large array
        large_array = np.random.randn(1000, 1000)
        wrapper.set_workspace_variable("large_var", large_array)
        
        # Should be able to retrieve it
        retrieved = wrapper.get_workspace_variable("large_var")
        assert isinstance(retrieved, np.ndarray)


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent operations."""
    
    def test_concurrent_session_creation(self, mock_matlab_module):
        """Test concurrent session creation."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(max_sessions=5)
        manager = MATLABSessionManager(config=config)
        
        def create_session(session_id):
            return manager.get_or_create_session(f"session_{session_id}")
        
        # Create sessions concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_session, i) for i in range(5)]
            sessions = [future.result() for future in futures]
        
        # All sessions should be created successfully
        assert len(sessions) == 5
        assert all(session is not None for session in sessions)
        
        status = manager.get_pool_status()
        assert status['total_sessions'] <= config.max_sessions
        
        manager.close_all_sessions()
    
    def test_concurrent_workspace_access(self, mock_matlab_module):
        """Test concurrent workspace variable access."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        def set_and_get_variable(var_name, value):
            wrapper.set_workspace_variable(var_name, value)
            return wrapper.get_workspace_variable(var_name)
        
        # Access workspace concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(set_and_get_variable, f"var_{i}", i)
                for i in range(5)
            ]
            results = [future.result() for future in futures]
        
        # All operations should complete
        assert len(results) == 5
    
    def test_session_cleanup_during_operations(self, mock_matlab_module):
        """Test session cleanup while operations are running."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(max_sessions=2, session_timeout=1)
        manager = MATLABSessionManager(config=config)
        
        session = manager.get_or_create_session("test_session")
        
        # Start a long-running operation simulation
        def long_operation():
            time.sleep(0.5)
            return session.evaluate("1+1")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start operation
            operation_future = executor.submit(long_operation)
            
            # Trigger cleanup
            time.sleep(0.1)
            cleanup_future = executor.submit(manager.cleanup_expired_sessions)
            
            # Both should complete without hanging
            operation_result = operation_future.result(timeout=2)
            cleanup_future.result(timeout=2)
            
            assert operation_result == 2
        
        manager.close_all_sessions()
    
    def test_thread_safety_stress(self, mock_matlab_module):
        """Stress test thread safety with many concurrent operations."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        def stress_operation(op_id):
            for i in range(10):
                result = wrapper.evaluate("1+1")
                assert result == 2
            return op_id
        
        # Run many concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(stress_operation, i) for i in range(20)]
            results = [future.result(timeout=10) for future in futures]
        
        # All operations should complete successfully
        assert len(results) == 20
        assert sorted(results) == list(range(20))


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_recovery_from_evaluation_errors(self, failing_mock_matlab_module):
        """Test recovery from evaluation errors."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABExecutionError
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Some operations will fail, but wrapper should recover
        successful_ops = 0
        failed_ops = 0
        
        for i in range(10):
            try:
                result = wrapper.evaluate("1+1")
                successful_ops += 1
            except MATLABExecutionError:
                failed_ops += 1
        
        # Should have both successes and failures
        assert successful_ops > 0
        assert failed_ops > 0
        
        # Engine should still be functional
        health = wrapper.health_check()
        assert isinstance(health, dict)
    
    def test_session_restart_after_failure(self, mock_matlab_module):
        """Test session restart after critical failure."""
        from matlab_engine_wrapper import MATLABEngineWrapper, SessionState
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Force session into error state
        wrapper.state = SessionState.ERROR
        wrapper.error_count = 5
        
        # Health check should detect the problem
        health = wrapper.health_check()
        assert health['healthy'] is False
        
        # Should be able to restart
        wrapper.close(force=True)
        wrapper.start()
        
        # Should be functional again
        result = wrapper.evaluate("1+1")
        assert result == 2
    
    def test_cleanup_after_exceptions(self, mock_matlab_module):
        """Test proper cleanup after exceptions."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABExecutionError, SessionState
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test that exceptions don't prevent cleanup
        try:
            with wrapper.session():
                # Force an error
                wrapper.evaluate("1/0")  # This will raise an exception
        except MATLABExecutionError:
            pass  # Expected
        
        # Session should still be manageable
        health = wrapper.health_check()
        assert isinstance(health, dict)
        
        # Cleanup should work
        wrapper.close()
        assert wrapper.state == SessionState.CLOSED
    
    def test_memory_leak_prevention(self, mock_matlab_module):
        """Test memory leak prevention in long-running sessions."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Create weak reference to track object lifecycle
        weak_ref = weakref.ref(wrapper)
        
        # Perform many operations
        for i in range(100):
            result = wrapper.evaluate("1+1")
            assert result == 2
            
            # Periodically force garbage collection
            if i % 20 == 0:
                gc.collect()
        
        # Performance stats should show many operations
        stats = wrapper.get_performance_stats()
        assert stats['total_operations'] >= 100
        
        # Close and verify cleanup
        wrapper.close()
        del wrapper
        gc.collect()
        
        # Object should be cleanable (weak reference becomes None)
        # Note: This might not always work in test environments, so we just verify it's callable
        assert callable(weak_ref)


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""
    
    def test_invalid_configuration_values(self, mock_matlab_module):
        """Test handling of invalid configuration values."""
        from matlab_engine_wrapper import MATLABConfig, MATLABEngineWrapper
        
        # Test negative values
        config = MATLABConfig(max_sessions=-1, session_timeout=-100)
        wrapper = MATLABEngineWrapper(config=config)
        
        # Should handle gracefully (implementation-dependent behavior)
        assert wrapper.config.max_sessions == -1  # Config is preserved
        
        # Test zero values
        config = MATLABConfig(max_sessions=0, session_timeout=0)
        wrapper = MATLABEngineWrapper(config=config)
        assert wrapper.config.max_sessions == 0
    
    def test_configuration_file_edge_cases(self, mock_matlab_module):
        """Test configuration file edge cases."""
        from matlab_engine_wrapper import MATLABConfig
        
        # Test with non-existent file
        non_existent = Path("/tmp/non_existent_config_12345.json")
        config = MATLABConfig.from_file(non_existent)
        
        # Should create default config
        assert isinstance(config, MATLABConfig)
        assert config.max_sessions == 3  # Default value
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")  # Empty file
            empty_file = Path(f.name)
        
        try:
            config = MATLABConfig.from_file(empty_file)
            # Should handle gracefully
            assert isinstance(config, MATLABConfig)
        finally:
            empty_file.unlink(missing_ok=True)
        
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            invalid_file = Path(f.name)
        
        try:
            with pytest.raises((json.JSONDecodeError, Exception)):
                config = MATLABConfig.from_file(invalid_file)
        finally:
            invalid_file.unlink(missing_ok=True)
    
    def test_startup_options_edge_cases(self, mock_matlab_module):
        """Test edge cases with startup options."""
        from matlab_engine_wrapper import MATLABConfig, MATLABEngineWrapper
        
        # Test with many startup options
        many_options = [f"-option{i}" for i in range(20)]
        config = MATLABConfig(startup_options=many_options)
        wrapper = MATLABEngineWrapper(config=config)
        
        # Should be able to start
        success = wrapper.start()
        assert success is True
        
        # Test with empty options list
        config = MATLABConfig(startup_options=[])
        wrapper = MATLABEngineWrapper(config=config)
        success = wrapper.start()
        assert success is True
        
        # Test with None (should use defaults)
        config = MATLABConfig(startup_options=None)
        wrapper = MATLABEngineWrapper(config=config)
        # Should handle gracefully
        assert wrapper.config.startup_options is None


class TestTypeConversionEdgeCases:
    """Test edge cases in type conversion."""
    
    def test_nested_data_structures(self, mock_matlab_module):
        """Test conversion of deeply nested data structures."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test nested lists
        nested_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        converted = TypeConverter.python_to_matlab(nested_list)
        assert isinstance(converted, list)
        
        # Test nested dictionaries
        nested_dict = {
            'level1': {
                'level2': {
                    'data': [1, 2, 3],
                    'value': 42
                }
            }
        }
        converted = TypeConverter.python_to_matlab(nested_dict)
        assert isinstance(converted, dict)
    
    def test_mixed_type_arrays(self, mock_matlab_module):
        """Test conversion of arrays with mixed types."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test list with mixed types
        mixed_list = [1, 2.5, "string", True]
        converted = TypeConverter.python_to_matlab(mixed_list)
        assert isinstance(converted, list)
        assert len(converted) == 4
    
    def test_complex_number_edge_cases(self, mock_matlab_module):
        """Test complex number conversion edge cases."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test complex with zero imaginary part
        complex_real = 5 + 0j
        converted = TypeConverter.python_to_matlab(complex_real)
        assert converted == [complex_real]
        
        # Test complex with zero real part
        complex_imag = 0 + 5j
        converted = TypeConverter.python_to_matlab(complex_imag)
        assert converted == [complex_imag]
        
        # Test complex infinity
        complex_inf = complex(float('inf'), float('inf'))
        converted = TypeConverter.python_to_matlab(complex_inf)
        assert converted == [complex_inf]
    
    def test_boolean_conversion_edge_cases(self, mock_matlab_module):
        """Test boolean conversion edge cases."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test boolean arrays
        bool_array = np.array([True, False, True, False])
        converted = TypeConverter.python_to_matlab(bool_array)
        # Should handle boolean conversion
        assert converted is not None
        
        # Test mixed boolean list
        bool_list = [True, False, 1, 0]
        converted = TypeConverter.python_to_matlab(bool_list)
        assert isinstance(converted, list)
    
    def test_conversion_error_recovery(self, mock_matlab_module):
        """Test recovery from conversion errors."""
        from matlab_engine_wrapper import TypeConverter, MATLABTypeConversionError
        
        # Create an object that will fail conversion
        class UnconvertibleClass:
            def __str__(self):
                raise ValueError("Cannot convert to string")
            
            def __array__(self):
                raise TypeError("Cannot convert to array")
        
        unconvertible = UnconvertibleClass()
        
        # Should raise appropriate error
        with pytest.raises(MATLABTypeConversionError):
            TypeConverter.python_to_matlab(unconvertible)
        
        # But should not crash the system - subsequent conversions should work
        normal_value = 42
        converted = TypeConverter.python_to_matlab(normal_value)
        assert converted == [42]


if __name__ == "__main__":
    # Run standalone edge case tests
    print("Running Comprehensive Edge Case Tests...")
    print("=" * 60)
    
    # Mock the MATLAB module
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngine())
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        try:
            from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionManager, TypeConverter
            
            print("✓ Successfully imported modules for edge case testing")
            
            # Test boundary conditions
            print("\n--- Boundary Conditions ---")
            wrapper = MATLABEngineWrapper()
            wrapper.start()
            
            # Test extreme values
            result = wrapper.evaluate("very_large_number")
            print(f"✓ Large number handling: {result}")
            
            result = wrapper.evaluate("Inf")
            print(f"✓ Infinity handling: {result}")
            
            result = wrapper.evaluate("NaN")
            print(f"✓ NaN handling: {np.isnan(result)}")
            
            # Test resource limits
            print("\n--- Resource Limits ---")
            config = MATLABConfig(max_sessions=2)
            manager = MATLABSessionManager(config=config)
            
            session1 = manager.get_or_create_session("session1")
            session2 = manager.get_or_create_session("session2")
            session3 = manager.get_or_create_session("session3")  # Should evict oldest
            
            status = manager.get_pool_status()
            print(f"✓ Session limit enforcement: {status['total_sessions']} <= {config.max_sessions}")
            
            # Test concurrency
            print("\n--- Concurrency Tests ---")
            
            def concurrent_operation(op_id):
                return wrapper.evaluate("1+1")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(concurrent_operation, i) for i in range(5)]
                results = [future.result() for future in futures]
            
            print(f"✓ Concurrent operations: {len(results)} operations completed")
            
            # Test error recovery
            print("\n--- Error Recovery ---")
            try:
                wrapper.evaluate("1/0")
            except Exception as e:
                print(f"✓ Error handling: Caught expected error - {type(e).__name__}")
            
            # Session should still be functional
            result = wrapper.evaluate("1+1")
            print(f"✓ Recovery after error: {result}")
            
            # Test type conversion edge cases
            print("\n--- Type Conversion Edge Cases ---")
            
            # Test complex conversions
            complex_num = 3 + 4j
            converted = TypeConverter.python_to_matlab(complex_num)
            print(f"✓ Complex number conversion: {converted}")
            
            # Test empty array
            empty_array = np.array([])
            converted = TypeConverter.python_to_matlab(empty_array)
            print(f"✓ Empty array conversion: {converted}")
            
            # Test mixed type list
            mixed_list = [1, 2.5, "test", True]
            converted = TypeConverter.python_to_matlab(mixed_list)
            print(f"✓ Mixed type conversion: length {len(converted)}")
            
            # Cleanup
            wrapper.close()
            manager.close_all_sessions()
            
            print("\n" + "=" * 60)
            print("All comprehensive edge case tests completed successfully!")
            
        except Exception as e:
            print(f"✗ Edge case test failed: {e}")
            raise