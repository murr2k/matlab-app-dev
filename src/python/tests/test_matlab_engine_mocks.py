"""
Mock-based Test Suite for MATLAB Engine API
===========================================

This module provides comprehensive tests using mocks for environments where
MATLAB is not available. These tests validate the Python code structure,
logic, and error handling without requiring an actual MATLAB installation.

This is part of Issue #1: MATLAB Engine API Integration testing framework.

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
import tempfile
from dataclasses import dataclass, field

# Mock MATLAB engine module
mock_matlab = MagicMock()
mock_matlab.engine = MagicMock()
mock_matlab.double = MagicMock(side_effect=lambda x: x)  # Pass through
mock_matlab.logical = MagicMock(side_effect=lambda x: x)  # Pass through

logger = logging.getLogger(__name__)


class MockMATLABEngine:
    """Mock MATLAB Engine for testing."""
    
    def __init__(self):
        self.workspace = {}
        self._closed = False
        self._call_history = []
        
    def eval(self, expression, nargout=0, **kwargs):
        """Mock MATLAB eval function."""
        self._call_history.append(('eval', expression, kwargs))
        
        if self._closed:
            raise RuntimeError("MATLAB engine is closed")
            
        # Handle specific test expressions
        if expression == "1+1":
            return 2
        elif expression == "sqrt(64)":
            return 8.0
        elif expression == "sin(pi/2)":
            return 1.0
        elif expression == "cos(pi)":
            return -1.0
        elif expression == "2 + 3":
            return 5
        elif expression == "10 - 4":
            return 6
        elif expression == "7 * 8":
            return 56
        elif expression == "15 / 3":
            return 5.0
        elif expression == "2^10":
            return 1024
        elif expression == "[1, 2; 3, 4]":
            return np.array([[1, 2], [3, 4]])
        elif "det(" in expression:
            return -2.0
        elif "inv(" in expression:
            return np.array([[-2.0, 1.0], [1.5, -0.5]])
        elif "fft(" in expression:
            return np.array([1, 1, 1, 1])  # Simple mock
        elif "1 / 0" in expression:
            raise RuntimeError("Division by zero")
        elif "clear" in expression:
            return None
        else:
            # Default return for unknown expressions
            return 1.0
    
    def quit(self):
        """Mock quit function."""
        self._closed = True
    
    def __getattr__(self, name):
        """Mock function calls."""
        def mock_func(*args, **kwargs):
            self._call_history.append(('function', name, args, kwargs))
            
            if name == 'sin':
                return np.sin(args[0]) if args else 0
            elif name == 'cos':
                return np.cos(args[0]) if args else 1
            elif name == 'sqrt':
                return np.sqrt(args[0]) if args else 0
            elif name == 'zeros':
                return np.zeros(args) if args else np.array([0])
            elif name == 'ones':
                return np.ones(args) if args else np.array([1])
            else:
                return 1.0
        
        return mock_func


@pytest.fixture
def mock_matlab_engine():
    """Fixture providing a mock MATLAB engine."""
    return MockMATLABEngine()


@pytest.fixture
def mock_matlab_module():
    """Fixture that patches the matlab module."""
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngine())
        yield mock_matlab


class TestMATLABEngineWrapperMocks:
    """Test MATLAB Engine Wrapper using mocks."""
    
    def test_import_wrapper_with_mocks(self, mock_matlab_module):
        """Test that we can import the wrapper with mocked MATLAB."""
        # Import here to ensure mock is active
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, SessionState
        
        config = MATLABConfig(startup_options=['-nojvm', '-nodisplay'])
        wrapper = MATLABEngineWrapper(config=config)
        
        assert wrapper.config == config
        assert wrapper.state == SessionState.IDLE
        assert wrapper.engine is None
    
    def test_session_startup_with_mocks(self, mock_matlab_module):
        """Test session startup with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, SessionState
        
        config = MATLABConfig(startup_options=['-nojvm'])
        wrapper = MATLABEngineWrapper(config=config)
        
        success = wrapper.start()
        
        assert success is True
        assert wrapper.state == SessionState.ACTIVE
        assert wrapper.engine is not None
        assert wrapper.session_start_time is not None
    
    def test_evaluate_with_mocks(self, mock_matlab_module):
        """Test evaluation with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        result = wrapper.evaluate("sqrt(64)")
        assert result == 8.0
        
        result = wrapper.evaluate("1+1")
        assert result == 2
    
    def test_function_call_with_mocks(self, mock_matlab_module):
        """Test function calls with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        result = wrapper.call_function("sin", np.pi/2)
        assert abs(result - 1.0) < 1e-10
        
        result = wrapper.call_function("sqrt", 64)
        assert result == 8.0
    
    def test_workspace_operations_with_mocks(self, mock_matlab_module):
        """Test workspace operations with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test setting variable
        test_array = np.array([1, 2, 3, 4, 5])
        wrapper.set_workspace_variable("test_var", test_array)
        
        # Mock the workspace access
        wrapper.engine.workspace["test_var"] = test_array
        result = wrapper.get_workspace_variable("test_var")
        
        assert np.array_equal(result, test_array)
    
    def test_error_handling_with_mocks(self, mock_matlab_module):
        """Test error handling with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABExecutionError
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Test division by zero
        with pytest.raises(MATLABExecutionError):
            wrapper.evaluate("1 / 0")
    
    def test_performance_stats_with_mocks(self, mock_matlab_module):
        """Test performance statistics with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Perform some operations
        wrapper.evaluate("sqrt(64)")
        wrapper.evaluate("1+1")
        
        stats = wrapper.get_performance_stats()
        
        assert stats['total_operations'] == 2
        assert stats['successful_operations'] == 2
        assert stats['failed_operations'] == 0
        assert stats['avg_execution_time'] >= 0
    
    def test_health_check_with_mocks(self, mock_matlab_module):
        """Test health check with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        health = wrapper.health_check()
        
        assert health['healthy'] is True
        assert health['test_passed'] is True
        assert 'session_id' in health
        assert health['uptime'] >= 0
    
    def test_session_context_manager_with_mocks(self, mock_matlab_module):
        """Test context manager with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, SessionState
        
        config = MATLABConfig(startup_options=['-nojvm'])
        
        with MATLABEngineWrapper(config=config) as wrapper:
            assert wrapper.state == SessionState.ACTIVE
            result = wrapper.evaluate("1+1")
            assert result == 2
        
        assert wrapper.state == SessionState.CLOSED


class TestMATLABSessionManagerMocks:
    """Test MATLAB Session Manager using mocks."""
    
    def test_session_manager_creation_with_mocks(self, mock_matlab_module):
        """Test session manager creation with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(max_sessions=2)
        manager = MATLABSessionManager(config=config)
        
        assert manager.config.max_sessions == 2
        assert len(manager.sessions) == 0
    
    def test_session_creation_with_mocks(self, mock_matlab_module):
        """Test session creation with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager, SessionState
        
        manager = MATLABSessionManager()
        session = manager.get_or_create_session("test_session")
        
        assert session is not None
        assert session.state == SessionState.ACTIVE
        assert "test_session" in manager.sessions
    
    def test_session_reuse_with_mocks(self, mock_matlab_module):
        """Test session reuse with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager
        
        manager = MATLABSessionManager()
        session1 = manager.get_or_create_session("test_session")
        session2 = manager.get_or_create_session("test_session")
        
        assert session1 is session2
        assert len(manager.sessions) == 1
    
    def test_pool_status_with_mocks(self, mock_matlab_module):
        """Test pool status with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager
        
        manager = MATLABSessionManager()
        session1 = manager.get_or_create_session("session1")
        session2 = manager.get_or_create_session("session2")
        
        status = manager.get_pool_status()
        
        assert status['total_sessions'] == 2
        assert status['active_sessions'] == 2
        assert status['pool_utilization'] > 0
        assert 'session_details' in status
    
    def test_session_cleanup_with_mocks(self, mock_matlab_module):
        """Test session cleanup with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager
        
        manager = MATLABSessionManager()
        session = manager.get_or_create_session("test_session")
        
        manager.close_session("test_session")
        
        assert "test_session" not in manager.sessions
        assert len(manager.sessions) == 0
    
    def test_context_manager_cleanup_with_mocks(self, mock_matlab_module):
        """Test context manager cleanup with mocked MATLAB."""
        from matlab_engine_wrapper import MATLABSessionManager
        
        with MATLABSessionManager() as manager:
            session = manager.get_or_create_session("test_session")
            assert len(manager.sessions) == 1
        
        # After context manager exit, sessions should be cleaned up
        assert len(manager.sessions) == 0


class TestTypeConverterMocks:
    """Test type converter with mocks."""
    
    def test_python_to_matlab_conversion_with_mocks(self, mock_matlab_module):
        """Test Python to MATLAB type conversion."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test numpy array
        py_array = np.array([1, 2, 3, 4])
        matlab_result = TypeConverter.python_to_matlab(py_array)
        assert matlab_result == [1, 2, 3, 4]  # Mock returns list
        
        # Test list
        py_list = [1, 2, 3, 4]
        matlab_result = TypeConverter.python_to_matlab(py_list)
        assert matlab_result == [1, 2, 3, 4]
        
        # Test scalar
        py_scalar = 42
        matlab_result = TypeConverter.python_to_matlab(py_scalar)
        assert matlab_result == [42]
    
    def test_matlab_to_python_conversion_with_mocks(self, mock_matlab_module):
        """Test MATLAB to Python type conversion."""
        from matlab_engine_wrapper import TypeConverter
        
        # Test with mock MATLAB array
        mock_matlab_array = Mock()
        mock_matlab_array._data = [1, 2, 3, 4]
        mock_matlab_array._size = (2, 2)
        
        result = TypeConverter.matlab_to_python(mock_matlab_array)
        expected = np.array([1, 2, 3, 4]).reshape((2, 2))
        assert np.array_equal(result, expected)
        
        # Test with regular Python objects
        py_list = [1, 2, 3]
        result = TypeConverter.matlab_to_python(py_list)
        assert result == [1, 2, 3]


class TestConfigurationMocks:
    """Test configuration handling with mocks."""
    
    def test_matlab_config_defaults(self, mock_matlab_module):
        """Test MATLABConfig defaults."""
        from matlab_engine_wrapper import MATLABConfig
        
        config = MATLABConfig()
        
        assert config.max_sessions == 3
        assert config.session_timeout == 300
        assert config.max_retries == 3
        assert config.workspace_persistence is True
    
    def test_matlab_config_from_dict(self, mock_matlab_module):
        """Test MATLABConfig from dictionary."""
        from matlab_engine_wrapper import MATLABConfig
        
        config_data = {
            'max_sessions': 5,
            'session_timeout': 600,
            'startup_options': ['-nojvm', '-nodisplay'],
            'headless_mode': True
        }
        
        config = MATLABConfig(**config_data)
        
        assert config.max_sessions == 5
        assert config.session_timeout == 600
        assert config.startup_options == ['-nojvm', '-nodisplay']
        assert config.headless_mode is True
    
    def test_config_file_operations_with_mocks(self, mock_matlab_module):
        """Test config file save/load operations."""
        from matlab_engine_wrapper import MATLABConfig
        
        config = MATLABConfig(max_sessions=10, session_timeout=1200)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Test save
            config.save_to_file(config_path)
            assert config_path.exists()
            
            # Test load
            loaded_config = MATLABConfig.from_file(config_path)
            assert loaded_config.max_sessions == 10
            assert loaded_config.session_timeout == 1200
        finally:
            config_path.unlink(missing_ok=True)


class TestErrorHandlingMocks:
    """Test comprehensive error handling with mocks."""
    
    def test_session_not_started_error(self, mock_matlab_module):
        """Test error when session not started."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABSessionError
        
        wrapper = MATLABEngineWrapper()
        # Don't start the engine
        
        with pytest.raises(MATLABSessionError):
            wrapper.evaluate("1+1")
    
    def test_invalid_expression_error(self, mock_matlab_module):
        """Test error handling for invalid expressions."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABExecutionError
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        with pytest.raises(MATLABExecutionError):
            wrapper.evaluate("1 / 0")  # This triggers the mock error
    
    def test_type_conversion_error(self, mock_matlab_module):
        """Test type conversion error handling."""
        from matlab_engine_wrapper import TypeConverter, MATLABTypeConversionError
        
        # Test with an object that can't be converted
        class UnconvertibleObject:
            def __array__(self):
                raise ValueError("Cannot convert to array")
        
        obj = UnconvertibleObject()
        
        with pytest.raises(MATLABTypeConversionError):
            TypeConverter.python_to_matlab(obj)
    
    def test_retry_mechanism_with_mocks(self, mock_matlab_module):
        """Test retry mechanism with mocked failures."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionError
        
        # Mock start_matlab to fail first attempts
        call_count = 0
        original_start_matlab = mock_matlab.engine.start_matlab
        
        def failing_start_matlab(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:  # Fail first attempt
                raise RuntimeError("Mock startup failure")
            return MockMATLABEngine()
        
        mock_matlab.engine.start_matlab = failing_start_matlab
        
        try:
            config = MATLABConfig(max_retries=3, retry_delay=0.1)
            wrapper = MATLABEngineWrapper(config=config)
            
            success = wrapper.start()
            assert success is True
            assert call_count == 2  # One failure, one success
        finally:
            # Restore original mock
            mock_matlab.engine.start_matlab = original_start_matlab


class TestPerformanceMonitoringMocks:
    """Test performance monitoring with mocks."""
    
    def test_performance_stats_tracking(self, mock_matlab_module):
        """Test performance statistics tracking."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        wrapper.start()
        
        # Perform some operations
        wrapper.evaluate("sqrt(64)")
        wrapper.evaluate("1+1")
        
        try:
            wrapper.evaluate("1 / 0")  # This should fail
        except:
            pass
        
        stats = wrapper.get_performance_stats()
        
        assert stats['total_operations'] == 3
        assert stats['successful_operations'] == 2
        assert stats['failed_operations'] == 1
        assert stats['avg_execution_time'] >= 0
        assert stats['total_execution_time'] >= 0
    
    def test_session_timing(self, mock_matlab_module):
        """Test session timing functionality."""
        from matlab_engine_wrapper import MATLABEngineWrapper
        
        wrapper = MATLABEngineWrapper()
        start_time = time.time()
        wrapper.start()
        
        # Wait a bit
        time.sleep(0.1)
        
        stats = wrapper.get_performance_stats()
        
        assert stats['session_duration'] >= 0.1
        assert wrapper.session_start_time >= start_time


if __name__ == "__main__":
    # Run the mock tests
    print("Running Mock-based MATLAB Engine Tests...")
    print("=" * 50)
    
    # Test import
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngine())
        
        try:
            from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionManager
            print("✓ Successfully imported MATLAB wrapper with mocks")
            
            # Test basic functionality
            config = MATLABConfig(startup_options=['-nojvm'])
            wrapper = MATLABEngineWrapper(config=config)
            
            success = wrapper.start()
            print(f"✓ Engine startup: {success}")
            
            result = wrapper.evaluate("sqrt(64)")
            print(f"✓ Basic evaluation: sqrt(64) = {result}")
            
            result = wrapper.call_function("sin", np.pi/2)
            print(f"✓ Function call: sin(pi/2) = {result}")
            
            stats = wrapper.get_performance_stats()
            print(f"✓ Performance stats: {stats['total_operations']} operations")
            
            health = wrapper.health_check()
            print(f"✓ Health check: {'Healthy' if health['healthy'] else 'Unhealthy'}")
            
            wrapper.close()
            print("✓ Session closed successfully")
            
            # Test session manager
            with MATLABSessionManager() as manager:
                session = manager.get_or_create_session("test")
                pool_status = manager.get_pool_status()
                print(f"✓ Session manager: {pool_status['total_sessions']} sessions")
            
            print("\n" + "=" * 50)
            print("All mock tests completed successfully!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise