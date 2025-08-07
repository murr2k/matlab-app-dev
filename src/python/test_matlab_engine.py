"""
Comprehensive Test Suite for MATLAB Engine Wrapper
=================================================

This module provides comprehensive testing for the enhanced MATLAB Engine wrapper,
covering session management, error handling, type conversion, and performance monitoring.

Author: Murray Kopit  
License: MIT
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Import our modules
from matlab_engine_wrapper import (
    MATLABEngineWrapper,
    MATLABSessionManager,
    MATLABConfig,
    TypeConverter,
    MATLABEngineError,
    MATLABSessionError,
    MATLABTypeConversionError,
    MATLABExecutionError,
    SessionState
)
from config_manager import (
    ConfigurationManager,
    MATLABEngineConfig,
    Environment
)


class TestTypeConverter:
    """Test type conversion functionality."""
    
    def test_python_to_matlab_basic_types(self):
        """Test conversion of basic Python types to MATLAB."""
        # Test numbers
        result = TypeConverter.python_to_matlab(42)
        assert hasattr(result, '_data') or isinstance(result, (int, float))
        
        # Test strings
        result = TypeConverter.python_to_matlab("test")
        assert result == "test"
        
        # Test boolean
        result = TypeConverter.python_to_matlab(True)
        assert hasattr(result, '_data') or isinstance(result, bool)
    
    def test_python_to_matlab_numpy_arrays(self):
        """Test conversion of numpy arrays to MATLAB."""
        # Test 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = TypeConverter.python_to_matlab(arr_1d)
        # Should be convertible back
        assert result is not None
        
        # Test 2D array
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        result = TypeConverter.python_to_matlab(arr_2d)
        assert result is not None
    
    def test_python_to_matlab_complex_structures(self):
        """Test conversion of complex Python structures."""
        # Test list
        python_list = [1, 2, 3, "hello", True]
        result = TypeConverter.python_to_matlab(python_list)
        assert isinstance(result, list)
        
        # Test dictionary
        python_dict = {"key1": 42, "key2": "value", "key3": [1, 2, 3]}
        result = TypeConverter.python_to_matlab(python_dict)
        assert isinstance(result, dict)
    
    def test_matlab_to_python_conversion(self):
        """Test MATLAB to Python type conversion."""
        # Test with mock MATLAB objects
        mock_matlab_array = Mock()
        mock_matlab_array._data = [1, 2, 3, 4]
        mock_matlab_array._size = (2, 2)
        
        result = TypeConverter.matlab_to_python(mock_matlab_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
    
    def test_type_conversion_errors(self):
        """Test type conversion error handling."""
        # Test with unsupported type that might cause issues
        class UnsupportedType:
            pass
        
        # Should not raise exception, but might return original or converted value
        result = TypeConverter.python_to_matlab(UnsupportedType())
        # The converter should handle this gracefully


class TestMATLABConfig:
    """Test MATLAB configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = MATLABConfig()
        
        assert config.max_sessions == 3
        assert config.session_timeout == 300
        assert config.workspace_persistence == True
        assert config.performance_monitoring == True
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_data = {
            "max_sessions": 5,
            "session_timeout": 600,
            "startup_options": ["-nojvm", "-nodisplay"],
            "workspace_persistence": False
        }
        
        config = MATLABConfig(**config_data)
        assert config.max_sessions == 5
        assert config.session_timeout == 600
        assert "-nojvm" in config.startup_options
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Create and save config
            original_config = MATLABConfig(
                max_sessions=5,
                session_timeout=400,
                startup_options=["-nojvm"]
            )
            original_config.save_to_file(config_path)
            
            # Load config
            loaded_config = MATLABConfig.from_file(config_path)
            
            assert loaded_config.max_sessions == 5
            assert loaded_config.session_timeout == 400
            assert "-nojvm" in loaded_config.startup_options


class TestMATLABEngineWrapper:
    """Test MATLAB Engine wrapper functionality."""
    
    @pytest.fixture
    def mock_matlab_engine(self):
        """Create mock MATLAB engine for testing."""
        with patch('matlab_engine_wrapper.matlab.engine') as mock_engine:
            mock_instance = Mock()
            mock_engine.start_matlab.return_value = mock_instance
            
            # Mock basic methods
            mock_instance.eval.return_value = 2.0
            mock_instance.sin.return_value = 1.0
            mock_instance.usejava.return_value = True
            mock_instance.workspace = {}
            mock_instance.quit = Mock()
            
            yield mock_instance
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        config = MATLABConfig(max_sessions=2, session_timeout=120)
        wrapper = MATLABEngineWrapper(config=config, session_id="test_session")
        
        assert wrapper.session_id == "test_session"
        assert wrapper.config.max_sessions == 2
        assert wrapper.state == SessionState.IDLE
    
    def test_wrapper_start_success(self, mock_matlab_engine):
        """Test successful wrapper startup."""
        wrapper = MATLABEngineWrapper()
        
        result = wrapper.start()
        
        assert result == True
        assert wrapper.state == SessionState.ACTIVE
        assert wrapper.session_start_time is not None
    
    def test_wrapper_start_with_retries(self):
        """Test wrapper startup with retry logic."""
        config = MATLABConfig(max_retries=3, retry_delay=0.1)
        wrapper = MATLABEngineWrapper(config=config)
        
        with patch('matlab_engine_wrapper.matlab.engine') as mock_engine:
            # First two attempts fail, third succeeds
            mock_engine.start_matlab.side_effect = [
                Exception("First failure"),
                Exception("Second failure"),
                Mock()  # Success
            ]
            
            result = wrapper.start()
            assert result == True
            assert mock_engine.start_matlab.call_count == 3
    
    def test_wrapper_start_failure(self):
        """Test wrapper startup failure after max retries."""
        config = MATLABConfig(max_retries=2, retry_delay=0.1)
        wrapper = MATLABEngineWrapper(config=config)
        
        with patch('matlab_engine_wrapper.matlab.engine') as mock_engine:
            mock_engine.start_matlab.side_effect = Exception("Persistent failure")
            
            with pytest.raises(MATLABSessionError):
                wrapper.start()
    
    def test_evaluate_function(self, mock_matlab_engine):
        """Test expression evaluation."""
        wrapper = MATLABEngineWrapper()
        wrapper.engine = mock_matlab_engine
        wrapper.state = SessionState.ACTIVE
        
        result = wrapper.evaluate("2+2")
        
        assert result == 2.0
        mock_matlab_engine.eval.assert_called_once()
    
    def test_evaluate_without_active_session(self):
        """Test evaluation without active session."""
        wrapper = MATLABEngineWrapper()
        wrapper.state = SessionState.IDLE
        
        with pytest.raises(MATLABSessionError):
            wrapper.evaluate("2+2")
    
    def test_call_function(self, mock_matlab_engine):
        """Test function calling."""
        wrapper = MATLABEngineWrapper()
        wrapper.engine = mock_matlab_engine
        wrapper.state = SessionState.ACTIVE
        
        result = wrapper.call_function("sin", np.pi/2)
        
        assert result == 1.0
        mock_matlab_engine.sin.assert_called_once()
    
    def test_workspace_operations(self, mock_matlab_engine):
        """Test workspace variable operations."""
        wrapper = MATLABEngineWrapper()
        wrapper.engine = mock_matlab_engine
        wrapper.state = SessionState.ACTIVE
        
        # Set variable
        wrapper.set_workspace_variable("test_var", 42)
        assert "test_var" in mock_matlab_engine.workspace
        
        # Get variable
        mock_matlab_engine.workspace["test_var"] = 42
        result = wrapper.get_workspace_variable("test_var")
        assert result == 42
    
    def test_performance_stats_tracking(self, mock_matlab_engine):
        """Test performance statistics tracking."""
        wrapper = MATLABEngineWrapper()
        wrapper.engine = mock_matlab_engine
        wrapper.state = SessionState.ACTIVE
        
        # Perform some operations
        wrapper.evaluate("2+2")
        wrapper.call_function("sin", 1)
        
        stats = wrapper.get_performance_stats()
        
        assert stats['total_operations'] == 2
        assert stats['successful_operations'] == 2
        assert stats['failed_operations'] == 0
        assert stats['avg_execution_time'] >= 0
    
    def test_health_check(self, mock_matlab_engine):
        """Test health check functionality."""
        wrapper = MATLABEngineWrapper()
        wrapper.engine = mock_matlab_engine
        wrapper.state = SessionState.ACTIVE
        
        health = wrapper.health_check()
        
        assert 'session_id' in health
        assert 'healthy' in health
        assert 'uptime' in health
    
    def test_context_manager(self, mock_matlab_engine):
        """Test context manager functionality."""
        with MATLABEngineWrapper() as wrapper:
            assert wrapper.state == SessionState.ACTIVE
        
        # Should be closed after context
        assert wrapper.state == SessionState.CLOSED


class TestMATLABSessionManager:
    """Test MATLAB session manager functionality."""
    
    @pytest.fixture
    def mock_matlab_engine(self):
        """Create mock MATLAB engine for testing."""
        with patch('matlab_engine_wrapper.matlab.engine') as mock_engine:
            mock_instance = Mock()
            mock_engine.start_matlab.return_value = mock_instance
            mock_instance.eval.return_value = 2.0
            mock_instance.usejava.return_value = True
            mock_instance.workspace = {}
            mock_instance.quit = Mock()
            yield mock_instance
    
    def test_session_manager_initialization(self):
        """Test session manager initialization."""
        config = MATLABConfig(max_sessions=3)
        manager = MATLABSessionManager(config=config)
        
        assert manager.config.max_sessions == 3
        assert len(manager.sessions) == 0
    
    def test_get_or_create_session(self, mock_matlab_engine):
        """Test session creation and retrieval."""
        manager = MATLABSessionManager()
        
        # Create first session
        session1 = manager.get_or_create_session("session_1")
        assert session1 is not None
        assert len(manager.sessions) == 1
        
        # Get same session again
        session1_again = manager.get_or_create_session("session_1")
        assert session1 is session1_again
        assert len(manager.sessions) == 1
    
    def test_session_pool_limit(self, mock_matlab_engine):
        """Test session pool size limits."""
        config = MATLABConfig(max_sessions=2)
        manager = MATLABSessionManager(config=config)
        
        # Create maximum number of sessions
        session1 = manager.get_or_create_session("session_1")
        session2 = manager.get_or_create_session("session_2")
        
        assert len(manager.sessions) == 2
        
        # Creating another session should evict oldest
        session3 = manager.get_or_create_session("session_3")
        
        assert len(manager.sessions) == 2
        assert "session_3" in manager.sessions
    
    def test_session_cleanup(self, mock_matlab_engine):
        """Test automatic session cleanup."""
        config = MATLABConfig(session_timeout=1)  # 1 second timeout
        manager = MATLABSessionManager(config=config)
        
        # Create a session
        session = manager.get_or_create_session("test_session")
        session.session_start_time = time.time() - 2  # Make it expired
        
        # Run cleanup
        manager.cleanup_expired_sessions()
        
        assert len(manager.sessions) == 0
    
    def test_pool_status(self, mock_matlab_engine):
        """Test pool status reporting."""
        manager = MATLABSessionManager()
        
        # Create some sessions
        session1 = manager.get_or_create_session("session_1")
        session2 = manager.get_or_create_session("session_2")
        
        status = manager.get_pool_status()
        
        assert status['total_sessions'] == 2
        assert status['active_sessions'] >= 0
        assert 'pool_utilization' in status
        assert 'session_details' in status
    
    def test_session_manager_context(self, mock_matlab_engine):
        """Test session manager as context manager."""
        with MATLABSessionManager() as manager:
            session = manager.get_or_create_session("test")
            assert session.state == SessionState.ACTIVE
        
        # All sessions should be closed after context


class TestConfigurationManager:
    """Test configuration manager functionality."""
    
    def test_configuration_manager_initialization(self):
        """Test configuration manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigurationManager(config_dir=config_dir)
            
            assert manager.config_dir == config_dir
            assert manager._current_env == Environment.DEVELOPMENT
    
    def test_environment_configurations(self):
        """Test environment-specific configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigurationManager(config_dir=Path(temp_dir))
            
            # Test different environments
            prod_config = manager.get_configuration(Environment.PRODUCTION)
            dev_config = manager.get_configuration(Environment.DEVELOPMENT)
            
            # Production should have more restrictive settings
            assert prod_config.max_sessions >= dev_config.max_sessions
            assert '-nojvm' in prod_config.startup_options
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigurationManager(config_dir=Path(temp_dir))
            
            # Test valid configuration
            validation = manager.validate_configuration()
            assert validation['valid'] == True
            assert isinstance(validation['errors'], list)
            assert isinstance(validation['warnings'], list)
    
    def test_configuration_updates(self):
        """Test configuration updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigurationManager(config_dir=Path(temp_dir))
            
            # Update configuration
            updates = {
                "max_sessions": 10,
                "session_timeout": 600
            }
            manager.update_configuration(updates)
            
            # Verify updates
            config = manager.get_configuration()
            assert config.max_sessions == 10
            assert config.session_timeout == 600


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.fixture
    def mock_matlab_engine(self):
        """Create mock MATLAB engine for testing."""
        with patch('matlab_engine_wrapper.matlab.engine') as mock_engine:
            mock_instance = Mock()
            mock_engine.start_matlab.return_value = mock_instance
            mock_instance.eval.return_value = 8.0  # sqrt(64)
            mock_instance.sin.return_value = 1.0
            mock_instance.usejava.return_value = True
            mock_instance.workspace = {}
            mock_instance.quit = Mock()
            yield mock_instance
    
    def test_full_workflow_integration(self, mock_matlab_engine):
        """Test complete workflow with configuration and session management."""
        # Create configuration
        config = MATLABConfig(
            max_sessions=2,
            session_timeout=300,
            startup_options=['-nojvm'],
            performance_monitoring=True
        )
        
        # Create session manager
        with MATLABSessionManager(config=config) as manager:
            # Get session
            session = manager.get_or_create_session("integration_test")
            
            # Perform operations
            result1 = session.evaluate("sqrt(64)")
            assert result1 == 8.0
            
            result2 = session.call_function("sin", np.pi/2)
            assert result2 == 1.0
            
            # Test workspace operations
            test_array = np.array([1, 2, 3, 4, 5])
            session.set_workspace_variable("test_array", test_array)
            retrieved = session.get_workspace_variable("test_array")
            
            # Check performance stats
            stats = session.get_performance_stats()
            assert stats['total_operations'] >= 3
            
            # Check pool status
            pool_status = manager.get_pool_status()
            assert pool_status['total_sessions'] == 1
    
    def test_concurrent_sessions(self, mock_matlab_engine):
        """Test concurrent session handling."""
        config = MATLABConfig(max_sessions=3)
        
        def worker_function(session_id, results_dict):
            """Worker function for concurrent testing."""
            try:
                with MATLABSessionManager(config=config) as manager:
                    session = manager.get_or_create_session(f"concurrent_{session_id}")
                    result = session.evaluate("2+2")
                    results_dict[session_id] = result
            except Exception as e:
                results_dict[session_id] = f"Error: {e}"
        
        # Create multiple threads
        threads = []
        results = {}
        
        for i in range(3):
            thread = threading.Thread(
                target=worker_function, 
                args=(i, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 3
        for result in results.values():
            assert not str(result).startswith("Error:")


# Test runner
if __name__ == "__main__":
    # Run tests with pytest
    print("Running MATLAB Engine Wrapper Tests...")
    print("=" * 50)
    
    # Run specific test categories
    test_classes = [
        TestTypeConverter,
        TestMATLABConfig,
        TestMATLABEngineWrapper,
        TestMATLABSessionManager,
        TestConfigurationManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        # Note: In real usage, would use pytest.main() or run with pytest command
        
    print("\nTo run these tests properly, use:")
    print("pytest test_matlab_engine.py -v")
    print("\nOr run specific test classes:")
    print("pytest test_matlab_engine.py::TestTypeConverter -v")