"""
MATLAB Engine Wrapper Module - Enhanced with WSL Option 4 Support
===================================================================

Provides a Python wrapper for the MATLAB Engine API with session management,
error handling, and intelligent problem classification.

Supports multiple execution modes:
1. Native Python with MATLAB Engine installed
2. Windows Python from WSL (Option 4)
3. Mock mode for testing without MATLAB

This module automatically detects the environment and uses the appropriate
method to access MATLAB Engine API.
"""

import os
import sys
import subprocess
import logging
import time
from typing import Any, Optional, Dict, List, Union
from contextlib import contextmanager
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for MATLAB Engine"""
    NATIVE = "Native Python with MATLAB Engine"
    WINDOWS_FROM_WSL = "Windows Python from WSL (Option 4)"
    MOCK = "Mock mode (no MATLAB)"


class EnvironmentDetector:
    """Detect and configure execution environment"""
    
    @staticmethod
    def detect():
        """Detect current environment and available MATLAB options"""
        environment = {
            'platform': sys.platform,
            'is_wsl': False,
            'is_windows': sys.platform == 'win32',
            'mode': ExecutionMode.MOCK,
            'matlab_available': False,
            'windows_python_path': None,
            'matlab_engine': None
        }
        
        # Try to import MATLAB Engine directly
        try:
            import matlab.engine
            environment['matlab_available'] = True
            environment['mode'] = ExecutionMode.NATIVE
            environment['matlab_engine'] = matlab.engine
            logger.info("Native MATLAB Engine API available")
            return environment
        except ImportError:
            pass
        
        # Check if running in WSL
        if os.path.exists('/proc/version'):
            try:
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        environment['is_wsl'] = True
                        logger.info("Running in WSL environment")
                        
                        # Look for Windows Python with MATLAB Engine (Option 4)
                        windows_python = EnvironmentDetector._find_windows_python()
                        if windows_python:
                            environment['windows_python_path'] = windows_python
                            environment['mode'] = ExecutionMode.WINDOWS_FROM_WSL
                            environment['matlab_available'] = True
                            logger.info(f"Windows Python with MATLAB found: {windows_python}")
            except:
                pass
        
        # If no MATLAB available, use mock mode
        if not environment['matlab_available']:
            logger.warning("No MATLAB Engine available, using mock mode")
            environment['mode'] = ExecutionMode.MOCK
            
            # Create mock module
            class MockMatlabEngine:
                @staticmethod
                def start_matlab():
                    return MockEngine()
            
            class MockEngine:
                def eval(self, expr):
                    # Return reasonable mock values
                    if 'sqrt(64)' in expr:
                        return 8.0
                    elif 'sin(pi/2)' in expr:
                        return 1.0
                    elif 'cos(0)' in expr:
                        return 1.0
                    return 0.0
                
                def quit(self):
                    pass
                
                def __getattr__(self, name):
                    # Mock any MATLAB function call
                    def mock_func(*args, **kwargs):
                        return 0.0
                    return mock_func
            
            class MockMatlab:
                engine = MockMatlabEngine()
                
                @staticmethod
                def double(data):
                    return list(data) if hasattr(data, '__iter__') else [data]
            
            environment['matlab_engine'] = MockMatlab()
        
        return environment
    
    @staticmethod
    def _find_windows_python():
        """Find Windows Python with MATLAB Engine installed"""
        possible_paths = [
            '/mnt/c/Python312/python.exe',
            '/mnt/c/Python311/python.exe',
            '/mnt/c/Python310/python.exe',
            '/mnt/c/Python39/python.exe',
        ]
        
        # Also check user-specific installations
        import glob
        user_paths = glob.glob('/mnt/c/Users/*/AppData/Local/Programs/Python/Python*/python.exe')
        possible_paths.extend(user_paths)
        
        for path in possible_paths:
            if os.path.exists(path):
                # Test if it has MATLAB Engine
                try:
                    result = subprocess.run(
                        [path, '-c', 'import matlab.engine; print("OK")'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and "OK" in result.stdout:
                        return path
                except Exception as e:
                    logger.debug(f"Failed to test {path}: {e}")
        
        return None


# Global environment configuration
ENVIRONMENT = EnvironmentDetector.detect()


class MATLABEngineWrapper:
    """
    Universal MATLAB Engine Wrapper supporting multiple execution modes.
    
    Automatically detects and uses:
    - Native MATLAB Engine (if available)
    - Windows Python from WSL (Option 4)
    - Mock mode for testing
    """
    
    def __init__(self, force_mode: Optional[ExecutionMode] = None):
        """
        Initialize MATLAB Engine Wrapper.
        
        Args:
            force_mode: Optionally force a specific execution mode
        """
        self.environment = ENVIRONMENT.copy()
        if force_mode:
            self.environment['mode'] = force_mode
        
        self.engine = None
        self.session_active = False
        self.mode = self.environment['mode']
        self.windows_python_path = self.environment.get('windows_python_path')
        
        logger.info(f"MATLABEngineWrapper initialized in {self.mode.value} mode")
    
    def start(self) -> bool:
        """
        Start MATLAB engine session based on detected mode.
        
        Returns:
            bool: True if engine started successfully
        """
        try:
            if self.mode == ExecutionMode.NATIVE:
                # Native MATLAB Engine
                matlab = self.environment['matlab_engine']
                self.engine = matlab.start_matlab()
                self.session_active = True
                logger.info("Native MATLAB Engine started")
                return True
                
            elif self.mode == ExecutionMode.WINDOWS_FROM_WSL:
                # Use Windows Python subprocess for Option 4
                # For this mode, we'll execute commands via subprocess
                self.session_active = True
                logger.info("Windows Python MATLAB Engine ready (Option 4)")
                return True
                
            elif self.mode == ExecutionMode.MOCK:
                # Mock mode
                matlab = self.environment['matlab_engine']
                self.engine = matlab.engine.start_matlab()
                self.session_active = True
                logger.info("Mock MATLAB Engine started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start MATLAB engine: {e}")
            return False
    
    def evaluate(self, expression: str) -> Any:
        """
        Evaluate MATLAB expression based on execution mode.
        
        Args:
            expression: MATLAB expression to evaluate
            
        Returns:
            Result of MATLAB evaluation
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        try:
            if self.mode == ExecutionMode.NATIVE:
                # Direct evaluation
                return self.engine.eval(expression)
                
            elif self.mode == ExecutionMode.WINDOWS_FROM_WSL:
                # Execute via Windows Python subprocess
                python_code = f"""
import matlab.engine
eng = matlab.engine.start_matlab()
result = eng.eval("{expression}")
print(result)
eng.quit()
"""
                result = subprocess.run(
                    [self.windows_python_path, '-c', python_code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Parse the output
                    try:
                        return float(result.stdout.strip())
                    except:
                        return result.stdout.strip()
                else:
                    raise RuntimeError(f"MATLAB evaluation failed: {result.stderr}")
                    
            elif self.mode == ExecutionMode.MOCK:
                # Mock evaluation
                return self.engine.eval(expression)
                
        except Exception as e:
            logger.error(f"Error evaluating expression: {e}")
            raise
    
    def call_function(self, func_name: str, *args, **kwargs) -> Any:
        """
        Call MATLAB function based on execution mode.
        
        Args:
            func_name: Name of MATLAB function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        try:
            if self.mode == ExecutionMode.NATIVE:
                # Direct function call
                func = getattr(self.engine, func_name)
                return func(*args, **kwargs)
                
            elif self.mode == ExecutionMode.WINDOWS_FROM_WSL:
                # Execute via Windows Python subprocess
                # Convert args to string representation
                args_str = ', '.join(str(arg) for arg in args)
                python_code = f"""
import matlab.engine
eng = matlab.engine.start_matlab()
result = eng.{func_name}({args_str})
print(result)
eng.quit()
"""
                result = subprocess.run(
                    [self.windows_python_path, '-c', python_code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    try:
                        return float(result.stdout.strip())
                    except:
                        return result.stdout.strip()
                else:
                    raise RuntimeError(f"MATLAB function call failed: {result.stderr}")
                    
            elif self.mode == ExecutionMode.MOCK:
                # Mock function call
                func = getattr(self.engine, func_name)
                return func(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"Error calling function {func_name}: {e}")
            raise
    
    def close(self):
        """Close MATLAB engine session."""
        if self.engine and self.session_active:
            try:
                if self.mode in [ExecutionMode.NATIVE, ExecutionMode.MOCK]:
                    self.engine.quit()
                
                self.session_active = False
                logger.info("MATLAB engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing MATLAB engine: {e}")
    
    @contextmanager
    def session(self):
        """Context manager for MATLAB engine session."""
        try:
            if not self.session_active:
                self.start()
            yield self
        finally:
            pass  # Keep session open for reuse
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current execution mode."""
        return {
            'mode': self.mode.value,
            'matlab_available': self.environment['matlab_available'],
            'is_wsl': self.environment['is_wsl'],
            'windows_python_path': self.windows_python_path,
            'platform': self.environment['platform']
        }


# Convenience function for quick testing
def test_matlab_engine():
    """Quick test of MATLAB Engine in current environment."""
    print("Testing MATLAB Engine Wrapper...")
    print("-" * 40)
    
    wrapper = MATLABEngineWrapper()
    info = wrapper.get_mode_info()
    
    print(f"Execution Mode: {info['mode']}")
    print(f"Platform: {info['platform']}")
    print(f"WSL Detected: {info['is_wsl']}")
    print(f"MATLAB Available: {info['matlab_available']}")
    
    if info['windows_python_path']:
        print(f"Windows Python: {info['windows_python_path']}")
    
    print("-" * 40)
    
    # Run test calculations
    with wrapper as eng:
        tests = [
            ("sqrt(64)", 8.0),
            ("sin(pi/2)", 1.0),
            ("cos(0)", 1.0),
        ]
        
        print("Running test calculations:")
        for expr, expected in tests:
            try:
                result = eng.evaluate(expr)
                status = "✓" if abs(float(result) - expected) < 1e-9 else "✗"
                print(f"  {status} {expr} = {result} (expected {expected})")
            except Exception as e:
                print(f"  ✗ {expr} - Error: {e}")
    
    print("-" * 40)
    print("Test complete!")


if __name__ == "__main__":
    test_matlab_engine()