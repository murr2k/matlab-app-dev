"""
MATLAB Engine Wrapper Module
============================

Provides a Python wrapper for the MATLAB Engine API with session management,
error handling, and intelligent problem classification.

This module is the foundation for Issue #1: MATLAB Engine API Integration.
"""

import matlab.engine
from typing import Any, Optional, Dict, List, Union
import logging
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MATLABEngineWrapper:
    """
    Wrapper class for MATLAB Engine API providing Python interface to MATLAB.
    
    Attributes:
        engine: MATLAB Engine instance
        session_active: Boolean indicating if session is active
        workspace_persistence: Whether to maintain workspace between calls
        headless_mode: Whether running in headless environment
    """
    
    def __init__(self, startup_options: Optional[List[str]] = None):
        """
        Initialize MATLAB Engine Wrapper.
        
        Args:
            startup_options: Optional MATLAB startup options (e.g., ['-nojvm', '-nodisplay'])
        """
        self.engine = None
        self.session_active = False
        self.startup_options = startup_options or []
        self.workspace_persistence = True
        self.headless_mode = False
        self.session_start_time = None
        
    def start(self) -> bool:
        """
        Start MATLAB engine session.
        
        Returns:
            bool: True if engine started successfully
        """
        try:
            logger.info("Starting MATLAB engine...")
            if self.startup_options:
                self.engine = matlab.engine.start_matlab(*self.startup_options)
            else:
                self.engine = matlab.engine.start_matlab()
            
            self.session_active = True
            self.session_start_time = time.time()
            
            # Detect headless mode
            self._detect_headless_mode()
            
            logger.info(f"MATLAB engine started successfully (headless={self.headless_mode})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MATLAB engine: {e}")
            return False
    
    def _detect_headless_mode(self):
        """Detect if running in headless environment."""
        try:
            has_display = self.engine.usejava('desktop')
            self.headless_mode = not has_display
            
            if self.headless_mode:
                # Configure for headless operation
                self.engine.eval("set(0, 'DefaultFigureVisible', 'off')", nargout=0)
                logger.info("Configured for headless operation")
        except:
            # Assume headless if detection fails
            self.headless_mode = True
    
    def evaluate(self, expression: str) -> Any:
        """
        Evaluate MATLAB expression.
        
        Args:
            expression: MATLAB expression to evaluate
            
        Returns:
            Result of MATLAB evaluation
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        try:
            result = self.engine.eval(expression)
            return result
        except Exception as e:
            logger.error(f"Error evaluating expression: {e}")
            raise
    
    def call_function(self, func_name: str, *args, **kwargs) -> Any:
        """
        Call MATLAB function.
        
        Args:
            func_name: Name of MATLAB function
            *args: Positional arguments for function
            **kwargs: Keyword arguments (e.g., nargout)
            
        Returns:
            Function result
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        try:
            func = getattr(self.engine, func_name)
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error calling function {func_name}: {e}")
            raise
    
    def set_workspace_variable(self, name: str, value: Any):
        """
        Set variable in MATLAB workspace.
        
        Args:
            name: Variable name
            value: Variable value
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        self.engine.workspace[name] = value
    
    def get_workspace_variable(self, name: str) -> Any:
        """
        Get variable from MATLAB workspace.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value
        """
        if not self.session_active:
            raise RuntimeError("MATLAB engine not started")
        
        return self.engine.workspace[name]
    
    def clear_workspace(self):
        """Clear MATLAB workspace."""
        if self.session_active:
            self.engine.eval("clear all", nargout=0)
    
    def close(self):
        """Close MATLAB engine session."""
        if self.engine and self.session_active:
            try:
                # Clean up workspace if not persisting
                if not self.workspace_persistence:
                    self.clear_workspace()
                
                # Calculate session duration
                if self.session_start_time:
                    duration = time.time() - self.session_start_time
                    logger.info(f"Session duration: {duration:.2f} seconds")
                
                self.engine.quit()
                self.session_active = False
                logger.info("MATLAB engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing MATLAB engine: {e}")
    
    @contextmanager
    def session(self):
        """
        Context manager for MATLAB engine session.
        
        Usage:
            with engine.session():
                result = engine.evaluate("sqrt(64)")
        """
        try:
            if not self.session_active:
                self.start()
            yield self
        finally:
            # Don't auto-close to allow session reuse
            pass
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure engine cleanup."""
        if self.session_active:
            self.close()


class MATLABSessionManager:
    """
    Manages multiple MATLAB engine sessions for concurrent operations.
    """
    
    def __init__(self, max_sessions: int = 3):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
        """
        self.sessions: Dict[str, MATLABEngineWrapper] = {}
        self.max_sessions = max_sessions
        self.session_timeout = 300  # 5 minutes
    
    def get_or_create_session(self, session_id: str) -> MATLABEngineWrapper:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            MATLABEngineWrapper instance
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session if at capacity
            oldest = min(self.sessions.items(), 
                        key=lambda x: x[1].session_start_time or 0)
            oldest[1].close()
            del self.sessions[oldest[0]]
        
        # Create new session
        session = MATLABEngineWrapper()
        session.start()
        self.sessions[session_id] = session
        return session
    
    def close_session(self, session_id: str):
        """
        Close specific session.
        
        Args:
            session_id: Session to close
        """
        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have exceeded timeout."""
        current_time = time.time()
        expired = []
        
        for session_id, session in self.sessions.items():
            if session.session_start_time:
                duration = current_time - session.session_start_time
                if duration > self.session_timeout:
                    expired.append(session_id)
        
        for session_id in expired:
            logger.info(f"Cleaning up expired session: {session_id}")
            self.close_session(session_id)
    
    def close_all_sessions(self):
        """Close all active sessions."""
        for session_id in list(self.sessions.keys()):
            self.close_session(session_id)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    print("Testing MATLAB Engine Wrapper...")
    
    # Test with context manager
    with MATLABEngineWrapper() as engine:
        # Test basic computation
        result = engine.evaluate("sqrt(64)")
        print(f"sqrt(64) = {result}")
        
        # Test function call
        result = engine.call_function("sin", 3.14159/2)
        print(f"sin(pi/2) = {result}")
        
        # Test workspace operations
        engine.set_workspace_variable("test_var", 42)
        value = engine.get_workspace_variable("test_var")
        print(f"Workspace variable test_var = {value}")
    
    print("All tests completed successfully!")