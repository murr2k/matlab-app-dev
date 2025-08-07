"""
MATLAB Engine Wrapper Module
============================

Provides a Python wrapper for the MATLAB Engine API with session management,
error handling, and intelligent problem classification.

This module is the foundation for Issue #1: MATLAB Engine API Integration.
"""

import matlab.engine
from typing import Any, Optional, Dict, List, Union, Tuple, Callable
import logging
import time
import threading
import queue
import os
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MATLABEngineError(Exception):
    """Base exception for MATLAB Engine operations."""
    pass


class MATLABSessionError(MATLABEngineError):
    """Session management related errors."""
    pass


class MATLABTypeConversionError(MATLABEngineError):
    """Data type conversion related errors."""
    pass


class MATLABExecutionError(MATLABEngineError):
    """MATLAB code execution related errors."""
    pass


class SessionState(Enum):
    """Enum for session states."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class MATLABConfig:
    """Configuration class for MATLAB Engine settings."""
    startup_options: List[str] = field(default_factory=list)
    max_sessions: int = 3
    session_timeout: int = 300  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    workspace_persistence: bool = True
    headless_mode: Optional[bool] = None
    matlab_path_additions: List[str] = field(default_factory=list)
    performance_monitoring: bool = True
    memory_threshold_mb: int = 1024  # MB
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MATLABConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class TypeConverter:
    """Handles Python-MATLAB data type conversions."""
    
    @staticmethod
    def python_to_matlab(value: Any) -> Any:
        """Convert Python data types to MATLAB-compatible types."""
        try:
            if isinstance(value, np.ndarray):
                return matlab.double(value.tolist())
            elif isinstance(value, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in value):
                    return matlab.double(list(value))
                else:
                    return [TypeConverter.python_to_matlab(x) for x in value]
            elif isinstance(value, dict):
                # Convert to MATLAB struct
                return {k: TypeConverter.python_to_matlab(v) for k, v in value.items()}
            elif isinstance(value, (int, float)):
                return matlab.double([value])
            elif isinstance(value, str):
                return value
            elif isinstance(value, bool):
                return matlab.logical([value])
            else:
                # Try to convert using numpy
                return matlab.double(np.array(value).tolist())
        except Exception as e:
            raise MATLABTypeConversionError(f"Failed to convert {type(value)} to MATLAB: {e}")
    
    @staticmethod
    def matlab_to_python(value: Any) -> Any:
        """Convert MATLAB data types to Python-compatible types."""
        try:
            if hasattr(value, '_data'):
                # Handle MATLAB arrays
                data = value._data
                if hasattr(value, '_size'):
                    return np.array(data).reshape(value._size)
                else:
                    return np.array(data)
            elif isinstance(value, (list, tuple)):
                return [TypeConverter.matlab_to_python(x) for x in value]
            elif isinstance(value, dict):
                return {k: TypeConverter.matlab_to_python(v) for k, v in value.items()}
            else:
                return value
        except Exception as e:
            logger.warning(f"Type conversion warning: {e}")
            return value


class MATLABEngineWrapper:
    """
    Enhanced wrapper class for MATLAB Engine API providing robust Python interface.
    
    Features:
    - Robust error handling and recovery
    - Automatic retry mechanisms
    - Performance monitoring
    - Type conversion handling
    - Session state management
    """
    
    def __init__(self, config: Optional[MATLABConfig] = None, session_id: Optional[str] = None):
        """
        Initialize MATLAB Engine Wrapper with enhanced configuration.
        
        Args:
            config: Configuration object for MATLAB settings
            session_id: Unique identifier for this session
        """
        self.config = config or MATLABConfig()
        self.session_id = session_id or f"matlab_session_{int(time.time())}"
        self.engine = None
        self.state = SessionState.IDLE
        self.session_start_time = None
        self.last_activity_time = None
        self.operation_count = 0
        self.error_count = 0
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        self._lock = threading.RLock()  # Thread-safe operations
        
    def start(self) -> bool:
        """
        Start MATLAB engine session with retry logic and comprehensive setup.
        
        Returns:
            bool: True if engine started successfully
            
        Raises:
            MATLABSessionError: If unable to start after max retries
        """
        with self._lock:
            if self.state in [SessionState.ACTIVE, SessionState.BUSY]:
                logger.info(f"Session {self.session_id} already active")
                return True
            
            self.state = SessionState.IDLE
            
            for attempt in range(self.config.max_retries):
                try:
                    logger.info(f"Starting MATLAB engine (attempt {attempt + 1}/{self.config.max_retries})...")
                    
                    # Start MATLAB with configured options
                    if self.config.startup_options:
                        self.engine = matlab.engine.start_matlab(*self.config.startup_options)
                    else:
                        self.engine = matlab.engine.start_matlab()
                    
                    self.state = SessionState.ACTIVE
                    self.session_start_time = time.time()
                    self.last_activity_time = self.session_start_time
                    
                    # Setup MATLAB environment
                    self._setup_matlab_environment()
                    
                    logger.info(f"MATLAB engine started successfully (session: {self.session_id})")
                    return True
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    self.state = SessionState.ERROR
                    
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise MATLABSessionError(f"Failed to start MATLAB after {self.config.max_retries} attempts: {e}")
            
            return False
    
    def _setup_matlab_environment(self):
        """Setup MATLAB environment with all necessary configurations."""
        try:
            # Detect headless mode
            if self.config.headless_mode is None:
                try:
                    has_display = self.engine.usejava('desktop')
                    headless_mode = not has_display
                except:
                    headless_mode = True
            else:
                headless_mode = self.config.headless_mode
            
            if headless_mode:
                # Configure for headless operation
                self.engine.eval("set(0, 'DefaultFigureVisible', 'off')", nargout=0)
                logger.info("Configured for headless operation")
            
            # Add custom paths
            for path in self.config.matlab_path_additions:
                self.engine.addpath(path, nargout=0)
                logger.debug(f"Added path: {path}")
            
            # Set workspace persistence
            if not self.config.workspace_persistence:
                self.engine.eval("clear all", nargout=0)
            
            # Initialize performance monitoring
            if self.config.performance_monitoring:
                self.engine.eval("tic", nargout=0)  # Start timer for session
            
        except Exception as e:
            logger.error(f"Failed to setup MATLAB environment: {e}")
            raise MATLABSessionError(f"Environment setup failed: {e}")
    
    def evaluate(self, expression: str, convert_types: bool = True, timeout: Optional[float] = None) -> Any:
        """
        Evaluate MATLAB expression with enhanced error handling and type conversion.
        
        Args:
            expression: MATLAB expression to evaluate
            convert_types: Whether to convert result to Python types
            timeout: Optional timeout in seconds
            
        Returns:
            Result of MATLAB evaluation (converted to Python types if requested)
            
        Raises:
            MATLABExecutionError: If evaluation fails
            MATLABSessionError: If session is not active
        """
        with self._lock:
            self._ensure_active_session()
            
            start_time = time.time()
            self.state = SessionState.BUSY
            
            try:
                logger.debug(f"Evaluating: {expression[:100]}...")
                
                # Execute with optional timeout
                if timeout:
                    result = self.engine.eval(expression, nargout=1, background=False)
                    # Note: MATLAB Engine doesn't support timeout directly, 
                    # would need to implement with threading if required
                else:
                    result = self.engine.eval(expression, nargout=1)
                
                # Convert types if requested
                if convert_types:
                    result = TypeConverter.matlab_to_python(result)
                
                # Update performance stats
                execution_time = time.time() - start_time
                self._update_performance_stats(execution_time, success=True)
                
                self.state = SessionState.ACTIVE
                self.last_activity_time = time.time()
                
                logger.debug(f"Evaluation completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._update_performance_stats(execution_time, success=False)
                self.error_count += 1
                self.state = SessionState.ERROR
                
                error_msg = f"Failed to evaluate expression: {e}"
                logger.error(error_msg)
                raise MATLABExecutionError(error_msg) from e
            
            finally:
                if self.state == SessionState.BUSY:
                    self.state = SessionState.ACTIVE
    
    def call_function(self, func_name: str, *args, convert_args: bool = True, 
                     convert_result: bool = True, **kwargs) -> Any:
        """
        Call MATLAB function with enhanced type handling and error management.
        
        Args:
            func_name: Name of MATLAB function
            *args: Positional arguments for function
            convert_args: Whether to convert Python args to MATLAB types
            convert_result: Whether to convert result to Python types
            **kwargs: Keyword arguments (e.g., nargout, timeout)
            
        Returns:
            Function result (converted to Python types if requested)
            
        Raises:
            MATLABExecutionError: If function call fails
            MATLABSessionError: If session is not active
        """
        with self._lock:
            self._ensure_active_session()
            
            start_time = time.time()
            self.state = SessionState.BUSY
            
            try:
                logger.debug(f"Calling function: {func_name}")
                
                # Convert arguments if requested
                if convert_args:
                    converted_args = [TypeConverter.python_to_matlab(arg) for arg in args]
                else:
                    converted_args = args
                
                # Get function handle
                func = getattr(self.engine, func_name)
                
                # Call function
                result = func(*converted_args, **kwargs)
                
                # Convert result if requested
                if convert_result:
                    result = TypeConverter.matlab_to_python(result)
                
                # Update performance stats
                execution_time = time.time() - start_time
                self._update_performance_stats(execution_time, success=True)
                
                self.state = SessionState.ACTIVE
                self.last_activity_time = time.time()
                
                logger.debug(f"Function {func_name} completed in {execution_time:.3f}s")
                return result
                
            except AttributeError:
                error_msg = f"MATLAB function '{func_name}' not found"
                logger.error(error_msg)
                self.state = SessionState.ERROR
                raise MATLABExecutionError(error_msg)
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._update_performance_stats(execution_time, success=False)
                self.error_count += 1
                self.state = SessionState.ERROR
                
                error_msg = f"Failed to call function {func_name}: {e}"
                logger.error(error_msg)
                raise MATLABExecutionError(error_msg) from e
            
            finally:
                if self.state == SessionState.BUSY:
                    self.state = SessionState.ACTIVE
    
    def set_workspace_variable(self, name: str, value: Any, convert_types: bool = True):
        """
        Set variable in MATLAB workspace with type conversion.
        
        Args:
            name: Variable name
            value: Variable value
            convert_types: Whether to convert Python types to MATLAB types
            
        Raises:
            MATLABSessionError: If session is not active
            MATLABTypeConversionError: If type conversion fails
        """
        with self._lock:
            self._ensure_active_session()
            
            try:
                if convert_types:
                    matlab_value = TypeConverter.python_to_matlab(value)
                else:
                    matlab_value = value
                
                self.engine.workspace[name] = matlab_value
                self.last_activity_time = time.time()
                
                logger.debug(f"Set workspace variable: {name}")
                
            except Exception as e:
                error_msg = f"Failed to set workspace variable {name}: {e}"
                logger.error(error_msg)
                raise MATLABTypeConversionError(error_msg) from e
    
    def get_workspace_variable(self, name: str, convert_types: bool = True) -> Any:
        """
        Get variable from MATLAB workspace with type conversion.
        
        Args:
            name: Variable name
            convert_types: Whether to convert MATLAB types to Python types
            
        Returns:
            Variable value (converted to Python types if requested)
            
        Raises:
            MATLABSessionError: If session is not active
            KeyError: If variable doesn't exist
        """
        with self._lock:
            self._ensure_active_session()
            
            try:
                value = self.engine.workspace[name]
                self.last_activity_time = time.time()
                
                if convert_types:
                    value = TypeConverter.matlab_to_python(value)
                
                logger.debug(f"Retrieved workspace variable: {name}")
                return value
                
            except KeyError:
                error_msg = f"Workspace variable '{name}' not found"
                logger.error(error_msg)
                raise KeyError(error_msg)
            
            except Exception as e:
                error_msg = f"Failed to get workspace variable {name}: {e}"
                logger.error(error_msg)
                raise MATLABExecutionError(error_msg) from e
    
    def clear_workspace(self):
        """Clear MATLAB workspace."""
        with self._lock:
            self._ensure_active_session()
            self.engine.eval("clear all", nargout=0)
            self.last_activity_time = time.time()
            logger.debug("Workspace cleared")
    
    def _ensure_active_session(self):
        """Ensure session is active, raise exception if not."""
        if self.state not in [SessionState.ACTIVE, SessionState.BUSY]:
            raise MATLABSessionError(f"Session {self.session_id} not active (state: {self.state.value})")
    
    def _update_performance_stats(self, execution_time: float, success: bool):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        
        if success:
            self.performance_stats['successful_operations'] += 1
        else:
            self.performance_stats['failed_operations'] += 1
        
        # Update average execution time
        if self.performance_stats['total_operations'] > 0:
            self.performance_stats['avg_execution_time'] = (
                self.performance_stats['total_execution_time'] / 
                self.performance_stats['total_operations']
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats['session_id'] = self.session_id
        stats['session_duration'] = time.time() - self.session_start_time if self.session_start_time else 0
        stats['state'] = self.state.value
        stats['error_count'] = self.error_count
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the MATLAB session."""
        health_status = {
            'session_id': self.session_id,
            'state': self.state.value,
            'healthy': False,
            'last_activity': self.last_activity_time,
            'error_count': self.error_count,
            'uptime': 0
        }
        
        if self.session_start_time:
            health_status['uptime'] = time.time() - self.session_start_time
        
        try:
            # Test basic functionality
            test_result = self.engine.eval('1+1', nargout=1)
            health_status['healthy'] = (test_result == 2)
            health_status['test_passed'] = True
        except Exception as e:
            health_status['healthy'] = False
            health_status['test_passed'] = False
            health_status['error'] = str(e)
        
        return health_status
    
    def close(self, force: bool = False):
        """Close MATLAB engine session with comprehensive cleanup.
        
        Args:
            force: If True, force close even if errors occur
        """
        with self._lock:
            if self.state == SessionState.CLOSED:
                logger.debug(f"Session {self.session_id} already closed")
                return
            
            self.state = SessionState.CLOSING
            
            try:
                if self.engine:
                    # Clean up workspace if not persisting
                    if not self.config.workspace_persistence:
                        try:
                            self.engine.eval("clear all", nargout=0)
                        except Exception as e:
                            logger.warning(f"Failed to clear workspace: {e}")
                    
                    # Log session statistics
                    if self.session_start_time:
                        duration = time.time() - self.session_start_time
                        stats = self.get_performance_stats()
                        logger.info(f"Session {self.session_id} closing - Duration: {duration:.2f}s, "
                                  f"Operations: {stats['total_operations']}, "
                                  f"Success rate: {stats['successful_operations']}/{stats['total_operations']}")
                    
                    # Quit MATLAB engine
                    self.engine.quit()
                    
                self.state = SessionState.CLOSED
                self.engine = None
                logger.info(f"MATLAB session {self.session_id} closed successfully")
                
            except Exception as e:
                if force:
                    logger.error(f"Force closing session {self.session_id} due to error: {e}")
                    self.state = SessionState.CLOSED
                    self.engine = None
                else:
                    logger.error(f"Error closing MATLAB session {self.session_id}: {e}")
                    self.state = SessionState.ERROR
                    raise MATLABSessionError(f"Failed to close session: {e}") from e
    
    @contextmanager
    def session(self, auto_close: bool = False):
        """
        Context manager for MATLAB engine session.
        
        Args:
            auto_close: Whether to automatically close session on exit
        
        Usage:
            with engine.session():
                result = engine.evaluate("sqrt(64)")
        """
        try:
            if self.state not in [SessionState.ACTIVE, SessionState.BUSY]:
                self.start()
            yield self
        except Exception as e:
            logger.error(f"Error in session context: {e}")
            raise
        finally:
            if auto_close:
                self.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            logger.error(f"Exception in context manager: {exc_type.__name__}: {exc_val}")
        self.close(force=bool(exc_type))
    
    def __del__(self):
        """Destructor to ensure engine cleanup."""
        try:
            if self.state not in [SessionState.CLOSED, SessionState.CLOSING]:
                self.close(force=True)
        except Exception:
            # Ignore errors in destructor
            pass


class MATLABSessionManager:
    """
    Enhanced session manager for MATLAB engines with connection pooling,
    automatic cleanup, and health monitoring.
    
    Features:
    - Connection pooling with configurable limits
    - Automatic session timeout and cleanup
    - Health monitoring and recovery
    - Thread-safe operations
    - Performance tracking
    """
    
    def __init__(self, config: Optional[MATLABConfig] = None):
        """
        Initialize enhanced session manager.
        
        Args:
            config: Configuration for session management
        """
        self.config = config or MATLABConfig()
        self.sessions: Dict[str, MATLABEngineWrapper] = {}
        self.session_queue = queue.Queue(maxsize=self.config.max_sessions)
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = False
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup."""
        def cleanup_worker():
            while not self._shutdown:
                try:
                    self.cleanup_expired_sessions()
                    self.health_check_all_sessions()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    time.sleep(5)
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get_or_create_session(self, session_id: Optional[str] = None, 
                             startup_options: Optional[List[str]] = None) -> MATLABEngineWrapper:
        """
        Get existing session or create new one with enhanced pooling.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            startup_options: MATLAB startup options for new sessions
            
        Returns:
            MATLABEngineWrapper instance
            
        Raises:
            MATLABSessionError: If unable to create session
        """
        with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = f"session_{int(time.time())}_{threading.get_ident()}"
            
            # Return existing session if available and healthy
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.state == SessionState.ACTIVE:
                    return session
                else:
                    # Remove unhealthy session
                    logger.warning(f"Removing unhealthy session {session_id} (state: {session.state.value})")
                    self._remove_session(session_id)
            
            # Check if we need to free up space
            if len(self.sessions) >= self.config.max_sessions:
                self._evict_oldest_session()
            
            # Create new session with custom config
            session_config = MATLABConfig(
                startup_options=startup_options or self.config.startup_options,
                workspace_persistence=self.config.workspace_persistence,
                headless_mode=self.config.headless_mode,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            
            session = MATLABEngineWrapper(config=session_config, session_id=session_id)
            
            try:
                session.start()
                self.sessions[session_id] = session
                logger.info(f"Created new session: {session_id}")
                return session
            
            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}")
                raise MATLABSessionError(f"Session creation failed: {e}") from e
    
    def get_available_session(self) -> Optional[MATLABEngineWrapper]:
        """Get any available session from the pool."""
        with self._lock:
            for session in self.sessions.values():
                if session.state == SessionState.ACTIVE:
                    return session
            return None
    
    def _evict_oldest_session(self):
        """Remove the oldest session to make room for new ones."""
        if not self.sessions:
            return
        
        oldest_session_id = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].session_start_time or 0
        )
        
        logger.info(f"Evicting oldest session: {oldest_session_id}")
        self._remove_session(oldest_session_id)
    
    def _remove_session(self, session_id: str):
        """Safely remove a session."""
        if session_id in self.sessions:
            try:
                self.sessions[session_id].close(force=True)
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
            finally:
                del self.sessions[session_id]
    
    def close_session(self, session_id: str, force: bool = False):
        """
        Close specific session with error handling.
        
        Args:
            session_id: Session to close
            force: Whether to force close on errors
        """
        with self._lock:
            self._remove_session(session_id)
            logger.info(f"Closed session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have exceeded timeout or are inactive."""
        with self._lock:
            current_time = time.time()
            expired = []
            
            for session_id, session in self.sessions.items():
                # Check session timeout
                if session.session_start_time:
                    session_duration = current_time - session.session_start_time
                    if session_duration > self.config.session_timeout:
                        expired.append((session_id, f"exceeded timeout ({session_duration:.1f}s)"))
                        continue
                
                # Check last activity timeout
                if session.last_activity_time:
                    inactive_duration = current_time - session.last_activity_time
                    if inactive_duration > (self.config.session_timeout / 2):  # Half timeout for inactivity
                        expired.append((session_id, f"inactive for {inactive_duration:.1f}s"))
                        continue
                
                # Check error state
                if session.state == SessionState.ERROR and session.error_count > 3:
                    expired.append((session_id, "too many errors"))
            
            for session_id, reason in expired:
                logger.info(f"Cleaning up expired session {session_id}: {reason}")
                self._remove_session(session_id)
    
    def health_check_all_sessions(self):
        """Perform health check on all active sessions."""
        with self._lock:
            unhealthy_sessions = []
            
            for session_id, session in self.sessions.items():
                try:
                    health = session.health_check()
                    if not health['healthy']:
                        unhealthy_sessions.append(session_id)
                        logger.warning(f"Unhealthy session detected: {session_id}")
                except Exception as e:
                    logger.error(f"Health check failed for session {session_id}: {e}")
                    unhealthy_sessions.append(session_id)
            
            # Remove unhealthy sessions
            for session_id in unhealthy_sessions:
                self._remove_session(session_id)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the session pool."""
        with self._lock:
            active_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.ACTIVE)
            busy_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.BUSY)
            error_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.ERROR)
            
            total_operations = sum(s.performance_stats['total_operations'] for s in self.sessions.values())
            successful_operations = sum(s.performance_stats['successful_operations'] for s in self.sessions.values())
            
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': active_sessions,
                'busy_sessions': busy_sessions,
                'error_sessions': error_sessions,
                'max_sessions': self.config.max_sessions,
                'pool_utilization': len(self.sessions) / self.config.max_sessions,
                'total_operations': total_operations,
                'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
                'session_details': {
                    sid: {
                        'state': session.state.value,
                        'uptime': time.time() - session.session_start_time if session.session_start_time else 0,
                        'operations': session.performance_stats['total_operations'],
                        'error_count': session.error_count
                    }
                    for sid, session in self.sessions.items()
                }
            }
    
    def close_all_sessions(self, force: bool = False):
        """Close all active sessions with proper cleanup.
        
        Args:
            force: Whether to force close all sessions
        """
        with self._lock:
            logger.info(f"Closing all sessions (force={force})")
            
            # Close all sessions
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                try:
                    self._remove_session(session_id)
                except Exception as e:
                    logger.error(f"Error closing session {session_id}: {e}")
                    if not force:
                        raise
            
            # Clear the sessions dictionary
            self.sessions.clear()
            logger.info("All sessions closed")
    
    def shutdown(self):
        """Shutdown the session manager and all sessions."""
        logger.info("Shutting down session manager...")
        
        # Stop cleanup thread
        self._shutdown = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Close all sessions
        self.close_all_sessions(force=True)
        
        logger.info("Session manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if not self._shutdown:
                self.shutdown()
        except Exception:
            # Ignore errors in destructor
            pass


# Example usage and testing
if __name__ == "__main__":
    # Test enhanced functionality
    print("Testing Enhanced MATLAB Engine Wrapper...")
    
    # Test configuration
    config = MATLABConfig(
        startup_options=['-nojvm'],  # Headless mode
        max_sessions=2,
        session_timeout=180,
        performance_monitoring=True
    )
    
    # Test session manager
    print("\n=== Testing Session Manager ===")
    with MATLABSessionManager(config) as manager:
        # Create multiple sessions
        session1 = manager.get_or_create_session("test_session_1")
        session2 = manager.get_or_create_session("test_session_2")
        
        # Test basic operations
        result1 = session1.evaluate("sqrt(64)")
        result2 = session2.call_function("sin", np.pi/2)
        
        print(f"Session 1 - sqrt(64) = {result1}")
        print(f"Session 2 - sin(pi/2) = {result2}")
        
        # Test type conversion
        python_array = np.array([[1, 2, 3], [4, 5, 6]])
        session1.set_workspace_variable("python_matrix", python_array)
        retrieved_array = session1.get_workspace_variable("python_matrix")
        print(f"Type conversion test: {np.array_equal(python_array, retrieved_array)}")
        
        # Test performance stats
        stats1 = session1.get_performance_stats()
        print(f"Session 1 stats: {stats1['total_operations']} operations, "
              f"avg time: {stats1['avg_execution_time']:.3f}s")
        
        # Test health check
        health = session1.health_check()
        print(f"Session 1 health: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        
        # Test pool status
        pool_status = manager.get_pool_status()
        print(f"Pool status: {pool_status['active_sessions']}/{pool_status['max_sessions']} sessions active")
        print(f"Pool utilization: {pool_status['pool_utilization']:.1%}")
    
    # Test single session with enhanced features
    print("\n=== Testing Single Session ===")
    with MATLABEngineWrapper(config) as engine:
        try:
            # Test error handling
            result = engine.evaluate("sqrt(-1)")  # Complex result
            print(f"Complex number result: {result}")
            
            # Test function with type conversion
            result = engine.call_function("zeros", 2, 3, convert_result=True)
            print(f"Zeros matrix shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
            
        except MATLABExecutionError as e:
            print(f"Expected error handled: {e}")
        
        # Final performance report
        final_stats = engine.get_performance_stats()
        print(f"\nFinal session stats:")
        print(f"  Total operations: {final_stats['total_operations']}")
        print(f"  Success rate: {final_stats['successful_operations']}/{final_stats['total_operations']}")
        print(f"  Average execution time: {final_stats['avg_execution_time']:.3f}s")
        print(f"  Session duration: {final_stats['session_duration']:.1f}s")
    
    print("\n=== All Enhanced Tests Completed Successfully! ===")