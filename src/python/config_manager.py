"""
Configuration Management System for MATLAB Engine API
====================================================

This module provides centralized configuration management for the MATLAB Engine
wrapper, supporting environment-specific configurations and runtime adjustments.

Author: Murray Kopit
License: MIT
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    CI = "ci"


@dataclass
class DatabaseConfig:
    """Database configuration for session persistence."""
    enabled: bool = False
    connection_string: Optional[str] = None
    session_timeout: int = 3600  # seconds
    cleanup_interval: int = 300   # seconds


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    enabled: bool = True
    metrics_collection_interval: int = 60  # seconds
    memory_threshold_mb: int = 1024
    cpu_threshold_percent: float = 80.0
    enable_profiling: bool = False
    profiling_output_dir: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_code_validation: bool = True
    allowed_functions: Optional[List[str]] = None
    blocked_functions: Optional[List[str]] = field(default_factory=lambda: [
        'system', 'dos', 'unix', 'winopen', 'web'
    ])
    max_execution_time: int = 300  # seconds
    enable_sandbox: bool = False


@dataclass
class MATLABEngineConfig:
    """Comprehensive MATLAB Engine configuration."""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Basic MATLAB settings
    startup_options: List[str] = field(default_factory=list)
    matlab_root: Optional[str] = None
    matlab_path_additions: List[str] = field(default_factory=list)
    
    # Session management
    max_sessions: int = 3
    session_timeout: int = 300
    session_idle_timeout: int = 150
    max_retries: int = 3
    retry_delay: float = 1.0
    workspace_persistence: bool = True
    
    # Performance settings
    headless_mode: Optional[bool] = None
    performance_monitoring: bool = True
    enable_background_cleanup: bool = True
    cleanup_interval: int = 30
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MATLABEngineConfig':
        """Create configuration from dictionary."""
        # Handle enum conversion
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = Environment(data['environment'])
        
        # Handle nested configurations
        if 'database' in data and isinstance(data['database'], dict):
            data['database'] = DatabaseConfig(**data['database'])
        
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        
        if 'performance' in data and isinstance(data['performance'], dict):
            data['performance'] = PerformanceConfig(**data['performance'])
        
        if 'security' in data and isinstance(data['security'], dict):
            data['security'] = SecurityConfig(**data['security'])
        
        return cls(**data)


class ConfigurationManager:
    """
    Centralized configuration management for MATLAB Engine API.
    
    Features:
    - Environment-specific configurations
    - Runtime configuration updates
    - Configuration validation
    - Default configuration generation
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path.home() / ".matlab_engine"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._configs: Dict[Environment, MATLABEngineConfig] = {}
        self._current_env = Environment.DEVELOPMENT
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all environment configurations."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.json"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    config = MATLABEngineConfig.from_dict(config_data)
                    config.environment = env
                    self._configs[env] = config
                    
                    logger.info(f"Loaded configuration for {env.value}")
                
                except Exception as e:
                    logger.error(f"Failed to load configuration for {env.value}: {e}")
                    self._configs[env] = self._get_default_config(env)
            else:
                # Create default configuration
                self._configs[env] = self._get_default_config(env)
                self.save_configuration(env)
    
    def _get_default_config(self, env: Environment) -> MATLABEngineConfig:
        """Get default configuration for environment."""
        config = MATLABEngineConfig(environment=env)
        
        # Environment-specific defaults
        if env == Environment.PRODUCTION:
            config.startup_options = ['-nojvm', '-nodisplay']
            config.max_sessions = 10
            config.session_timeout = 600
            config.headless_mode = True
            config.logging.level = "WARNING"
            config.security.enable_code_validation = True
            config.security.enable_sandbox = True
            
        elif env == Environment.TESTING:
            config.startup_options = ['-nojvm']
            config.max_sessions = 2
            config.session_timeout = 120
            config.logging.level = "DEBUG"
            config.performance.enable_profiling = True
            
        elif env == Environment.CI:
            config.startup_options = ['-nojvm', '-nodisplay', '-batch']
            config.max_sessions = 1
            config.session_timeout = 300
            config.headless_mode = True
            config.logging.level = "INFO"
            config.performance.enabled = False
            
        else:  # DEVELOPMENT
            config.startup_options = []
            config.max_sessions = 3
            config.session_timeout = 300
            config.logging.level = "DEBUG"
            config.performance.enable_profiling = False
        
        return config
    
    def get_configuration(self, env: Optional[Environment] = None) -> MATLABEngineConfig:
        """
        Get configuration for specified environment.
        
        Args:
            env: Environment to get configuration for (current if None)
            
        Returns:
            Configuration object
        """
        env = env or self._current_env
        return self._configs.get(env, self._get_default_config(env))
    
    def set_environment(self, env: Environment):
        """Set current environment."""
        self._current_env = env
        logger.info(f"Environment set to: {env.value}")
    
    def get_current_environment(self) -> Environment:
        """Get current environment."""
        return self._current_env
    
    def save_configuration(self, env: Optional[Environment] = None):
        """
        Save configuration to file.
        
        Args:
            env: Environment to save (current if None)
        """
        env = env or self._current_env
        config = self._configs.get(env)
        
        if not config:
            logger.error(f"No configuration found for {env.value}")
            return
        
        config_file = self.config_dir / f"{env.value}.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Configuration saved for {env.value}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration for {env.value}: {e}")
    
    def update_configuration(self, updates: Dict[str, Any], 
                           env: Optional[Environment] = None):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            env: Environment to update (current if None)
        """
        env = env or self._current_env
        config = self._configs.get(env)
        
        if not config:
            logger.error(f"No configuration found for {env.value}")
            return
        
        try:
            # Apply updates to configuration
            config_dict = config.to_dict()
            self._deep_update(config_dict, updates)
            
            # Recreate configuration object
            self._configs[env] = MATLABEngineConfig.from_dict(config_dict)
            
            logger.info(f"Configuration updated for {env.value}")
            
        except Exception as e:
            logger.error(f"Failed to update configuration for {env.value}: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_configuration(self, env: Optional[Environment] = None) -> Dict[str, Any]:
        """
        Validate configuration and return validation results.
        
        Args:
            env: Environment to validate (current if None)
            
        Returns:
            Validation results dictionary
        """
        env = env or self._current_env
        config = self._configs.get(env)
        
        if not config:
            return {"valid": False, "errors": ["Configuration not found"]}
        
        errors = []
        warnings = []
        
        # Validate basic settings
        if config.max_sessions < 1:
            errors.append("max_sessions must be at least 1")
        
        if config.session_timeout < 30:
            warnings.append("session_timeout is very low (< 30s)")
        
        if config.max_retries < 1:
            errors.append("max_retries must be at least 1")
        
        # Validate MATLAB paths
        for path in config.matlab_path_additions:
            if not Path(path).exists():
                warnings.append(f"MATLAB path does not exist: {path}")
        
        # Validate security settings
        if config.security.enable_sandbox and not config.headless_mode:
            warnings.append("Sandbox mode recommended with headless mode")
        
        # Validate performance settings
        if config.performance.memory_threshold_mb < 512:
            warnings.append("Memory threshold is very low (< 512MB)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def create_sample_configs(self):
        """Create sample configuration files for all environments."""
        for env in Environment:
            config = self._get_default_config(env)
            self._configs[env] = config
            self.save_configuration(env)
        
        logger.info("Sample configurations created")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        summary = {
            "current_environment": self._current_env.value,
            "config_directory": str(self.config_dir),
            "environments": {}
        }
        
        for env, config in self._configs.items():
            validation = self.validate_configuration(env)
            summary["environments"][env.value] = {
                "valid": validation["valid"],
                "max_sessions": config.max_sessions,
                "session_timeout": config.session_timeout,
                "headless_mode": config.headless_mode,
                "performance_monitoring": config.performance_monitoring,
                "security_enabled": config.security.enable_code_validation
            }
        
        return summary


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_current_config() -> MATLABEngineConfig:
    """Get current environment configuration."""
    return get_config_manager().get_configuration()


def set_environment(env: Environment):
    """Set current environment."""
    get_config_manager().set_environment(env)


# Auto-detect environment from environment variables
def _auto_detect_environment() -> Environment:
    """Auto-detect environment from environment variables."""
    env_var = os.getenv('MATLAB_ENGINE_ENV', '').lower()
    ci_vars = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'JENKINS_URL']
    
    if env_var in ['prod', 'production']:
        return Environment.PRODUCTION
    elif env_var in ['test', 'testing']:
        return Environment.TESTING
    elif any(os.getenv(var) for var in ci_vars):
        return Environment.CI
    else:
        return Environment.DEVELOPMENT


# Initialize with auto-detected environment
def initialize_config():
    """Initialize configuration with auto-detected environment."""
    env = _auto_detect_environment()
    get_config_manager().set_environment(env)
    logger.info(f"Auto-detected environment: {env.value}")


# Initialize on module import
initialize_config()


if __name__ == "__main__":
    # Demo configuration management
    print("MATLAB Engine Configuration Management Demo")
    print("=" * 50)
    
    manager = get_config_manager()
    
    # Show current configuration
    current_config = manager.get_configuration()
    print(f"Current environment: {current_config.environment.value}")
    print(f"Max sessions: {current_config.max_sessions}")
    print(f"Session timeout: {current_config.session_timeout}s")
    print(f"Headless mode: {current_config.headless_mode}")
    
    # Validate configuration
    validation = manager.validate_configuration()
    print(f"\nConfiguration valid: {validation['valid']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Show configuration summary
    summary = manager.get_config_summary()
    print(f"\nConfiguration Summary:")
    for env_name, env_config in summary['environments'].items():
        print(f"  {env_name}: {env_config['max_sessions']} sessions, "
              f"timeout: {env_config['session_timeout']}s")
    
    print("\nConfiguration management demo completed!")