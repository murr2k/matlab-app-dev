# Python-MATLAB Integration

This directory contains the Python implementation for MATLAB Engine API integration (Issue #1).

## Installation

1. **Install MATLAB Engine API for Python**:
   ```bash
   pip install matlabengine
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from matlab_engine_wrapper import MATLABEngineWrapper

# Create engine instance
engine = MATLABEngineWrapper()
engine.start()

# Execute MATLAB computations
result = engine.evaluate("sqrt(64)")
print(f"Result: {result}")  # Output: 8.0

# Call MATLAB functions
result = engine.call_function("sin", 3.14159/2)
print(f"sin(pi/2) = {result}")

# Clean up
engine.close()
```

## Using Context Manager

```python
from matlab_engine_wrapper import MATLABEngineWrapper

with MATLABEngineWrapper() as engine:
    result = engine.evaluate("2 + 2")
    print(f"Result: {result}")
    # Engine automatically closes when exiting context
```

## Session Management

```python
from matlab_engine_wrapper import MATLABSessionManager

# Create session manager
manager = MATLABSessionManager(max_sessions=3)

# Get or create session
session = manager.get_or_create_session("user_123")
result = session.evaluate("magic(3)")

# Clean up expired sessions
manager.cleanup_expired_sessions()

# Close all sessions
manager.close_all_sessions()
```

## Headless Mode Support

The wrapper automatically detects headless environments (CI/CD) and configures MATLAB accordingly:

```python
# For CI/CD environments
engine = MATLABEngineWrapper(startup_options=['-nodisplay', '-nojvm'])
engine.start()

# Figures will be created invisibly and can be saved
engine.evaluate("plot([1,2,3,4])")
engine.evaluate("saveas(gcf, 'plot.png')")
```

## Testing

Run the test suite:
```bash
pytest tests/test_matlab_engine.py -v
```

## Files

- `matlab_engine_wrapper.py` - Core wrapper class for MATLAB Engine API
- `hybrid_simulations.py` - Python-MATLAB simulation bridge (coming soon)
- `computational_interface.py` - Interface for matlab-computational-engineer (coming soon)
- `requirements.txt` - Python package dependencies
- `tests/` - Unit and integration tests

## Integration with Physics Simulations

Coming in Phase 1 implementation:
- Direct Python calls to physics simulations
- Hybrid Python-MATLAB workflows
- Performance benchmarking
- Async computation support