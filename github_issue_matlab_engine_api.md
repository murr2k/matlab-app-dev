# Enhancement: Integrate MATLAB Engine API for Python

## Summary
Integrate the MATLAB Engine API for Python to enable direct Python-MATLAB computational communication within the physics simulation framework. This will replace potential REST API approaches with native Python bindings and provide seamless access to MATLAB's computational engine.

## Background
Currently verified that MATLAB Engine API for Python works on this system:
```python
import matlab.engine
eng = matlab.engine.start_matlab()
result = eng.sqrt(64.0)  # Returns 8.0
```

## Technical Specifications

### MATLAB Engine API for Python (2025)
- **Latest Version**: 25.1.2 (Released January 22, 2025)
- **MATLAB Compatibility**: R2025a recommended
- **Python Support**: 3.9, 3.10, 3.11, or 3.12 (64-bit required)
- **Installation**: `pip install matlabengine`

### Core API Components

#### Primary Classes
- `matlab.engine.MatlabEngine` - Main computational interface
- `matlab.engine.FutureResult` - Asynchronous call results

#### Key Methods
- `matlab.engine.start_matlab()` - Start new MATLAB session
- `matlab.engine.find_matlab()` - Find existing shared sessions
- `matlab.engine.connect_matlab()` - Connect to shared session
- `eng.eval(code_string)` - Execute MATLAB code directly
- `eng.workspace` - Access MATLAB workspace variables
- `eng.quit()` - Close MATLAB session

#### Supported Data Types
- Numeric: `matlab.double`, `matlab.single`
- Integers: `matlab.int8` through `matlab.int64`
- Unsigned: `matlab.uint8` through `matlab.uint64`
- Other: `matlab.logical`, `matlab.object`

## Implementation Plan

### Phase 1: Core Integration
1. **Create Python wrapper module** (`src/python/matlab_engine_wrapper.py`)
   ```python
   import matlab.engine
   
   class MATLABEngineWrapper:
       def __init__(self):
           self.engine = None
       
       def start(self):
           self.engine = matlab.engine.start_matlab()
       
       def evaluate(self, expression):
           return self.engine.eval(expression)
       
       def call_function(self, func_name, *args):
           return getattr(self.engine, func_name)(*args)
       
       def close(self):
           if self.engine:
               self.engine.quit()
   ```

2. **Integration with existing simulations**
   - Modify physics simulations to optionally use Python-MATLAB bridge
   - Enable hybrid Python-MATLAB computational workflows

### Phase 2: Advanced Features
1. **Asynchronous computation support**
   ```python
   future_result = eng.sqrt(64.0, async=True)
   result = future_result.result()  # Non-blocking
   ```

2. **Workspace management**
   ```python
   # Set MATLAB variables from Python
   eng.workspace['myVar'] = matlab.double([[1, 2, 3]])
   
   # Get MATLAB variables in Python
   result = eng.workspace['computedResult']
   ```

3. **Session sharing and persistence**
   ```python
   # Share MATLAB session across Python processes
   eng = matlab.engine.start_matlab('-desktop')
   session_name = eng.matlab.engine.shareEngine()
   
   # Connect from another Python process
   shared_eng = matlab.engine.connect_matlab(session_name)
   ```

### Phase 3: Enhanced matlab-computational-engineer Agent
1. **Direct Python-MATLAB communication** instead of REST API
2. **Improved performance** with native bindings
3. **Better error handling** and session management
4. **Integration with existing CI/CD pipeline**

## Benefits

### Technical Advantages
- **Native Performance**: Direct Python-MATLAB bindings eliminate HTTP overhead
- **Type Safety**: Proper data type conversion between Python and MATLAB
- **Session Persistence**: Maintain MATLAB workspace across multiple computations
- **Error Propagation**: Native exception handling from MATLAB to Python

### Project Integration
- **Seamless Workflow**: Combine Python's ecosystem with MATLAB's computational power
- **Enhanced Testing**: Python unittest framework can directly test MATLAB functions
- **CI/CD Compatibility**: Python-based testing integrates better with GitHub Actions
- **Flexible Deployment**: Choose between pure MATLAB or hybrid Python-MATLAB approaches

## Implementation Requirements

### Dependencies
```bash
# Python packages
pip install matlabengine numpy scipy

# MATLAB requirements
# - Full MATLAB installation (not just Runtime)
# - MATLAB R2025a or compatible version
# - 64-bit architecture match with Python
```

### File Structure
```
src/python/
├── matlab_engine_wrapper.py    # Core wrapper class
├── hybrid_simulations.py       # Python-MATLAB simulation bridge  
├── computational_interface.py  # Interface for matlab-computational-engineer
└── tests/
    ├── test_matlab_engine.py   # Unit tests for MATLAB integration
    └── test_hybrid_workflows.py # Integration tests
```

### Configuration
```python
# config/matlab_engine_config.py
MATLAB_ENGINE_CONFIG = {
    'startup_options': ['-nojvm', '-nodisplay'],  # For headless CI
    'session_timeout': 300,  # 5 minutes
    'async_enabled': True,
    'workspace_persistence': True
}
```

## Testing Strategy

### Unit Tests
- Test MATLAB Engine startup/shutdown
- Verify data type conversions
- Test function call mechanisms
- Validate error handling

### Integration Tests  
- Test with existing physics simulations
- Verify CI/CD pipeline compatibility
- Test session sharing scenarios
- Performance benchmarking vs pure MATLAB

### CI/CD Integration
```yaml
# .github/workflows/python-matlab-tests.yml
- name: Test MATLAB Engine API
  run: |
    python -m pytest src/python/tests/ -v
    python -c "import matlab.engine; eng = matlab.engine.start_matlab(); print('MATLAB Engine OK')"
```

## Success Metrics
- [ ] Python can start/stop MATLAB engine reliably
- [ ] All existing MATLAB simulations work through Python interface
- [ ] CI/CD pipeline passes with Python-MATLAB integration
- [ ] Performance meets or exceeds current pure MATLAB approach
- [ ] matlab-computational-engineer agent uses Python interface successfully

## Documentation Updates Required
- Update README with Python-MATLAB setup instructions
- Add API documentation for Python wrapper classes
- Update wiki with hybrid workflow examples
- Document troubleshooting for Python-MATLAB integration

## Risk Mitigation
- **MATLAB License**: Ensure sufficient MATLAB licenses for concurrent Python sessions
- **Memory Management**: Properly handle MATLAB workspace cleanup
- **Version Compatibility**: Test across MATLAB and Python version combinations
- **CI/CD Stability**: Implement robust session management for automated testing

## References
- [MATLAB Engine API for Python Documentation](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
- [Installation Guide](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- [API Reference](https://www.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html)
- [PyPI Package](https://pypi.org/project/matlabengine/)

---
**Labels**: `enhancement`, `matlab`, `python`, `integration`, `computational-engine`
**Milestone**: v2.0 - Python-MATLAB Integration
**Assignee**: @murr2k
**Priority**: High