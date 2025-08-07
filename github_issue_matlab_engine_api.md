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

### Mathematical Validation Pipeline

#### Test Categories
1. **Basic Arithmetic & Algebra**
   - Verify fundamental operations
   - Test equation solving
   - Validate matrix operations

2. **Calculus & Analysis**
   - Differentiation and integration
   - Limits and series
   - Differential equations

3. **Linear Algebra**
   - Matrix decompositions
   - Eigenvalue problems
   - System solving

4. **Statistical Computations**
   - Distribution functions
   - Statistical tests
   - Random number generation

5. **Signal Processing**
   - FFT operations
   - Filter design
   - Convolution

#### Mathematical Test Suite Implementation
```python
# src/python/tests/test_mathematical_validation.py
import pytest
import matlab.engine
import numpy as np
from math import isclose, pi, e
import json

class TestMathematicalValidation:
    """
    Comprehensive mathematical validation tests comparing Python calculations
    with MATLAB Engine API results.
    """
    
    @classmethod
    def setup_class(cls):
        """Initialize MATLAB engine for all tests."""
        cls.eng = matlab.engine.start_matlab()
        
    @classmethod
    def teardown_class(cls):
        """Clean up MATLAB engine."""
        cls.eng.quit()
    
    # ============= BASIC ARITHMETIC TESTS =============
    
    def test_basic_arithmetic(self):
        """Test fundamental arithmetic operations."""
        test_cases = [
            {"expr": "2 + 2", "expected": 4.0},
            {"expr": "10 - 7", "expected": 3.0},
            {"expr": "6 * 7", "expected": 42.0},
            {"expr": "15 / 3", "expected": 5.0},
            {"expr": "2^8", "expected": 256.0},
            {"expr": "sqrt(64)", "expected": 8.0},
            {"expr": "exp(1)", "expected": e},
            {"expr": "log(exp(5))", "expected": 5.0},
        ]
        
        for case in test_cases:
            result = self.eng.eval(case["expr"])
            assert isclose(result, case["expected"], rel_tol=1e-9), \
                f"Failed: {case['expr']} = {result}, expected {case['expected']}"
    
    # ============= TRIGONOMETRIC TESTS =============
    
    def test_trigonometric_functions(self):
        """Test trigonometric calculations."""
        test_cases = [
            {"expr": "sin(pi/2)", "expected": 1.0},
            {"expr": "cos(0)", "expected": 1.0},
            {"expr": "tan(pi/4)", "expected": 1.0},
            {"expr": "asin(1)", "expected": pi/2},
            {"expr": "acos(0)", "expected": pi/2},
            {"expr": "atan(1)", "expected": pi/4},
            {"expr": "sinh(0)", "expected": 0.0},
            {"expr": "cosh(0)", "expected": 1.0},
        ]
        
        for case in test_cases:
            result = self.eng.eval(case["expr"])
            assert isclose(result, case["expected"], rel_tol=1e-9), \
                f"Failed: {case['expr']} = {result}, expected {case['expected']}"
    
    # ============= MATRIX OPERATIONS TESTS =============
    
    def test_matrix_operations(self):
        """Test matrix calculations and linear algebra."""
        
        # Matrix multiplication
        A = matlab.double([[1, 2], [3, 4]])
        B = matlab.double([[5, 6], [7, 8]])
        result = self.eng.mtimes(A, B)
        expected = [[19, 22], [43, 50]]
        assert np.allclose(result, expected), "Matrix multiplication failed"
        
        # Determinant
        det_result = self.eng.det(A)
        assert isclose(det_result, -2.0), f"Determinant failed: {det_result}"
        
        # Inverse
        inv_result = self.eng.inv(A)
        identity = self.eng.mtimes(A, inv_result)
        expected_identity = [[1, 0], [0, 1]]
        assert np.allclose(identity, expected_identity, atol=1e-10), "Matrix inverse failed"
        
        # Eigenvalues
        eigenvals = self.eng.eig(A, nargout=1)
        expected_eigenvals = sorted([5.372281323, -0.372281323])
        actual_eigenvals = sorted([float(e) for e in eigenvals])
        assert np.allclose(actual_eigenvals, expected_eigenvals, rtol=1e-6), \
            f"Eigenvalues failed: {actual_eigenvals}"
    
    # ============= EQUATION SOLVING TESTS =============
    
    def test_equation_solving(self):
        """Test equation solving capabilities."""
        
        # Linear system: Ax = b
        A = matlab.double([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
        b = matlab.double([[1], [-2], [0]])
        x = self.eng.mldivide(A, b)  # A\b in MATLAB
        
        # Verify solution
        verification = self.eng.mtimes(A, x)
        assert np.allclose(verification, b), "Linear system solution failed"
        
        # Polynomial roots
        coeffs = matlab.double([1, -6, 11, -6])  # x^3 - 6x^2 + 11x - 6
        roots = self.eng.roots(coeffs)
        expected_roots = [1.0, 2.0, 3.0]
        actual_roots = sorted([float(r) for r in roots])
        assert np.allclose(actual_roots, expected_roots), \
            f"Polynomial roots failed: {actual_roots}"
    
    # ============= CALCULUS TESTS =============
    
    def test_numerical_calculus(self):
        """Test numerical differentiation and integration."""
        
        # Numerical integration using trapz
        x = self.eng.linspace(0, float(pi), 1000)
        y = self.eng.sin(x)
        integral = self.eng.trapz(y, x)
        expected = 2.0  # ∫sin(x)dx from 0 to π = 2
        assert isclose(integral, expected, rel_tol=1e-3), \
            f"Integration failed: {integral}, expected {expected}"
        
        # Numerical differentiation using diff
        x_vals = matlab.double([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        y_vals = self.eng.power(x_vals, 2)  # y = x^2
        dy = self.eng.diff(y_vals)
        dx = 0.1
        derivative = [d/dx for d in dy[0]]
        expected_derivative = [0.1, 0.3, 0.5, 0.7, 0.9]  # dy/dx = 2x
        assert np.allclose(derivative, expected_derivative, atol=1e-10), \
            f"Differentiation failed: {derivative}"
    
    # ============= STATISTICS TESTS =============
    
    def test_statistical_functions(self):
        """Test statistical computations."""
        
        data = matlab.double([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Mean
        mean_val = self.eng.mean(data)
        assert isclose(mean_val, 5.5), f"Mean failed: {mean_val}"
        
        # Standard deviation
        std_val = self.eng.std(data)
        expected_std = 3.02765035409749  # Population std dev
        assert isclose(std_val, expected_std, rel_tol=1e-6), f"Std dev failed: {std_val}"
        
        # Median
        median_val = self.eng.median(data)
        assert isclose(median_val, 5.5), f"Median failed: {median_val}"
        
        # Correlation coefficient
        x = matlab.double([1, 2, 3, 4, 5])
        y = matlab.double([2, 4, 6, 8, 10])
        corr_matrix = self.eng.corrcoef(x, y)
        corr_coeff = corr_matrix[0][1]
        assert isclose(corr_coeff, 1.0), f"Correlation failed: {corr_coeff}"
    
    # ============= COMPLEX NUMBER TESTS =============
    
    def test_complex_numbers(self):
        """Test complex number operations."""
        
        # Complex arithmetic
        z1 = complex(3, 4)  # 3 + 4i
        z2 = complex(1, -2)  # 1 - 2i
        
        # Magnitude
        mag = self.eng.abs(z1)
        assert isclose(mag, 5.0), f"Complex magnitude failed: {mag}"
        
        # Phase
        phase = self.eng.angle(z1)
        expected_phase = 0.92729521800161
        assert isclose(phase, expected_phase, rel_tol=1e-6), f"Complex phase failed: {phase}"
        
        # Complex exponential (Euler's formula)
        result = self.eng.eval("exp(1i*pi)")
        # exp(iπ) = -1
        assert isclose(result.real, -1.0, abs_tol=1e-10), "Euler's formula real part failed"
        assert isclose(result.imag, 0.0, abs_tol=1e-10), "Euler's formula imaginary part failed"
    
    # ============= FFT TESTS =============
    
    def test_fft_operations(self):
        """Test Fast Fourier Transform operations."""
        
        # Create a signal with known frequency components
        fs = 1000  # Sampling frequency
        t = self.eng.linspace(0, 1, fs)
        
        # Signal with 50Hz and 120Hz components
        signal = self.eng.eval("sin(2*pi*50*[0:999]/1000) + 0.5*sin(2*pi*120*[0:999]/1000)")
        
        # Compute FFT
        fft_result = self.eng.fft(signal)
        fft_magnitude = self.eng.abs(fft_result)
        
        # Find peaks (should be at 50Hz and 120Hz)
        half_spectrum = fft_magnitude[0][:500]  # First half of spectrum
        
        # Check that peaks exist at expected frequencies
        peak_50hz = half_spectrum[50]
        peak_120hz = half_spectrum[120]
        
        assert peak_50hz > 400, f"50Hz peak not found: {peak_50hz}"
        assert peak_120hz > 200, f"120Hz peak not found: {peak_120hz}"
```

#### Pipeline Test Configuration
```yaml
# src/python/tests/pipeline_config.yaml
test_suites:
  basic_validation:
    description: "Fundamental mathematical operations"
    tests:
      - arithmetic
      - trigonometry
      - algebra
    tolerance: 1e-9
    
  advanced_validation:
    description: "Complex mathematical computations"
    tests:
      - matrix_operations
      - equation_solving
      - calculus
      - statistics
    tolerance: 1e-6
    
  domain_specific:
    description: "Specialized domain tests"
    tests:
      - signal_processing
      - complex_analysis
      - optimization
    tolerance: 1e-4

expected_results:
  # Pre-computed expected values for validation
  golden_values:
    sqrt_2: 1.4142135623730951
    pi: 3.141592653589793
    e: 2.718281828459045
    golden_ratio: 1.618033988749895
    
performance_benchmarks:
  startup_time: 5.0  # seconds
  computation_time:
    simple: 0.1     # seconds
    complex: 1.0    # seconds
    heavy: 10.0     # seconds
```

#### Continuous Validation Pipeline
```python
# src/python/tests/test_pipeline_validation.py
import pytest
import yaml
import time
import json
from pathlib import Path
from typing import Dict, List, Any

class MATLABValidationPipeline:
    """
    Automated pipeline for validating MATLAB Engine API computations
    against known mathematical results.
    """
    
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        """Initialize validation pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.results = {"tests": [], "summary": {}}
        self.engine = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_validation_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a complete validation suite."""
        suite = self.config['test_suites'][suite_name]
        results = {
            'suite': suite_name,
            'description': suite['description'],
            'tests': [],
            'passed': 0,
            'failed': 0,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        for test_category in suite['tests']:
            test_result = self._run_test_category(test_category, suite['tolerance'])
            results['tests'].append(test_result)
            
            if test_result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        results['execution_time'] = time.time() - start_time
        results['success_rate'] = results['passed'] / len(suite['tests']) * 100
        
        return results
    
    def _run_test_category(self, category: str, tolerance: float) -> Dict:
        """Run tests for a specific category."""
        # Implementation would run specific test methods
        pass
    
    def validate_against_golden_values(self) -> List[Dict]:
        """Validate computations against pre-computed golden values."""
        golden = self.config['expected_results']['golden_values']
        validations = []
        
        for name, expected in golden.items():
            if name == 'sqrt_2':
                actual = self.engine.eval('sqrt(2)')
            elif name == 'pi':
                actual = self.engine.eval('pi')
            elif name == 'e':
                actual = self.engine.eval('exp(1)')
            elif name == 'golden_ratio':
                actual = self.engine.eval('(1 + sqrt(5))/2')
            
            is_valid = abs(actual - expected) < 1e-15
            validations.append({
                'name': name,
                'expected': expected,
                'actual': actual,
                'valid': is_valid,
                'error': abs(actual - expected)
            })
        
        return validations
    
    def generate_report(self, output_path: str = "validation_report.json"):
        """Generate comprehensive validation report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'matlab_version': self.engine.version('-release'),
            'test_results': self.results,
            'performance_metrics': self._measure_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _measure_performance(self) -> Dict:
        """Measure performance metrics."""
        metrics = {}
        
        # Startup time
        start = time.time()
        eng = matlab.engine.start_matlab()
        metrics['startup_time'] = time.time() - start
        
        # Simple computation
        start = time.time()
        eng.eval('2 + 2')
        metrics['simple_computation'] = time.time() - start
        
        # Complex computation
        start = time.time()
        eng.eval('inv(rand(100,100))')
        metrics['complex_computation'] = time.time() - start
        
        eng.quit()
        
        return metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.results['summary'].get('success_rate', 0) < 95:
            recommendations.append("Investigate failing tests for accuracy issues")
        
        perf = self.results.get('performance_metrics', {})
        if perf.get('startup_time', 0) > 10:
            recommendations.append("Consider session pooling for better performance")
        
        return recommendations
```

### Unit Tests
- Test MATLAB Engine startup/shutdown
- Verify data type conversions
- Test function call mechanisms
- Validate error handling
- **Mathematical validation against expected results**
- **Performance benchmarking of computations**

### Integration Tests  
- Test with existing physics simulations
- Verify CI/CD pipeline compatibility
- Test session sharing scenarios
- Performance benchmarking vs pure MATLAB
- **Cross-validation with Python numerical libraries (NumPy, SciPy)**
- **Regression testing with golden datasets**

### CI/CD Integration
```yaml
# .github/workflows/python-matlab-tests.yml
name: MATLAB Engine API Validation Pipeline

on: [push, pull_request]

jobs:
  mathematical-validation:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Set up MATLAB
        uses: matlab-actions/setup-matlab@v2
        with:
          release: R2025a
          
      - name: Install dependencies
        run: |
          pip install matlabengine pytest numpy scipy pyyaml
          
      - name: Run Mathematical Validation Suite
        run: |
          python -m pytest src/python/tests/test_mathematical_validation.py -v --tb=short
          
      - name: Run Pipeline Validation
        run: |
          python src/python/tests/run_validation_pipeline.py
          
      - name: Compare with Expected Results
        run: |
          python scripts/validate_results.py --tolerance 1e-9
          
      - name: Generate Validation Report
        run: |
          python scripts/generate_validation_report.py
          
      - name: Upload Validation Results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: |
            validation_report.json
            test_results.xml
            
      - name: Performance Benchmarking
        run: |
          python scripts/benchmark_performance.py
          
      - name: Check Accuracy Thresholds
        run: |
          python scripts/check_accuracy.py --min-accuracy 99.99
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