"""
Mathematical Validation Test Suite for MATLAB Engine API
=======================================================

This module provides comprehensive mathematical validation tests ensuring 99.99% accuracy
requirements for the MATLAB Engine API integration as specified in Issue #1.

Test Categories:
1. Basic arithmetic and algebra
2. Trigonometric functions
3. Matrix operations
4. Equation solving
5. Calculus operations
6. Statistical functions
7. Complex numbers
8. FFT operations

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import math
import cmath
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, field

# Import our modules
from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionManager
from config_manager import get_current_config, Environment, set_environment

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results of a mathematical validation test."""
    test_name: str
    expected: Any
    actual: Any
    error: float
    passed: bool
    tolerance: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryResults:
    """Results for a category of mathematical tests."""
    category: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_error: float
    max_error: float
    total_time: float
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0


class MathematicalValidator:
    """
    Comprehensive mathematical validation framework for MATLAB Engine API.
    
    Ensures 99.99% accuracy requirements are met across all mathematical operations.
    """
    
    def __init__(self, engine: MATLABEngineWrapper, tolerance: float = 1e-12):
        """
        Initialize mathematical validator.
        
        Args:
            engine: Active MATLAB engine wrapper
            tolerance: Default numerical tolerance for comparisons
        """
        self.engine = engine
        self.default_tolerance = tolerance
        self.results: Dict[str, CategoryResults] = {}
        self.golden_values_path = Path(__file__).parent / "golden_values.json"
        self.golden_values = self._load_golden_values()
        
        # Required accuracy for Issue #1
        self.required_accuracy = 0.9999  # 99.99%
    
    def _load_golden_values(self) -> Dict[str, Any]:
        """Load pre-computed golden values for validation."""
        if self.golden_values_path.exists():
            with open(self.golden_values_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_golden_values(self):
        """Save golden values to file."""
        with open(self.golden_values_path, 'w') as f:
            json.dump(self.golden_values, f, indent=2)
    
    def _compare_values(self, expected: Any, actual: Any, tolerance: float) -> Tuple[float, bool]:
        """
        Compare expected vs actual values with specified tolerance.
        
        Returns:
            Tuple of (error, passed)
        """
        try:
            if isinstance(expected, complex) or isinstance(actual, complex):
                error = abs(complex(expected) - complex(actual))
            elif isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
                expected_arr = np.asarray(expected)
                actual_arr = np.asarray(actual)
                if expected_arr.shape != actual_arr.shape:
                    return float('inf'), False
                error = np.max(np.abs(expected_arr - actual_arr))
            else:
                error = abs(float(expected) - float(actual))
            
            passed = error <= tolerance
            return float(error), passed
            
        except Exception as e:
            logger.error(f"Error comparing values: {e}")
            return float('inf'), False
    
    def _run_test(self, test_name: str, matlab_expr: str, expected: Any, 
                  tolerance: Optional[float] = None, metadata: Optional[Dict] = None) -> ValidationResult:
        """
        Run a single mathematical validation test.
        
        Args:
            test_name: Name of the test
            matlab_expr: MATLAB expression to evaluate
            expected: Expected result
            tolerance: Numerical tolerance (uses default if None)
            metadata: Additional test metadata
            
        Returns:
            ValidationResult object
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        if metadata is None:
            metadata = {}
        
        start_time = time.time()
        
        try:
            # Execute MATLAB expression
            actual = self.engine.evaluate(matlab_expr, convert_types=True)
            execution_time = time.time() - start_time
            
            # Compare results
            error, passed = self._compare_values(expected, actual, tolerance)
            
            return ValidationResult(
                test_name=test_name,
                expected=expected,
                actual=actual,
                error=error,
                passed=passed,
                tolerance=tolerance,
                execution_time=execution_time,
                metadata=metadata
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            
            return ValidationResult(
                test_name=test_name,
                expected=expected,
                actual=None,
                error=float('inf'),
                passed=False,
                tolerance=tolerance,
                execution_time=execution_time,
                metadata={**metadata, 'exception': str(e)}
            )
    
    def validate_basic_arithmetic(self) -> CategoryResults:
        """Validate basic arithmetic and algebraic operations."""
        logger.info("Running basic arithmetic validation...")
        
        tests = [
            ("addition", "2 + 3", 5),
            ("subtraction", "10 - 4", 6),
            ("multiplication", "7 * 8", 56),
            ("division", "15 / 3", 5),
            ("power", "2^10", 1024),
            ("square_root", "sqrt(64)", 8),
            ("logarithm_natural", "log(exp(1))", 1),
            ("logarithm_base10", "log10(100)", 2),
            ("exponential", "exp(0)", 1),
            ("absolute_value", "abs(-42)", 42),
            ("floor", "floor(3.7)", 3),
            ("ceil", "ceil(3.2)", 4),
            ("round", "round(3.6)", 4),
            ("modulo", "mod(17, 5)", 2),
            ("factorial", "factorial(5)", 120),
            ("combination", "nchoosek(10, 3)", 120),
            ("polynomial_eval", "polyval([1, 2, 3], 2)", 11),  # x^2 + 2x + 3 at x=2
            ("quadratic_roots", "roots([1, -5, 6])", np.array([3, 2])),
            ("complex_arithmetic", "3 + 4i", 3 + 4j),
            ("matrix_determinant", "det([1, 2; 3, 4])", -2),
        ]
        
        results = []
        for test_name, expr, expected in tests:
            result = self._run_test(f"arithmetic_{test_name}", expr, expected)
            results.append(result)
        
        return self._compile_category_results("Basic Arithmetic", results)
    
    def validate_trigonometric_functions(self) -> CategoryResults:
        """Validate trigonometric functions."""
        logger.info("Running trigonometric function validation...")
        
        # Use high precision values
        pi = math.pi
        tests = [
            ("sin_zero", "sin(0)", 0),
            ("sin_pi_half", "sin(pi/2)", 1),
            ("sin_pi", "sin(pi)", 0, 1e-15),  # Very small tolerance for numerical precision
            ("cos_zero", "cos(0)", 1),
            ("cos_pi_half", "cos(pi/2)", 0, 1e-15),
            ("cos_pi", "cos(pi)", -1),
            ("tan_zero", "tan(0)", 0),
            ("tan_pi_quarter", "tan(pi/4)", 1),
            ("asin_half", "asin(0.5)", pi/6),
            ("acos_half", "acos(0.5)", pi/3),
            ("atan_one", "atan(1)", pi/4),
            ("sinh_zero", "sinh(0)", 0),
            ("cosh_zero", "cosh(0)", 1),
            ("tanh_zero", "tanh(0)", 0),
            ("sec_zero", "sec(0)", 1),
            ("csc_pi_half", "csc(pi/2)", 1),
            ("cot_pi_quarter", "cot(pi/4)", 1),
            ("deg2rad", "deg2rad(180)", pi),
            ("rad2deg", "rad2deg(pi)", 180),
            ("hypot", "hypot(3, 4)", 5),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"trig_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Trigonometric Functions", results)
    
    def validate_matrix_operations(self) -> CategoryResults:
        """Validate matrix operations."""
        logger.info("Running matrix operations validation...")
        
        # Set up test matrices in MATLAB workspace
        self.engine.set_workspace_variable("A", np.array([[1, 2], [3, 4]]))
        self.engine.set_workspace_variable("B", np.array([[5, 6], [7, 8]]))
        self.engine.set_workspace_variable("v", np.array([1, 2, 3]))
        
        tests = [
            ("matrix_addition", "A + B", np.array([[6, 8], [10, 12]])),
            ("matrix_subtraction", "A - B", np.array([[-4, -4], [-4, -4]])),
            ("matrix_multiplication", "A * B", np.array([[19, 22], [43, 50]])),
            ("matrix_transpose", "A'", np.array([[1, 3], [2, 4]])),
            ("matrix_inverse", "inv(A)", np.array([[-2, 1], [1.5, -0.5]])),
            ("matrix_determinant", "det(A)", -2),
            ("matrix_trace", "trace(A)", 5),
            ("matrix_rank", "rank(A)", 2),
            ("matrix_norm_frobenius", "norm(A, 'fro')", math.sqrt(30)),
            ("matrix_norm_2", "norm(A, 2)", 5.4649857042, 1e-8),
            ("matrix_condition", "cond(A)", 14.9330814481, 1e-8),
            ("eigenvalues", "sort(eig(A))", np.sort([-0.3722813232690143, 5.372281323269014])),
            ("vector_dot_product", "dot(v, v)", 14),
            ("vector_cross_product", "cross([1;0;0], [0;1;0])", np.array([0, 0, 1])),
            ("vector_norm", "norm(v)", math.sqrt(14)),
            ("matrix_power", "A^2", np.array([[7, 10], [15, 22]])),
            ("kronecker_product", "kron([1, 2], [3; 4])", np.array([[3, 6], [4, 8]])),
            ("matrix_exponential", "expm(zeros(2))", np.eye(2)),
            ("singular_values", "svd(A)", np.array([5.4649857042, 0.36596619416]), 1e-8),
            ("qr_decomposition_q", "[Q,R] = qr(A); norm(Q'*Q - eye(2))", 0, 1e-14),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"matrix_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Matrix Operations", results)
    
    def validate_equation_solving(self) -> CategoryResults:
        """Validate equation solving capabilities."""
        logger.info("Running equation solving validation...")
        
        tests = [
            # Linear system: x + y = 3, 2x - y = 0 => x=1, y=2
            ("linear_system", "[1, 1; 2, -1] \\ [3; 0]", np.array([1, 2])),
            
            # Polynomial roots: x^2 - 5x + 6 = 0 => x = 2, 3
            ("polynomial_roots", "sort(roots([1, -5, 6]))", np.array([2, 3])),
            
            # Nonlinear equation: x^2 = 2 => x ≈ 1.414
            ("nonlinear_sqrt2", "fzero(@(x) x^2 - 2, 1)", math.sqrt(2), 1e-10),
            
            # Optimization: minimize x^2 + 4x + 4 => x = -2
            ("optimization_min", "fminsearch(@(x) x^2 + 4*x + 4, 0)", -2, 1e-6),
            
            # System of nonlinear equations using symbolic (if available)
            ("least_squares", "(A'*A) \\ (A'*b) where A=[1,1;1,2;1,3], b=[2;3;4]", 
             np.array([1, 1]), 1e-12),
        ]
        
        results = []
        
        # Set up matrices for least squares
        self.engine.evaluate("A = [1,1;1,2;1,3]; b = [2;3;4];")
        
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            # Handle special expressions
            if "where" in expr:
                expr = expr.split(" where ")[0]
            
            result = self._run_test(f"solve_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Equation Solving", results)
    
    def validate_calculus_operations(self) -> CategoryResults:
        """Validate calculus operations."""
        logger.info("Running calculus operations validation...")
        
        tests = [
            # Numerical derivatives
            ("derivative_x2", "gradient([1, 4, 9, 16, 25], 1)", np.array([3, 3, 5, 7, 9])),
            
            # Numerical integration using trapz
            ("integration_x2", "trapz([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])", 64/3, 0.1),
            
            # Integration using quad (if available)
            ("integration_sin", "integral(@sin, 0, pi)", 2, 1e-10),
            
            # Differential equations - simple exponential
            ("ode_exponential", "[t,y] = ode45(@(t,y) y, [0 1], 1); y(end)", math.e, 1e-6),
            
            # Gradient of a function
            ("gradient_2d", "[dx,dy] = gradient([1,4,9;2,5,10;3,6,11]); dx(2,2)", 3),
            
            # Laplacian (second derivatives)
            ("laplacian_1d", "del2([1, 4, 9, 16, 25])", np.array([0, 0, 0, 0, 0]), 1e-10),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"calculus_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Calculus Operations", results)
    
    def validate_statistical_functions(self) -> CategoryResults:
        """Validate statistical functions."""
        logger.info("Running statistical functions validation...")
        
        # Create test data
        self.engine.set_workspace_variable("data", np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.engine.set_workspace_variable("normal_data", np.random.normal(0, 1, 1000))
        
        tests = [
            ("mean", "mean(data)", 5.5),
            ("median", "median(data)", 5.5),
            ("std", "std(data)", math.sqrt(33/4), 1e-10),
            ("var", "var(data)", 33/4),
            ("min", "min(data)", 1),
            ("max", "max(data)", 10),
            ("range", "range(data)", 9),
            ("sum", "sum(data)", 55),
            ("prod", "prod(1:5)", 120),
            ("cumsum", "cumsum([1,2,3,4])", np.array([1, 3, 6, 10])),
            ("cumprod", "cumprod([1,2,3,4])", np.array([1, 2, 6, 24])),
            ("sort", "sort([3,1,4,1,5])", np.array([1, 1, 3, 4, 5])),
            ("correlation", "corrcoef([1,2,3], [2,4,6])", np.array([[1, 1], [1, 1]])),
            ("covariance", "cov([1,2,3], [2,4,6])", np.array([[1, 2], [2, 4]])),
            ("histogram_counts", "histcounts(1:10, 5)", np.array([2, 2, 2, 2, 2])),
            ("percentile", "prctile(data, 50)", 5.5),
            ("quantile", "quantile(data, 0.5)", 5.5),
            ("mode", "mode([1,1,2,2,2,3])", 2),
            ("skewness", "skewness([1,2,3,4,5])", 0, 1e-12),
            ("kurtosis", "kurtosis([1,2,3,4,5])", 1.7, 0.1),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"stats_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Statistical Functions", results)
    
    def validate_complex_numbers(self) -> CategoryResults:
        """Validate complex number operations."""
        logger.info("Running complex number validation...")
        
        tests = [
            ("complex_creation", "1 + 2i", 1 + 2j),
            ("complex_addition", "(1+2i) + (3+4i)", 4 + 6j),
            ("complex_multiplication", "(1+2i) * (3+4i)", -5 + 10j),
            ("complex_conjugate", "conj(3+4i)", 3 - 4j),
            ("complex_absolute", "abs(3+4i)", 5),
            ("complex_angle", "angle(1+i)", math.pi/4),
            ("complex_real", "real(3+4i)", 3),
            ("complex_imag", "imag(3+4i)", 4),
            ("complex_exp", "exp(1i*pi)", -1 + 0j, 1e-15),
            ("complex_log", "log(-1)", 0 + math.pi*1j, 1e-15),
            ("complex_sqrt", "sqrt(-1)", 0 + 1j, 1e-15),
            ("complex_power", "(1+i)^2", 0 + 2j, 1e-15),
            ("complex_roots", "roots([1, 0, 1])", np.array([1j, -1j]), 1e-15),
            ("complex_matrix", "eig([0, -1; 1, 0])", np.array([1j, -1j]), 1e-15),
            ("complex_fft_symmetry", "fft([1, 0, 0, 0])", np.array([1, 1, 1, 1])),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"complex_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("Complex Numbers", results)
    
    def validate_fft_operations(self) -> CategoryResults:
        """Validate FFT and signal processing operations."""
        logger.info("Running FFT operations validation...")
        
        # Create test signals
        N = 8
        n = np.arange(N)
        
        # Unit impulse
        delta = np.zeros(N)
        delta[0] = 1
        self.engine.set_workspace_variable("delta", delta)
        
        # Sinusoid
        sinusoid = np.sin(2 * np.pi * n / N)
        self.engine.set_workspace_variable("sinusoid", sinusoid)
        
        tests = [
            # FFT of unit impulse should be all ones
            ("fft_impulse", "fft(delta)", np.ones(N)),
            
            # FFT properties: linearity
            ("fft_linearity", "fft([1, 0, 0, 0]) + fft([0, 1, 0, 0])", 
             np.array([1, 1, 1, 1]) + np.array([1, 1j, -1, -1j]), 1e-15),
            
            # IFFT(FFT(x)) = x
            ("ifft_fft_identity", "ifft(fft([1, 2, 3, 4]))", np.array([1, 2, 3, 4]), 1e-15),
            
            # Parseval's theorem: sum(|x|^2) = sum(|X|^2)/N
            ("parseval_theorem", "sum(abs([1,2,3,4]).^2) - sum(abs(fft([1,2,3,4])).^2)/4", 
             0, 1e-15),
            
            # DC component
            ("fft_dc_component", "fft([1, 1, 1, 1])", np.array([4, 0, 0, 0]), 1e-15),
            
            # FFT of real signal has conjugate symmetry
            ("fft_conjugate_symmetry", "X = fft([1,2,1,0]); X(1) - conj(X(4))", 0, 1e-15),
            
            # Power spectral density
            ("psd_white_noise", "mean(abs(fft(randn(1, 1000))).^2)", 1000, 100),  # Approximate
            
            # Convolution theorem: FFT(conv(a,b)) = FFT(a).*FFT(b)
            ("convolution_theorem", 
             "norm(fft(conv([1,1], [1,2]), 4) - fft([1,1,0,0]).*fft([1,2,0,0]))", 
             0, 1e-14),
            
            # Window functions
            ("hamming_window", "hamming(4)", 
             np.array([0.08, 0.54, 0.54, 0.08]), 0.01),
        ]
        
        results = []
        for test_data in tests:
            if len(test_data) == 3:
                test_name, expr, expected = test_data
                tolerance = None
            else:
                test_name, expr, expected, tolerance = test_data
            
            result = self._run_test(f"fft_{test_name}", expr, expected, tolerance)
            results.append(result)
        
        return self._compile_category_results("FFT Operations", results)
    
    def _compile_category_results(self, category: str, results: List[ValidationResult]) -> CategoryResults:
        """Compile individual test results into category results."""
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        
        errors = [r.error for r in results if r.error != float('inf')]
        average_error = sum(errors) / len(errors) if errors else float('inf')
        max_error = max(errors) if errors else float('inf')
        total_time = sum(r.execution_time for r in results)
        
        category_result = CategoryResults(
            category=category,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_error=average_error,
            max_error=max_error,
            total_time=total_time,
            results=results
        )
        
        self.results[category] = category_result
        return category_result
    
    def run_full_validation(self) -> Dict[str, CategoryResults]:
        """Run complete mathematical validation suite."""
        logger.info("Starting full mathematical validation suite...")
        
        validation_methods = [
            self.validate_basic_arithmetic,
            self.validate_trigonometric_functions,
            self.validate_matrix_operations,
            self.validate_equation_solving,
            self.validate_calculus_operations,
            self.validate_statistical_functions,
            self.validate_complex_numbers,
            self.validate_fft_operations,
        ]
        
        for method in validation_methods:
            try:
                method()
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {e}")
        
        return self.results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = sum(cat.total_tests for cat in self.results.values())
        total_passed = sum(cat.passed_tests for cat in self.results.values())
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_tests - total_passed,
                "overall_success_rate": overall_success_rate,
                "meets_requirements": overall_success_rate >= self.required_accuracy * 100,
                "required_accuracy": self.required_accuracy * 100,
            },
            "categories": {
                cat_name: {
                    "total_tests": cat.total_tests,
                    "passed_tests": cat.passed_tests,
                    "failed_tests": cat.failed_tests,
                    "success_rate": cat.success_rate,
                    "average_error": cat.average_error,
                    "max_error": cat.max_error,
                    "total_time": cat.total_time,
                }
                for cat_name, cat in self.results.items()
            },
            "failed_tests": [
                {
                    "category": cat_name,
                    "test": result.test_name,
                    "expected": str(result.expected),
                    "actual": str(result.actual),
                    "error": result.error,
                    "tolerance": result.tolerance,
                }
                for cat_name, cat in self.results.items()
                for result in cat.results
                if not result.passed
            ]
        }
        
        return report


# Pytest Test Classes
class TestMathematicalValidation:
    """Pytest test class for mathematical validation."""
    
    @pytest.fixture(scope="class")
    def matlab_engine(self):
        """Create MATLAB engine for testing."""
        config = MATLABConfig(
            startup_options=['-nojvm', '-nodisplay'],
            headless_mode=True,
            performance_monitoring=True
        )
        
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        yield engine
        engine.close()
    
    @pytest.fixture(scope="class")
    def validator(self, matlab_engine):
        """Create mathematical validator."""
        return MathematicalValidator(matlab_engine)
    
    def test_basic_arithmetic_validation(self, validator):
        """Test basic arithmetic validation."""
        results = validator.validate_basic_arithmetic()
        
        assert results.success_rate >= 99.99, f"Basic arithmetic success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_trigonometric_validation(self, validator):
        """Test trigonometric function validation."""
        results = validator.validate_trigonometric_functions()
        
        assert results.success_rate >= 99.99, f"Trigonometric success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_matrix_operations_validation(self, validator):
        """Test matrix operations validation."""
        results = validator.validate_matrix_operations()
        
        assert results.success_rate >= 99.99, f"Matrix operations success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_equation_solving_validation(self, validator):
        """Test equation solving validation."""
        results = validator.validate_equation_solving()
        
        assert results.success_rate >= 99.99, f"Equation solving success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_calculus_operations_validation(self, validator):
        """Test calculus operations validation."""
        results = validator.validate_calculus_operations()
        
        assert results.success_rate >= 99.99, f"Calculus operations success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_statistical_functions_validation(self, validator):
        """Test statistical functions validation."""
        results = validator.validate_statistical_functions()
        
        assert results.success_rate >= 99.99, f"Statistical functions success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_complex_numbers_validation(self, validator):
        """Test complex numbers validation."""
        results = validator.validate_complex_numbers()
        
        assert results.success_rate >= 99.99, f"Complex numbers success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_fft_operations_validation(self, validator):
        """Test FFT operations validation."""
        results = validator.validate_fft_operations()
        
        assert results.success_rate >= 99.99, f"FFT operations success rate: {results.success_rate}%"
        assert results.failed_tests == 0, f"Failed tests: {[r.test_name for r in results.results if not r.passed]}"
    
    def test_full_validation_suite(self, validator):
        """Test complete validation suite."""
        results = validator.run_full_validation()
        report = validator.generate_validation_report()
        
        # Check overall requirements
        assert report["summary"]["meets_requirements"], \
            f"Overall success rate {report['summary']['overall_success_rate']}% < {validator.required_accuracy * 100}%"
        
        # Log detailed results
        logger.info(f"Validation Report Summary:")
        logger.info(f"Total Tests: {report['summary']['total_tests']}")
        logger.info(f"Overall Success Rate: {report['summary']['overall_success_rate']:.2f}%")
        logger.info(f"Requirements Met: {report['summary']['meets_requirements']}")
        
        if report["failed_tests"]:
            logger.warning(f"Failed Tests: {len(report['failed_tests'])}")
            for failed in report["failed_tests"]:
                logger.warning(f"  {failed['category']}.{failed['test']}: error={failed['error']}")


if __name__ == "__main__":
    # Run standalone validation
    print("MATLAB Engine Mathematical Validation Suite")
    print("=" * 60)
    
    # Create configuration
    config = MATLABConfig(
        startup_options=['-nojvm', '-nodisplay'],
        headless_mode=True,
        performance_monitoring=True
    )
    
    # Run validation
    with MATLABEngineWrapper(config=config) as engine:
        validator = MathematicalValidator(engine)
        results = validator.run_full_validation()
        report = validator.generate_validation_report()
        
        # Print summary
        print(f"\nValidation Results Summary:")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['total_passed']}")
        print(f"Failed: {report['summary']['total_failed']}")
        print(f"Overall Success Rate: {report['summary']['overall_success_rate']:.2f}%")
        print(f"Requirements Met (99.99%): {report['summary']['meets_requirements']}")
        
        # Print category breakdown
        print(f"\nCategory Breakdown:")
        for cat_name, cat_data in report['categories'].items():
            print(f"  {cat_name}: {cat_data['success_rate']:.2f}% "
                  f"({cat_data['passed_tests']}/{cat_data['total_tests']})")
        
        # Print failed tests if any
        if report['failed_tests']:
            print(f"\nFailed Tests:")
            for failed in report['failed_tests']:
                print(f"  {failed['category']}.{failed['test']}: "
                      f"error={failed['error']:.2e}, tolerance={failed['tolerance']:.2e}")
        
        print(f"\nValidation completed successfully!")