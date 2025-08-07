"""
Comprehensive Regression Test Suite
==================================

This module provides regression testing capabilities with golden dataset
validation, ensuring that changes don't break existing functionality.

Part of Issue #1: MATLAB Engine API Integration testing framework.

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import json
import time
import hashlib
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


@dataclass
class RegressionTestResult:
    """Result of a regression test."""
    test_name: str
    test_version: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenDataset:
    """Golden dataset for regression validation."""
    name: str
    version: str
    creation_date: datetime
    test_results: List[RegressionTestResult] = field(default_factory=list)
    validation_checksums: Dict[str, str] = field(default_factory=dict)


class MockMATLABEngineWithHistory:
    """Mock MATLAB engine that maintains operation history for regression testing."""
    
    def __init__(self):
        self.workspace = {}
        self._closed = False
        self._call_history = []
        self._operation_results = {}
        
        # Initialize with consistent results for regression testing
        self._setup_golden_results()
    
    def _setup_golden_results(self):
        """Setup golden results that should remain consistent."""
        self._operation_results = {
            # Basic arithmetic
            "2 + 3": 5,
            "10 - 4": 6,
            "7 * 8": 56,
            "15 / 3": 5.0,
            "2^10": 1024,
            "sqrt(64)": 8.0,
            "sin(0)": 0.0,
            "cos(0)": 1.0,
            "tan(pi/4)": 1.0,
            
            # Matrix operations
            "det([1, 2; 3, 4])": -2.0,
            "trace([1, 2; 3, 4])": 5.0,
            "norm([3, 4])": 5.0,
            
            # Statistical functions
            "mean([1, 2, 3, 4, 5])": 3.0,
            "std([1, 2, 3, 4, 5])": 1.5811388300841898,
            "var([1, 2, 3, 4, 5])": 2.5,
            
            # Complex operations
            "abs(3 + 4i)": 5.0,
            "angle(1 + i)": 0.7853981633974483,  # pi/4
            
            # FFT operations
            "sum(abs(fft([1, 0, 0, 0])))": 4.0,
            
            # Special functions
            "gamma(5)": 24.0,  # 4!
            "factorial(4)": 24,
            "nchoosek(10, 3)": 120,
        }
    
    def eval(self, expression, nargout=0, **kwargs):
        """Mock MATLAB eval with consistent results for regression testing."""
        self._call_history.append(('eval', expression, kwargs, time.time()))
        
        if self._closed:
            raise RuntimeError("MATLAB engine is closed")
        
        # Return consistent golden values
        if expression in self._operation_results:
            return self._operation_results[expression]
        
        # Handle workspace variables
        if expression in self.workspace:
            return self.workspace[expression]
        
        # Default return for unknown expressions
        return 1.0
    
    def get_operation_history(self):
        """Get the complete operation history."""
        return self._call_history.copy()
    
    def quit(self):
        """Mock quit function."""
        self._closed = True


class RegressionTestFramework:
    """Framework for running regression tests with golden dataset validation."""
    
    def __init__(self, golden_data_dir: Optional[Path] = None):
        """Initialize regression test framework."""
        self.golden_data_dir = golden_data_dir or Path(__file__).parent / "golden_data"
        self.golden_data_dir.mkdir(exist_ok=True)
        self.current_dataset = None
        self.test_results = []
        
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for change detection."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, np.ndarray):
            data_str = str(data.tobytes())
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def load_golden_dataset(self, dataset_name: str) -> Optional[GoldenDataset]:
        """Load golden dataset from file."""
        dataset_file = self.golden_data_dir / f"{dataset_name}.json"
        
        if not dataset_file.exists():
            return None
        
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            for result_data in data.get('test_results', []):
                result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
            
            data['creation_date'] = datetime.fromisoformat(data['creation_date'])
            
            # Recreate dataclass instances
            test_results = [RegressionTestResult(**result_data) for result_data in data.get('test_results', [])]
            
            return GoldenDataset(
                name=data['name'],
                version=data['version'],
                creation_date=data['creation_date'],
                test_results=test_results,
                validation_checksums=data.get('validation_checksums', {})
            )
        
        except Exception as e:
            logger.error(f"Failed to load golden dataset {dataset_name}: {e}")
            return None
    
    def save_golden_dataset(self, dataset: GoldenDataset):
        """Save golden dataset to file."""
        dataset_file = self.golden_data_dir / f"{dataset.name}.json"
        
        # Convert to serializable format
        data = asdict(dataset)
        
        # Convert datetime objects to ISO format strings
        data['creation_date'] = dataset.creation_date.isoformat()
        
        for result_data in data['test_results']:
            result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp']).isoformat() if isinstance(result_data['timestamp'], str) else result_data['timestamp'].isoformat()
        
        try:
            with open(dataset_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved golden dataset to {dataset_file}")
        
        except Exception as e:
            logger.error(f"Failed to save golden dataset {dataset.name}: {e}")
    
    def create_golden_dataset(self, name: str, version: str = "1.0") -> GoldenDataset:
        """Create a new golden dataset."""
        dataset = GoldenDataset(
            name=name,
            version=version,
            creation_date=datetime.now()
        )
        
        self.current_dataset = dataset
        return dataset
    
    def run_regression_test(self, test_name: str, test_function: callable, 
                          input_data: Any, expected_output: Any = None) -> RegressionTestResult:
        """Run a single regression test."""
        start_time = time.time()
        input_hash = self._compute_hash(input_data)
        
        try:
            # Run the test function
            actual_output = test_function(input_data)
            execution_time = time.time() - start_time
            
            output_hash = self._compute_hash(actual_output)
            
            # Compare with expected output if provided
            success = True
            error_message = None
            
            if expected_output is not None:
                expected_hash = self._compute_hash(expected_output)
                success = (output_hash == expected_hash)
                
                if not success:
                    error_message = f"Output mismatch: expected {expected_hash}, got {output_hash}"
            
            result = RegressionTestResult(
                test_name=test_name,
                test_version="current",
                timestamp=datetime.now(),
                input_hash=input_hash,
                output_hash=output_hash,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                metadata={'actual_output': str(actual_output)[:1000]}  # Truncate for storage
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = RegressionTestResult(
                test_name=test_name,
                test_version="current",
                timestamp=datetime.now(),
                input_hash=input_hash,
                output_hash="error",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
        
        self.test_results.append(result)
        
        if self.current_dataset is not None:
            self.current_dataset.test_results.append(result)
        
        return result
    
    def validate_against_golden(self, current_result: RegressionTestResult, 
                              golden_result: RegressionTestResult) -> Tuple[bool, List[str]]:
        """Validate current result against golden result."""
        issues = []
        
        # Check output hash
        if current_result.output_hash != golden_result.output_hash:
            issues.append(f"Output hash changed: {golden_result.output_hash} -> {current_result.output_hash}")
        
        # Check success status
        if current_result.success != golden_result.success:
            issues.append(f"Success status changed: {golden_result.success} -> {current_result.success}")
        
        # Check for significant performance regression (>50% slower)
        if current_result.execution_time > golden_result.execution_time * 1.5:
            issues.append(f"Performance regression: {golden_result.execution_time:.3f}s -> {current_result.execution_time:.3f}s")
        
        # Check for error message changes
        if current_result.error_message != golden_result.error_message:
            if golden_result.error_message is None and current_result.error_message is not None:
                issues.append(f"New error introduced: {current_result.error_message}")
            elif golden_result.error_message is not None and current_result.error_message is None:
                issues.append(f"Error resolved: {golden_result.error_message}")
            elif golden_result.error_message != current_result.error_message:
                issues.append(f"Error message changed")
        
        return len(issues) == 0, issues
    
    def run_full_regression_suite(self, engine_wrapper, test_suite_name: str = "full_suite") -> Dict[str, Any]:
        """Run comprehensive regression test suite."""
        logger.info(f"Starting regression test suite: {test_suite_name}")
        
        # Load existing golden dataset or create new one
        golden_dataset = self.load_golden_dataset(test_suite_name)
        if golden_dataset is None:
            golden_dataset = self.create_golden_dataset(test_suite_name)
            logger.info("Created new golden dataset")
        else:
            logger.info(f"Loaded golden dataset version {golden_dataset.version}")
        
        # Define regression test cases
        test_cases = self._define_regression_test_cases()
        
        # Run all test cases
        current_results = []
        for test_case in test_cases:
            result = self.run_regression_test(
                test_case['name'],
                lambda input_data, tc=test_case: self._execute_test_case(engine_wrapper, tc),
                test_case['input'],
                test_case.get('expected_output')
            )
            current_results.append(result)
        
        # Validate against golden dataset
        validation_results = []
        for current_result in current_results:
            # Find corresponding golden result
            golden_result = next(
                (gr for gr in golden_dataset.test_results if gr.test_name == current_result.test_name),
                None
            )
            
            if golden_result is not None:
                is_valid, issues = self.validate_against_golden(current_result, golden_result)
                validation_results.append({
                    'test_name': current_result.test_name,
                    'valid': is_valid,
                    'issues': issues
                })
            else:
                validation_results.append({
                    'test_name': current_result.test_name,
                    'valid': True,
                    'issues': ['New test case - no golden reference']
                })
        
        # Update golden dataset with current results if this is a new dataset
        if len(golden_dataset.test_results) == 0:
            golden_dataset.test_results = current_results
            self.save_golden_dataset(golden_dataset)
        
        # Compile results
        total_tests = len(validation_results)
        valid_tests = sum(1 for vr in validation_results if vr['valid'])
        
        report = {
            'suite_name': test_suite_name,
            'total_tests': total_tests,
            'valid_tests': valid_tests,
            'regression_rate': (valid_tests / total_tests) * 100 if total_tests > 0 else 0,
            'validation_results': validation_results,
            'execution_summary': {
                'total_execution_time': sum(r.execution_time for r in current_results),
                'average_execution_time': sum(r.execution_time for r in current_results) / len(current_results) if current_results else 0,
                'successful_tests': sum(1 for r in current_results if r.success),
                'failed_tests': sum(1 for r in current_results if not r.success),
            }
        }
        
        return report
    
    def _define_regression_test_cases(self) -> List[Dict[str, Any]]:
        """Define the standard regression test cases."""
        return [
            {
                'name': 'basic_arithmetic',
                'input': ['2 + 3', '10 - 4', '7 * 8', '15 / 3'],
                'expected_output': [5, 6, 56, 5.0]
            },
            {
                'name': 'trigonometric_functions',
                'input': ['sin(0)', 'cos(0)', 'tan(pi/4)'],
                'expected_output': [0.0, 1.0, 1.0]
            },
            {
                'name': 'matrix_operations',
                'input': ['det([1, 2; 3, 4])', 'trace([1, 2; 3, 4])', 'norm([3, 4])'],
                'expected_output': [-2.0, 5.0, 5.0]
            },
            {
                'name': 'statistical_functions',
                'input': ['mean([1, 2, 3, 4, 5])', 'std([1, 2, 3, 4, 5])', 'var([1, 2, 3, 4, 5])'],
                'expected_output': [3.0, 1.5811388300841898, 2.5]
            },
            {
                'name': 'complex_operations',
                'input': ['abs(3 + 4i)', 'angle(1 + i)'],
                'expected_output': [5.0, 0.7853981633974483]
            },
            {
                'name': 'special_functions',
                'input': ['factorial(4)', 'nchoosek(10, 3)'],
                'expected_output': [24, 120]
            }
        ]
    
    def _execute_test_case(self, engine_wrapper, test_case: Dict[str, Any]) -> List[Any]:
        """Execute a single test case and return results."""
        results = []
        
        for expression in test_case['input']:
            try:
                result = engine_wrapper.evaluate(expression)
                results.append(result)
            except Exception as e:
                results.append(f"ERROR: {str(e)}")
        
        return results


@pytest.fixture
def mock_matlab_module():
    """Fixture that patches the matlab module."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngineWithHistory())
        yield mock_matlab


class TestRegressionSuite:
    """Test suite for regression testing."""
    
    def test_golden_dataset_creation_and_loading(self, mock_matlab_module):
        """Test creating and loading golden datasets."""
        framework = RegressionTestFramework()
        
        # Create a new dataset
        dataset = framework.create_golden_dataset("test_dataset", "1.0")
        
        assert dataset.name == "test_dataset"
        assert dataset.version == "1.0"
        assert len(dataset.test_results) == 0
        
        # Add a test result
        test_result = RegressionTestResult(
            test_name="test1",
            test_version="1.0",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="def456",
            execution_time=0.1,
            success=True
        )
        
        dataset.test_results.append(test_result)
        
        # Save and reload
        framework.save_golden_dataset(dataset)
        loaded_dataset = framework.load_golden_dataset("test_dataset")
        
        assert loaded_dataset is not None
        assert loaded_dataset.name == "test_dataset"
        assert len(loaded_dataset.test_results) == 1
        assert loaded_dataset.test_results[0].test_name == "test1"
    
    def test_regression_test_execution(self, mock_matlab_module):
        """Test running individual regression tests."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = RegressionTestFramework()
        
        # Define a simple test function
        def simple_test(input_data):
            return engine.evaluate(input_data)
        
        # Run regression test
        result = framework.run_regression_test(
            "arithmetic_test",
            simple_test,
            "2 + 3",
            5
        )
        
        assert result.test_name == "arithmetic_test"
        assert result.success is True
        assert result.execution_time > 0
        assert result.input_hash is not None
        assert result.output_hash is not None
    
    def test_full_regression_suite(self, mock_matlab_module):
        """Test running the full regression suite."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = RegressionTestFramework(golden_data_dir=Path(temp_dir))
            
            # Run regression suite
            report = framework.run_full_regression_suite(engine, "test_suite")
            
            assert report['suite_name'] == "test_suite"
            assert report['total_tests'] > 0
            assert report['regression_rate'] >= 0
            assert 'execution_summary' in report
            assert len(report['validation_results']) > 0
    
    def test_golden_validation(self, mock_matlab_module):
        """Test validation against golden results."""
        framework = RegressionTestFramework()
        
        # Create golden result
        golden_result = RegressionTestResult(
            test_name="test1",
            test_version="1.0",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="def456",
            execution_time=0.1,
            success=True
        )
        
        # Create matching current result
        current_result = RegressionTestResult(
            test_name="test1",
            test_version="current",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="def456",
            execution_time=0.1,
            success=True
        )
        
        # Should validate successfully
        is_valid, issues = framework.validate_against_golden(current_result, golden_result)
        assert is_valid is True
        assert len(issues) == 0
        
        # Create non-matching result
        different_result = RegressionTestResult(
            test_name="test1",
            test_version="current",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="xyz789",  # Different output
            execution_time=0.1,
            success=True
        )
        
        # Should fail validation
        is_valid, issues = framework.validate_against_golden(different_result, golden_result)
        assert is_valid is False
        assert len(issues) > 0
        assert "Output hash changed" in issues[0]
    
    def test_performance_regression_detection(self, mock_matlab_module):
        """Test detection of performance regressions."""
        framework = RegressionTestFramework()
        
        # Golden result with good performance
        golden_result = RegressionTestResult(
            test_name="perf_test",
            test_version="1.0",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="def456",
            execution_time=0.1,
            success=True
        )
        
        # Current result with poor performance
        slow_result = RegressionTestResult(
            test_name="perf_test",
            test_version="current",
            timestamp=datetime.now(),
            input_hash="abc123",
            output_hash="def456",
            execution_time=0.2,  # 2x slower - should trigger regression detection
            success=True
        )
        
        is_valid, issues = framework.validate_against_golden(slow_result, golden_result)
        assert is_valid is False
        assert any("Performance regression" in issue for issue in issues)
    
    def test_hash_computation_consistency(self, mock_matlab_module):
        """Test that hash computation is consistent."""
        framework = RegressionTestFramework()
        
        # Same data should produce same hash
        data1 = [1, 2, 3, 4, 5]
        data2 = [1, 2, 3, 4, 5]
        
        hash1 = framework._compute_hash(data1)
        hash2 = framework._compute_hash(data2)
        
        assert hash1 == hash2
        
        # Different data should produce different hash
        data3 = [1, 2, 3, 4, 6]
        hash3 = framework._compute_hash(data3)
        
        assert hash1 != hash3
        
        # Different data types with same values might produce different hashes
        data_array = np.array([1, 2, 3, 4, 5])
        hash_array = framework._compute_hash(data_array)
        
        # This could be same or different - depends on implementation
        # The key is that it's consistent
        hash_array2 = framework._compute_hash(data_array)
        assert hash_array == hash_array2


if __name__ == "__main__":
    # Run standalone regression tests
    print("Running Comprehensive Regression Test Suite...")
    print("=" * 60)
    
    # Mock the MATLAB module
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEngineWithHistory())
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        try:
            from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
            
            print("✓ Successfully imported modules for regression testing")
            
            # Create temporary directory for golden data
            with tempfile.TemporaryDirectory() as temp_dir:
                framework = RegressionTestFramework(golden_data_dir=Path(temp_dir))
                
                # Create MATLAB engine
                config = MATLABConfig(headless_mode=True)
                engine = MATLABEngineWrapper(config=config)
                engine.start()
                
                print("✓ Started MATLAB engine wrapper")
                
                # Run full regression suite
                print("\n--- Running Full Regression Suite ---")
                report = framework.run_full_regression_suite(engine, "demonstration_suite")
                
                print(f"✓ Regression suite completed:")
                print(f"  Total tests: {report['total_tests']}")
                print(f"  Valid tests: {report['valid_tests']}")
                print(f"  Regression rate: {report['regression_rate']:.1f}%")
                print(f"  Total execution time: {report['execution_summary']['total_execution_time']:.3f}s")
                print(f"  Average execution time: {report['execution_summary']['average_execution_time']:.3f}s")
                
                # Show individual test results
                print(f"\n--- Individual Test Results ---")
                for vr in report['validation_results']:
                    status = "PASS" if vr['valid'] else "FAIL"
                    print(f"  {vr['test_name']}: {status}")
                    if vr['issues'] and vr['issues'] != ['New test case - no golden reference']:
                        for issue in vr['issues']:
                            print(f"    - {issue}")
                
                # Test golden dataset operations
                print(f"\n--- Testing Golden Dataset Operations ---")
                dataset = framework.create_golden_dataset("test_golden", "1.0")
                
                # Add a test result
                test_result = RegressionTestResult(
                    test_name="golden_test",
                    test_version="1.0",
                    timestamp=datetime.now(),
                    input_hash="test_input_hash",
                    output_hash="test_output_hash",
                    execution_time=0.05,
                    success=True
                )
                
                dataset.test_results.append(test_result)
                framework.save_golden_dataset(dataset)
                
                # Load it back
                loaded_dataset = framework.load_golden_dataset("test_golden")
                assert loaded_dataset is not None
                assert len(loaded_dataset.test_results) == 1
                
                print("✓ Golden dataset save/load operations work correctly")
                
                # Test hash consistency
                print(f"\n--- Testing Hash Consistency ---")
                test_data = [1, 2, 3, 4, 5]
                hash1 = framework._compute_hash(test_data)
                hash2 = framework._compute_hash(test_data)
                
                assert hash1 == hash2
                print(f"✓ Hash computation is consistent: {hash1}")
                
                # Clean up
                engine.close()
                
                print("\n" + "=" * 60)
                print("All regression tests completed successfully!")
                
        except Exception as e:
            print(f"✗ Regression test failed: {e}")
            raise