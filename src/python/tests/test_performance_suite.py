"""
Comprehensive Performance and Load Testing Suite
===============================================

This module provides performance testing, load testing, and benchmarking
capabilities for the MATLAB Engine API wrapper.

Part of Issue #1: MATLAB Engine API Integration testing framework.

Author: Murray Kopit
License: MIT
"""

import pytest
import numpy as np
import time
import threading
import psutil
import gc
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import mean, median, stdev
import json
from datetime import datetime
import resource

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test."""
    test_name: str
    operation_count: int
    total_time: float
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_time: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestResult:
    """Results from a load test."""
    test_name: str
    concurrent_users: int
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_ops_per_second: float
    peak_memory_mb: float
    average_cpu_percent: float
    error_rate: float


class MockMATLABEnginePerformance:
    """Mock MATLAB engine optimized for performance testing."""
    
    def __init__(self, operation_delay=0.001, failure_rate=0.0):
        self.workspace = {}
        self._closed = False
        self._call_history = []
        self._operation_delay = operation_delay
        self._failure_rate = failure_rate
        self._operation_count = 0
        
        # Pre-computed results for common operations
        self._results_cache = {
            "1+1": 2,
            "2+2": 4,
            "sqrt(4)": 2.0,
            "sin(0)": 0.0,
            "cos(0)": 1.0,
            "mean([1,2,3,4,5])": 3.0,
            "sum([1,2,3,4,5])": 15,
            "det([1,2;3,4])": -2.0,
            "simple_operation": 42
        }
    
    def eval(self, expression, nargout=0, **kwargs):
        """Mock MATLAB eval with configurable performance characteristics."""
        self._operation_count += 1
        
        # Simulate operation delay
        if self._operation_delay > 0:
            time.sleep(self._operation_delay)
        
        # Simulate failure rate
        if self._failure_rate > 0 and np.random.random() < self._failure_rate:
            raise RuntimeError(f"Simulated failure for operation {self._operation_count}")
        
        if self._closed:
            raise RuntimeError("MATLAB engine is closed")
        
        # Return cached result or default
        return self._results_cache.get(expression, 1.0)
    
    def quit(self):
        """Mock quit function."""
        self._closed = True


class PerformanceTestFramework:
    """Framework for performance and load testing."""
    
    def __init__(self):
        """Initialize performance test framework."""
        self.results = []
        self.baseline_metrics = {}
        
    def measure_system_resources(self) -> Tuple[float, float]:
        """Measure current system resource usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        return memory_mb, cpu_percent
    
    def benchmark_operation(self, operation_func, operation_name: str, 
                          iterations: int = 1000, warmup_iterations: int = 100) -> PerformanceMetrics:
        """Benchmark a single operation."""
        logger.info(f"Benchmarking {operation_name} with {iterations} iterations")
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                operation_func()
            except Exception:
                pass  # Ignore warmup failures
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Measure baseline resources
        start_memory, _ = self.measure_system_resources()
        
        # Perform benchmarking
        times = []
        successful_ops = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                operation_func()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                successful_ops += 1
            except Exception as e:
                end_time = time.perf_counter()
                # Still record the time even for failed operations
                times.append(end_time - start_time)
                logger.debug(f"Operation {i} failed: {e}")
        
        # Measure final resources
        end_memory, cpu_percent = self.measure_system_resources()
        
        # Calculate metrics
        if times:
            total_time = sum(times)
            min_time = min(times)
            max_time = max(times)
            mean_time = mean(times)
            median_time = median(times)
            std_time = stdev(times) if len(times) > 1 else 0.0
            ops_per_second = successful_ops / total_time if total_time > 0 else 0
        else:
            total_time = min_time = max_time = mean_time = median_time = std_time = ops_per_second = 0
        
        metrics = PerformanceMetrics(
            test_name=operation_name,
            operation_count=iterations,
            total_time=total_time,
            min_time=min_time,
            max_time=max_time,
            mean_time=mean_time,
            median_time=median_time,
            std_time=std_time,
            operations_per_second=ops_per_second,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=cpu_percent,
            success_rate=successful_ops / iterations if iterations > 0 else 0
        )
        
        self.results.append(metrics)
        return metrics
    
    def load_test(self, operation_func, test_name: str, concurrent_users: int = 10, 
                  duration_seconds: int = 30, ramp_up_seconds: int = 5) -> LoadTestResult:
        """Perform load testing with multiple concurrent users."""
        logger.info(f"Starting load test: {test_name} with {concurrent_users} users for {duration_seconds}s")
        
        # Shared state
        results_lock = threading.Lock()
        response_times = []
        successful_ops = 0
        failed_ops = 0
        test_start_time = time.time()
        test_end_time = test_start_time + duration_seconds
        
        # System monitoring
        memory_samples = []
        cpu_samples = []
        
        def monitor_resources():
            """Monitor system resources during the test."""
            while time.time() < test_end_time:
                memory_mb, cpu_percent = self.measure_system_resources()
                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)
                time.sleep(0.5)
        
        def user_workload(user_id: int):
            """Workload for a single user."""
            nonlocal successful_ops, failed_ops
            
            # Ramp up: stagger user start times
            ramp_delay = (user_id / concurrent_users) * ramp_up_seconds
            time.sleep(ramp_delay)
            
            while time.time() < test_end_time:
                start_time = time.perf_counter()
                
                try:
                    operation_func()
                    end_time = time.perf_counter()
                    response_time = end_time - start_time
                    
                    with results_lock:
                        response_times.append(response_time)
                        successful_ops += 1
                        
                except Exception as e:
                    end_time = time.perf_counter()
                    response_time = end_time - start_time
                    
                    with results_lock:
                        response_times.append(response_time)
                        failed_ops += 1
                    
                    logger.debug(f"User {user_id} operation failed: {e}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Start user threads
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            user_futures = [
                executor.submit(user_workload, user_id)
                for user_id in range(concurrent_users)
            ]
            
            # Wait for all users to complete
            for future in as_completed(user_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"User thread failed: {e}")
        
        # Wait for monitoring to complete
        monitor_thread.join(timeout=1)
        
        # Calculate results
        total_ops = successful_ops + failed_ops
        actual_duration = time.time() - test_start_time
        
        if response_times:
            response_times.sort()
            avg_response_time = mean(response_times)
            median_response_time = median(response_times)
            p95_index = int(0.95 * len(response_times))
            p99_index = int(0.99 * len(response_times))
            p95_response_time = response_times[min(p95_index, len(response_times) - 1)]
            p99_response_time = response_times[min(p99_index, len(response_times) - 1)]
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        result = LoadTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            duration_seconds=actual_duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput_ops_per_second=total_ops / actual_duration if actual_duration > 0 else 0,
            peak_memory_mb=max(memory_samples) if memory_samples else 0,
            average_cpu_percent=mean(cpu_samples) if cpu_samples else 0,
            error_rate=failed_ops / total_ops if total_ops > 0 else 0
        )
        
        return result
    
    def stress_test_session_management(self, session_manager, max_sessions: int = 20, 
                                     operations_per_session: int = 50) -> Dict[str, Any]:
        """Stress test session management capabilities."""
        logger.info(f"Stress testing session management: {max_sessions} sessions, {operations_per_session} ops each")
        
        start_time = time.time()
        created_sessions = []
        session_times = []
        operation_times = []
        failed_sessions = 0
        failed_operations = 0
        
        def create_and_test_session(session_id: int):
            nonlocal failed_sessions, failed_operations
            
            session_start = time.time()
            try:
                session = session_manager.get_or_create_session(f"stress_session_{session_id}")
                session_create_time = time.time() - session_start
                session_times.append(session_create_time)
                created_sessions.append(session)
                
                # Perform operations on the session
                for op_id in range(operations_per_session):
                    op_start = time.time()
                    try:
                        result = session.evaluate("1+1")
                        op_time = time.time() - op_start
                        operation_times.append(op_time)
                        
                        assert result == 2, f"Unexpected result: {result}"
                    except Exception as e:
                        failed_operations += 1
                        logger.debug(f"Session {session_id} operation {op_id} failed: {e}")
                
            except Exception as e:
                failed_sessions += 1
                logger.error(f"Session {session_id} creation failed: {e}")
        
        # Create sessions concurrently
        with ThreadPoolExecutor(max_workers=min(max_sessions, 10)) as executor:
            session_futures = [
                executor.submit(create_and_test_session, session_id)
                for session_id in range(max_sessions)
            ]
            
            for future in as_completed(session_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Session thread failed: {e}")
        
        total_time = time.time() - start_time
        
        # Get final pool status
        pool_status = session_manager.get_pool_status()
        
        return {
            'test_name': 'session_management_stress',
            'requested_sessions': max_sessions,
            'successful_sessions': len(created_sessions),
            'failed_sessions': failed_sessions,
            'total_operations': max_sessions * operations_per_session,
            'failed_operations': failed_operations,
            'total_time': total_time,
            'average_session_creation_time': mean(session_times) if session_times else 0,
            'average_operation_time': mean(operation_times) if operation_times else 0,
            'final_pool_status': pool_status,
            'session_creation_rate': len(created_sessions) / total_time if total_time > 0 else 0,
            'operation_throughput': (max_sessions * operations_per_session - failed_operations) / total_time if total_time > 0 else 0
        }
    
    def memory_leak_test(self, operation_func, test_name: str, iterations: int = 1000, 
                        check_interval: int = 100) -> Dict[str, Any]:
        """Test for memory leaks over many operations."""
        logger.info(f"Running memory leak test: {test_name} with {iterations} iterations")
        
        # Force garbage collection before starting
        gc.collect()
        
        initial_memory, _ = self.measure_system_resources()
        memory_measurements = [initial_memory]
        
        for i in range(iterations):
            try:
                operation_func()
            except Exception:
                pass  # Continue testing even with failures
            
            # Measure memory periodically
            if (i + 1) % check_interval == 0:
                gc.collect()  # Force GC to get accurate measurement
                memory_mb, _ = self.measure_system_resources()
                memory_measurements.append(memory_mb)
        
        # Final measurement
        gc.collect()
        final_memory, _ = self.measure_system_resources()
        memory_measurements.append(final_memory)
        
        # Analyze memory growth
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_measurements)
        avg_memory = mean(memory_measurements)
        
        # Detect potential leak (simple heuristic)
        leak_detected = memory_growth > 50  # More than 50MB growth
        
        return {
            'test_name': test_name,
            'iterations': iterations,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'max_memory_mb': max_memory,
            'average_memory_mb': avg_memory,
            'leak_detected': leak_detected,
            'memory_measurements': memory_measurements
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {'error': 'No performance results available'}
        
        # Overall statistics
        total_ops = sum(r.operation_count for r in self.results)
        total_time = sum(r.total_time for r in self.results)
        avg_ops_per_second = mean([r.operations_per_second for r in self.results if r.operations_per_second > 0])
        
        # Best and worst performing tests
        best_throughput = max(self.results, key=lambda r: r.operations_per_second)
        worst_throughput = min(self.results, key=lambda r: r.operations_per_second)
        
        fastest_operation = min(self.results, key=lambda r: r.mean_time)
        slowest_operation = max(self.results, key=lambda r: r.mean_time)
        
        return {
            'summary': {
                'total_tests': len(self.results),
                'total_operations': total_ops,
                'total_execution_time': total_time,
                'average_throughput_ops_per_second': avg_ops_per_second,
                'overall_success_rate': mean([r.success_rate for r in self.results])
            },
            'best_performance': {
                'highest_throughput': {
                    'test': best_throughput.test_name,
                    'ops_per_second': best_throughput.operations_per_second
                },
                'fastest_operation': {
                    'test': fastest_operation.test_name,
                    'mean_time': fastest_operation.mean_time
                }
            },
            'worst_performance': {
                'lowest_throughput': {
                    'test': worst_throughput.test_name,
                    'ops_per_second': worst_throughput.operations_per_second
                },
                'slowest_operation': {
                    'test': slowest_operation.test_name,
                    'mean_time': slowest_operation.mean_time
                }
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'operations_per_second': r.operations_per_second,
                    'mean_time': r.mean_time,
                    'success_rate': r.success_rate,
                    'memory_usage_mb': r.memory_usage_mb
                }
                for r in self.results
            ]
        }


@pytest.fixture
def mock_matlab_module():
    """Fixture that patches the matlab module."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEnginePerformance())
        yield mock_matlab


@pytest.fixture
def slow_mock_matlab_module():
    """Fixture with slower mock operations."""
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEnginePerformance(operation_delay=0.01))
        yield mock_matlab


class TestPerformanceSuite:
    """Performance test suite."""
    
    def test_single_operation_benchmark(self, mock_matlab_module):
        """Test benchmarking of single operations."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = PerformanceTestFramework()
        
        # Benchmark simple operation
        def simple_op():
            return engine.evaluate("1+1")
        
        metrics = framework.benchmark_operation(simple_op, "simple_arithmetic", iterations=100)
        
        assert metrics.test_name == "simple_arithmetic"
        assert metrics.operation_count == 100
        assert metrics.operations_per_second > 0
        assert metrics.success_rate > 0.9  # Should be very high with mocks
        assert metrics.mean_time > 0
    
    def test_load_testing(self, mock_matlab_module):
        """Test load testing capabilities."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = PerformanceTestFramework()
        
        def load_operation():
            return engine.evaluate("simple_operation")
        
        # Short load test for quick testing
        result = framework.load_test(
            load_operation, 
            "arithmetic_load_test",
            concurrent_users=5,
            duration_seconds=2,
            ramp_up_seconds=1
        )
        
        assert result.test_name == "arithmetic_load_test"
        assert result.concurrent_users == 5
        assert result.total_operations > 0
        assert result.throughput_ops_per_second > 0
        assert result.error_rate <= 0.1  # Should be low with mocks
    
    def test_session_management_stress(self, mock_matlab_module):
        """Test session management under stress."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(headless_mode=True, max_sessions=5)
        manager = MATLABSessionManager(config=config)
        
        framework = PerformanceTestFramework()
        
        # Stress test with moderate parameters for quick testing
        result = framework.stress_test_session_management(
            manager,
            max_sessions=10,
            operations_per_session=20
        )
        
        assert result['test_name'] == 'session_management_stress'
        assert result['requested_sessions'] == 10
        assert result['successful_sessions'] > 0
        assert result['session_creation_rate'] > 0
        assert result['operation_throughput'] > 0
        
        manager.close_all_sessions()
    
    def test_memory_leak_detection(self, mock_matlab_module):
        """Test memory leak detection."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = PerformanceTestFramework()
        
        def memory_test_operation():
            # Create some data that should be cleaned up
            result = engine.evaluate("1+1")
            temp_data = [i for i in range(100)]  # Small temporary data
            return result
        
        # Test with moderate iterations for quick testing
        result = framework.memory_leak_test(
            memory_test_operation,
            "memory_leak_test",
            iterations=200,
            check_interval=50
        )
        
        assert result['test_name'] == "memory_leak_test"
        assert result['iterations'] == 200
        assert result['initial_memory_mb'] > 0
        assert result['final_memory_mb'] > 0
        assert len(result['memory_measurements']) > 1
        
        # Should not detect significant leak in simple operations
        assert not result['leak_detected'] or result['memory_growth_mb'] < 100
    
    def test_performance_report_generation(self, mock_matlab_module):
        """Test performance report generation."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = PerformanceTestFramework()
        
        # Run several benchmarks
        def op1():
            return engine.evaluate("1+1")
        
        def op2():
            return engine.evaluate("2+2")
        
        framework.benchmark_operation(op1, "addition_test", iterations=50)
        framework.benchmark_operation(op2, "addition_test_2", iterations=50)
        
        # Generate report
        report = framework.generate_performance_report()
        
        assert 'summary' in report
        assert report['summary']['total_tests'] == 2
        assert report['summary']['total_operations'] == 100
        assert report['summary']['average_throughput_ops_per_second'] > 0
        
        assert 'best_performance' in report
        assert 'worst_performance' in report
        assert 'detailed_results' in report
        assert len(report['detailed_results']) == 2
    
    @pytest.mark.slow
    def test_extended_performance_benchmark(self, slow_mock_matlab_module):
        """Extended performance benchmark with slower operations."""
        from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
        
        config = MATLABConfig(headless_mode=True)
        engine = MATLABEngineWrapper(config=config)
        engine.start()
        
        framework = PerformanceTestFramework()
        
        operations = [
            ("arithmetic", lambda: engine.evaluate("1+1")),
            ("trigonometric", lambda: engine.evaluate("sin(0)")),
            ("matrix_det", lambda: engine.evaluate("det([1,2;3,4])")),
            ("statistical", lambda: engine.evaluate("mean([1,2,3,4,5])"))
        ]
        
        # Run benchmarks
        for op_name, op_func in operations:
            metrics = framework.benchmark_operation(op_func, op_name, iterations=100)
            logger.info(f"{op_name}: {metrics.operations_per_second:.2f} ops/sec")
        
        # Generate comprehensive report
        report = framework.generate_performance_report()
        
        assert len(report['detailed_results']) == 4
        assert all(r['operations_per_second'] > 0 for r in report['detailed_results'])
    
    def test_concurrent_session_performance(self, mock_matlab_module):
        """Test performance with concurrent sessions."""
        from matlab_engine_wrapper import MATLABSessionManager, MATLABConfig
        
        config = MATLABConfig(headless_mode=True, max_sessions=3)
        manager = MATLABSessionManager(config=config)
        
        framework = PerformanceTestFramework()
        
        def concurrent_operation():
            session = manager.get_or_create_session(f"perf_session_{threading.get_ident()}")
            return session.evaluate("1+1")
        
        # Benchmark concurrent operations
        metrics = framework.benchmark_operation(
            concurrent_operation,
            "concurrent_session_ops",
            iterations=100
        )
        
        assert metrics.operations_per_second > 0
        assert metrics.success_rate > 0.8  # Should be high even with session management
        
        # Check pool utilization
        status = manager.get_pool_status()
        assert status['total_sessions'] <= config.max_sessions
        
        manager.close_all_sessions()


if __name__ == "__main__":
    # Run standalone performance tests
    print("Running Comprehensive Performance Test Suite...")
    print("=" * 60)
    
    # Mock the MATLAB module
    mock_matlab = MagicMock()
    mock_matlab.engine = MagicMock()
    mock_matlab.double = MagicMock(side_effect=lambda x: x)
    mock_matlab.logical = MagicMock(side_effect=lambda x: x)
    mock_matlab.engine.start_matlab = Mock(return_value=MockMATLABEnginePerformance())
    
    with patch.dict('sys.modules', {'matlab': mock_matlab, 'matlab.engine': mock_matlab.engine}):
        try:
            from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig, MATLABSessionManager
            
            print("✓ Successfully imported modules for performance testing")
            
            framework = PerformanceTestFramework()
            
            # Single operation benchmarks
            print("\n--- Single Operation Benchmarks ---")
            config = MATLABConfig(headless_mode=True)
            engine = MATLABEngineWrapper(config=config)
            engine.start()
            
            operations = [
                ("arithmetic", lambda: engine.evaluate("1+1")),
                ("trigonometric", lambda: engine.evaluate("sin(0)")),
                ("matrix_operation", lambda: engine.evaluate("det([1,2;3,4])")),
                ("statistical", lambda: engine.evaluate("mean([1,2,3,4,5])"))
            ]
            
            for op_name, op_func in operations:
                metrics = framework.benchmark_operation(op_func, op_name, iterations=500)
                print(f"✓ {op_name:15}: {metrics.operations_per_second:8.1f} ops/sec, "
                      f"{metrics.mean_time*1000:6.2f}ms avg, {metrics.success_rate*100:5.1f}% success")
            
            # Load testing
            print("\n--- Load Testing ---")
            def load_operation():
                return engine.evaluate("simple_operation")
            
            load_result = framework.load_test(
                load_operation,
                "arithmetic_load_test",
                concurrent_users=8,
                duration_seconds=5,
                ramp_up_seconds=2
            )
            
            print(f"✓ Load test completed:")
            print(f"  Users: {load_result.concurrent_users}")
            print(f"  Duration: {load_result.duration_seconds:.1f}s")
            print(f"  Total ops: {load_result.total_operations}")
            print(f"  Throughput: {load_result.throughput_ops_per_second:.1f} ops/sec")
            print(f"  Avg response: {load_result.average_response_time*1000:.2f}ms")
            print(f"  P95 response: {load_result.p95_response_time*1000:.2f}ms")
            print(f"  Error rate: {load_result.error_rate*100:.1f}%")
            
            # Session management stress test
            print("\n--- Session Management Stress Test ---")
            config = MATLABConfig(headless_mode=True, max_sessions=5)
            manager = MATLABSessionManager(config=config)
            
            stress_result = framework.stress_test_session_management(
                manager,
                max_sessions=15,
                operations_per_session=30
            )
            
            print(f"✓ Session stress test completed:")
            print(f"  Requested sessions: {stress_result['requested_sessions']}")
            print(f"  Successful sessions: {stress_result['successful_sessions']}")
            print(f"  Session creation rate: {stress_result['session_creation_rate']:.1f} sessions/sec")
            print(f"  Operation throughput: {stress_result['operation_throughput']:.1f} ops/sec")
            print(f"  Final pool status: {stress_result['final_pool_status']['total_sessions']} active sessions")
            
            # Memory leak test
            print("\n--- Memory Leak Test ---")
            def memory_test_op():
                result = engine.evaluate("1+1")
                temp_data = [i**2 for i in range(50)]  # Some temporary computation
                return result
            
            memory_result = framework.memory_leak_test(
                memory_test_op,
                "memory_leak_test",
                iterations=500,
                check_interval=100
            )
            
            print(f"✓ Memory leak test completed:")
            print(f"  Iterations: {memory_result['iterations']}")
            print(f"  Initial memory: {memory_result['initial_memory_mb']:.1f} MB")
            print(f"  Final memory: {memory_result['final_memory_mb']:.1f} MB")
            print(f"  Memory growth: {memory_result['memory_growth_mb']:.1f} MB")
            print(f"  Leak detected: {memory_result['leak_detected']}")
            
            # Generate comprehensive report
            print("\n--- Performance Report ---")
            report = framework.generate_performance_report()
            
            print(f"✓ Performance report generated:")
            print(f"  Total tests: {report['summary']['total_tests']}")
            print(f"  Total operations: {report['summary']['total_operations']}")
            print(f"  Average throughput: {report['summary']['average_throughput_ops_per_second']:.1f} ops/sec")
            print(f"  Overall success rate: {report['summary']['overall_success_rate']*100:.1f}%")
            
            print(f"\n  Best performance:")
            print(f"    Highest throughput: {report['best_performance']['highest_throughput']['test']} "
                  f"({report['best_performance']['highest_throughput']['ops_per_second']:.1f} ops/sec)")
            print(f"    Fastest operation: {report['best_performance']['fastest_operation']['test']} "
                  f"({report['best_performance']['fastest_operation']['mean_time']*1000:.2f}ms)")
            
            # Cleanup
            engine.close()
            manager.close_all_sessions()
            
            print("\n" + "=" * 60)
            print("All performance tests completed successfully!")
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            raise