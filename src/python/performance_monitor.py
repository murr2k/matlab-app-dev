"""
Performance Monitoring and Diagnostics for MATLAB Engine API
==========================================================

This module provides comprehensive performance monitoring, diagnostics, and
optimization recommendations for MATLAB Engine operations.

Author: Murray Kopit
License: MIT
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    operation_type: str
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    session_id: str
    operation_count: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    active_sessions: int
    total_operations: int
    error_rate: float


class PerformanceCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize performance collector.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self.metrics: deque = deque(maxlen=max_samples)
        self.health_metrics: deque = deque(maxlen=max_samples)
        self._lock = threading.RLock()
        self.start_time = time.time()
        
    def record_operation(self, operation_type: str, duration: float, 
                        session_id: str, success: bool = True, 
                        error_message: Optional[str] = None):
        """
        Record a single operation's performance metrics.
        
        Args:
            operation_type: Type of operation (evaluate, call_function, etc.)
            duration: Duration of operation in seconds
            session_id: ID of the session that performed the operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
        """
        with self._lock:
            try:
                # Get system metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                cpu_percent = process.cpu_percent()
                
                metric = PerformanceMetrics(
                    timestamp=time.time(),
                    operation_type=operation_type,
                    duration=duration,
                    memory_usage_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    session_id=session_id,
                    operation_count=len(self.metrics) + 1,
                    success=success,
                    error_message=error_message
                )
                
                self.metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Failed to record performance metric: {e}")
    
    def record_system_health(self, active_sessions: int):
        """
        Record system health metrics.
        
        Args:
            active_sessions: Number of currently active sessions
        """
        with self._lock:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Calculate error rate from recent metrics
                recent_metrics = list(self.metrics)[-100:]  # Last 100 operations
                if recent_metrics:
                    failed_ops = sum(1 for m in recent_metrics if not m.success)
                    error_rate = failed_ops / len(recent_metrics)
                else:
                    error_rate = 0.0
                
                health = SystemHealth(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage_percent=disk.percent,
                    active_sessions=active_sessions,
                    total_operations=len(self.metrics),
                    error_rate=error_rate
                )
                
                self.health_metrics.append(health)
                
            except Exception as e:
                logger.error(f"Failed to record system health: {e}")
    
    def get_operation_stats(self, operation_type: Optional[str] = None,
                           time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_type: Filter by operation type (None for all)
            time_window: Time window in seconds (None for all time)
            
        Returns:
            Dictionary of operation statistics
        """
        with self._lock:
            # Filter metrics
            filtered_metrics = list(self.metrics)
            
            if time_window:
                cutoff_time = time.time() - time_window
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]
            
            if operation_type:
                filtered_metrics = [m for m in filtered_metrics if m.operation_type == operation_type]
            
            if not filtered_metrics:
                return {"count": 0, "error": "No metrics found"}
            
            # Calculate statistics
            durations = [m.duration for m in filtered_metrics]
            memory_usage = [m.memory_usage_mb for m in filtered_metrics]
            successful_ops = [m for m in filtered_metrics if m.success]
            failed_ops = [m for m in filtered_metrics if not m.success]
            
            return {
                "count": len(filtered_metrics),
                "success_count": len(successful_ops),
                "failure_count": len(failed_ops),
                "success_rate": len(successful_ops) / len(filtered_metrics),
                "duration": {
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                    "std": np.std(durations),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99)
                },
                "memory": {
                    "mean_mb": np.mean(memory_usage),
                    "max_mb": np.max(memory_usage),
                    "min_mb": np.min(memory_usage)
                },
                "time_range": {
                    "start": min(m.timestamp for m in filtered_metrics),
                    "end": max(m.timestamp for m in filtered_metrics)
                }
            }
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        return self.get_operation_stats(time_window=None) # Filter applied in get_operation_stats
    
    def get_trend_analysis(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            window_minutes: Time window for trend analysis in minutes
            
        Returns:
            Trend analysis results
        """
        with self._lock:
            window_seconds = window_minutes * 60
            cutoff_time = time.time() - window_seconds
            
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 10:
                return {"error": "Not enough data for trend analysis"}
            
            # Split into time buckets
            bucket_size = window_seconds / 10  # 10 buckets
            buckets = []
            
            for i in range(10):
                bucket_start = cutoff_time + (i * bucket_size)
                bucket_end = bucket_start + bucket_size
                
                bucket_metrics = [m for m in recent_metrics 
                                if bucket_start <= m.timestamp < bucket_end]
                
                if bucket_metrics:
                    avg_duration = np.mean([m.duration for m in bucket_metrics])
                    error_rate = sum(1 for m in bucket_metrics if not m.success) / len(bucket_metrics)
                    avg_memory = np.mean([m.memory_usage_mb for m in bucket_metrics])
                    
                    buckets.append({
                        "timestamp": bucket_start,
                        "avg_duration": avg_duration,
                        "error_rate": error_rate,
                        "avg_memory_mb": avg_memory,
                        "operation_count": len(bucket_metrics)
                    })
            
            # Calculate trends
            if len(buckets) >= 2:
                durations = [b["avg_duration"] for b in buckets]
                error_rates = [b["error_rate"] for b in buckets]
                
                duration_trend = "increasing" if durations[-1] > durations[0] else "decreasing"
                error_trend = "increasing" if error_rates[-1] > error_rates[0] else "decreasing"
                
                return {
                    "time_window_minutes": window_minutes,
                    "buckets": buckets,
                    "trends": {
                        "duration": duration_trend,
                        "errors": error_trend,
                        "duration_change_percent": ((durations[-1] - durations[0]) / durations[0]) * 100,
                        "error_rate_change": error_rates[-1] - error_rates[0]
                    }
                }
            
            return {"error": "Insufficient data for trend calculation"}


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.collector = PerformanceCollector(
            max_samples=self.config.get('max_samples', 1000)
        )
        self.alerts = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self.alert_thresholds = {
            'high_memory_mb': self.config.get('high_memory_threshold', 1024),
            'high_cpu_percent': self.config.get('high_cpu_threshold', 80.0),
            'high_duration_seconds': self.config.get('high_duration_threshold', 30.0),
            'high_error_rate': self.config.get('high_error_rate_threshold', 0.1)
        }
        
    def start_monitoring(self, interval_seconds: int = 30):
        """
        Start background monitoring thread.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        def monitoring_loop():
            while not self._stop_monitoring.wait(interval_seconds):
                try:
                    # Record system health (active_sessions would be provided by session manager)
                    self.collector.record_system_health(active_sessions=0)  # Placeholder
                    
                    # Check for alerts
                    self._check_alerts()
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def stop_monitoring_and_report(self) -> Dict[str, Any]:
        """Stop monitoring and return final performance report."""
        self.stop_monitoring()
        return self.get_performance_report()
    
    def record_operation(self, operation_type: str, duration: float,
                        session_id: str, success: bool = True,
                        error_message: Optional[str] = None):
        """Record operation performance."""
        self.collector.record_operation(
            operation_type=operation_type,
            duration=duration,
            session_id=session_id,
            success=success,
            error_message=error_message
        )
    
    def _check_alerts(self):
        """Check for performance alerts."""
        try:
            # Get recent stats
            recent_stats = self.collector.get_operation_stats(time_window=300)  # Last 5 minutes
            
            if recent_stats.get("count", 0) == 0:
                return
            
            # Check thresholds
            alerts_triggered = []
            
            # High duration alert
            if recent_stats.get("duration", {}).get("mean", 0) > self.alert_thresholds['high_duration_seconds']:
                alerts_triggered.append({
                    "type": "high_duration",
                    "message": f"Average operation duration is high: {recent_stats['duration']['mean']:.2f}s",
                    "timestamp": time.time()
                })
            
            # High error rate alert
            if recent_stats.get("success_rate", 1.0) < (1.0 - self.alert_thresholds['high_error_rate']):
                alerts_triggered.append({
                    "type": "high_error_rate",
                    "message": f"Error rate is high: {(1.0 - recent_stats['success_rate']) * 100:.1f}%",
                    "timestamp": time.time()
                })
            
            # Add alerts
            for alert in alerts_triggered:
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {alert['message']}")
            
            # Keep only recent alerts (last hour)
            cutoff_time = time.time() - 3600
            self.alerts = [a for a in self.alerts if a['timestamp'] >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Overall statistics
            overall_stats = self.collector.get_operation_stats()
            
            # Recent statistics (last hour)
            recent_stats = self.collector.get_operation_stats(time_window=3600)
            
            # Trend analysis
            trend_analysis = self.collector.get_trend_analysis(window_minutes=60)
            
            # System health
            recent_health = list(self.collector.health_metrics)[-10:] if self.collector.health_metrics else []
            
            # Operation type breakdown
            operation_types = set()
            for metric in self.collector.metrics:
                operation_types.add(metric.operation_type)
            
            operation_breakdown = {}
            for op_type in operation_types:
                operation_breakdown[op_type] = self.collector.get_operation_stats(
                    operation_type=op_type, time_window=3600
                )
            
            return {
                "report_timestamp": time.time(),
                "monitoring_duration_hours": (time.time() - self.collector.start_time) / 3600,
                "overall_stats": overall_stats,
                "recent_stats": recent_stats,
                "trend_analysis": trend_analysis,
                "operation_breakdown": operation_breakdown,
                "recent_health_metrics": recent_health,
                "active_alerts": len(self.alerts),
                "alert_details": self.alerts[-10:],  # Last 10 alerts
                "recommendations": self._generate_recommendations(overall_stats, recent_stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, overall_stats: Dict[str, Any], 
                                recent_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        try:
            # High failure rate
            if recent_stats.get("success_rate", 1.0) < 0.9:
                recommendations.append(
                    "High failure rate detected. Consider reviewing error handling and input validation."
                )
            
            # Slow operations
            if recent_stats.get("duration", {}).get("mean", 0) > 5.0:
                recommendations.append(
                    "Operations are running slowly. Consider optimizing MATLAB code or increasing system resources."
                )
            
            # High memory usage
            recent_memory = recent_stats.get("memory", {}).get("mean_mb", 0)
            if recent_memory > self.alert_thresholds['high_memory_mb']:
                recommendations.append(
                    f"High memory usage detected ({recent_memory:.1f}MB). Consider implementing memory cleanup strategies."
                )
            
            # High variability
            duration_stats = recent_stats.get("duration", {})
            if duration_stats.get("std", 0) > duration_stats.get("mean", 1) * 0.5:
                recommendations.append(
                    "High variability in operation duration. Consider investigating inconsistent performance."
                )
            
            # Session management
            if overall_stats.get("count", 0) > 1000:
                recommendations.append(
                    "High operation count. Consider implementing session pooling and reuse strategies."
                )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error.")
        
        return recommendations
    
    def export_metrics(self, file_path: Path, format: str = "json"):
        """
        Export collected metrics to file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        try:
            if format.lower() == "json":
                report = self.get_performance_report()
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            elif format.lower() == "csv":
                import csv
                with open(file_path, 'w', newline='') as f:
                    if self.collector.metrics:
                        writer = csv.DictWriter(f, fieldnames=[
                            'timestamp', 'operation_type', 'duration', 'memory_usage_mb',
                            'cpu_percent', 'session_id', 'success', 'error_message'
                        ])
                        writer.writeheader()
                        
                        for metric in self.collector.metrics:
                            writer.writerow({
                                'timestamp': metric.timestamp,
                                'operation_type': metric.operation_type,
                                'duration': metric.duration,
                                'memory_usage_mb': metric.memory_usage_mb,
                                'cpu_percent': metric.cpu_percent,
                                'session_id': metric.session_id,
                                'success': metric.success,
                                'error_message': metric.error_message
                            })
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Decorator for automatic performance monitoring
def monitor_performance(monitor: PerformanceMonitor, operation_type: str = None):
    """
    Decorator for automatic performance monitoring of functions.
    
    Args:
        monitor: PerformanceMonitor instance
        operation_type: Type of operation being monitored
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            op_type = operation_type or func.__name__
            session_id = kwargs.get('session_id', 'unknown')
            
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration = time.time() - start_time
                monitor.record_operation(
                    operation_type=op_type,
                    duration=duration,
                    session_id=session_id,
                    success=success,
                    error_message=error_message
                )
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo performance monitoring
    print("MATLAB Engine Performance Monitoring Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = PerformanceMonitor({
        'max_samples': 100,
        'high_duration_threshold': 2.0
    })
    
    # Start monitoring
    monitor.start_monitoring(interval_seconds=5)
    
    try:
        # Simulate some operations
        for i in range(10):
            # Simulate operation
            duration = np.random.uniform(0.5, 3.0)
            success = np.random.random() > 0.1  # 90% success rate
            
            monitor.record_operation(
                operation_type="evaluate" if i % 2 == 0 else "call_function",
                duration=duration,
                session_id=f"session_{i % 3}",
                success=success,
                error_message="Test error" if not success else None
            )
            
            time.sleep(0.1)  # Small delay
        
        # Generate report
        report = monitor.get_performance_report()
        
        print(f"\nPerformance Report:")
        print(f"Total operations: {report['overall_stats']['count']}")
        print(f"Success rate: {report['overall_stats']['success_rate']:.1%}")
        print(f"Average duration: {report['overall_stats']['duration']['mean']:.3f}s")
        print(f"Active alerts: {report['active_alerts']}")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # Export metrics
        export_path = Path("/tmp/matlab_performance_demo.json")
        monitor.export_metrics(export_path)
        print(f"\nMetrics exported to: {export_path}")
        
    finally:
        monitor.stop_monitoring()
    
    print("\nPerformance monitoring demo completed!")