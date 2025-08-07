#!/usr/bin/env python3
"""
MATLAB Engine API Health Check Script
Validates system health and MATLAB Engine availability
"""

import sys
import time
import argparse
import subprocess
import psutil
from pathlib import Path
import json


class HealthChecker:
    """Comprehensive health checker for MATLAB Engine API"""
    
    def __init__(self):
        self.checks = []
        self.startup_mode = False
        
    def add_check(self, name, check_func, required=True):
        """Add a health check function"""
        self.checks.append({
            'name': name,
            'func': check_func,
            'required': required
        })
    
    def run_checks(self):
        """Run all health checks and return results"""
        results = {
            'timestamp': time.time(),
            'checks': {},
            'overall_status': 'healthy',
            'errors': []
        }
        
        for check in self.checks:
            try:
                start_time = time.time()
                status, message, details = check['func']()
                duration = time.time() - start_time
                
                results['checks'][check['name']] = {
                    'status': status,
                    'message': message,
                    'details': details,
                    'duration': duration,
                    'required': check['required']
                }
                
                if not status and check['required']:
                    results['overall_status'] = 'unhealthy'
                    results['errors'].append(f"{check['name']}: {message}")
                    
            except Exception as e:
                results['checks'][check['name']] = {
                    'status': False,
                    'message': f"Check failed with exception: {str(e)}",
                    'details': {},
                    'duration': 0,
                    'required': check['required']
                }
                
                if check['required']:
                    results['overall_status'] = 'unhealthy'
                    results['errors'].append(f"{check['name']}: {str(e)}")
        
        return results
    
    def check_system_resources(self):
        """Check system resource availability"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Check thresholds
            if cpu_percent > 95:
                return False, f"CPU usage too high: {cpu_percent}%", details
            if memory.percent > 90:
                return False, f"Memory usage too high: {memory.percent}%", details
            if disk.percent > 95:
                return False, f"Disk usage too high: {disk.percent}%", details
            
            return True, "System resources healthy", details
            
        except Exception as e:
            return False, f"Failed to check system resources: {str(e)}", {}
    
    def check_python_environment(self):
        """Check Python environment and dependencies"""
        try:
            import numpy
            import scipy
            import matplotlib
            
            details = {
                'python_version': sys.version,
                'numpy_version': numpy.__version__,
                'scipy_version': scipy.__version__,
                'matplotlib_version': matplotlib.__version__
            }
            
            # Test basic numpy operation
            test_array = numpy.array([1, 2, 3, 4, 5])
            if numpy.sum(test_array) != 15:
                return False, "NumPy basic test failed", details
            
            return True, "Python environment healthy", details
            
        except ImportError as e:
            return False, f"Missing Python dependency: {str(e)}", {}
        except Exception as e:
            return False, f"Python environment check failed: {str(e)}", {}
    
    def check_matlab_engine_availability(self):
        """Check if MATLAB Engine can be imported and initialized"""
        try:
            if self.startup_mode:
                # During startup, just check if we can import
                import matlab.engine
                return True, "MATLAB Engine import successful", {'mode': 'import_only'}
            else:
                # During runtime, test actual engine startup
                import matlab.engine
                
                # Try to start engine with timeout
                engines = matlab.engine.find_matlab()
                if engines:
                    engine = matlab.engine.connect_matlab(engines[0])
                else:
                    engine = matlab.engine.start_matlab('-nojvm -nodisplay')
                
                # Test basic operation
                result = engine.eval('2 + 2')
                engine.quit()
                
                if abs(result - 4.0) > 1e-10:
                    return False, "MATLAB Engine basic test failed", {'result': result}
                
                return True, "MATLAB Engine healthy", {'result': result}
                
        except ImportError:
            return False, "MATLAB Engine not available (import failed)", {}
        except Exception as e:
            return False, f"MATLAB Engine check failed: {str(e)}", {}
    
    def check_application_files(self):
        """Check if required application files exist"""
        required_files = [
            '/app/src/python/matlab_engine_wrapper.py',
            '/app/src/python/performance_monitor.py',
            '/app/src/python/config_manager.py'
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        details = {
            'required_files': required_files,
            'existing_files': existing_files,
            'missing_files': missing_files
        }
        
        if missing_files:
            return False, f"Missing required files: {missing_files}", details
        
        return True, "All required files present", details
    
    def check_display_availability(self):
        """Check if display is available for headless MATLAB"""
        try:
            display = subprocess.run(
                ['xdpyinfo'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            details = {'display_env': sys.os.environ.get('DISPLAY', 'Not set')}
            
            if display.returncode == 0:
                return True, "Display available", details
            else:
                # Not critical for headless operation
                return True, "Display not available (headless mode)", details
                
        except subprocess.TimeoutExpired:
            return False, "Display check timed out", {}
        except FileNotFoundError:
            return True, "xdpyinfo not available (headless mode)", {}
        except Exception as e:
            return False, f"Display check failed: {str(e)}", {}
    
    def check_network_connectivity(self):
        """Check basic network connectivity (if needed)"""
        try:
            # Simple connectivity test
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True, "Network connectivity available", {}
        except Exception as e:
            # Network might not be required for all deployments
            return True, f"Network check failed (may not be required): {str(e)}", {}


def main():
    parser = argparse.ArgumentParser(description='MATLAB Engine API Health Check')
    parser.add_argument('--startup-check', action='store_true',
                       help='Run startup health checks (less intensive)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    checker.startup_mode = args.startup_check
    
    # Add all health checks
    checker.add_check('system_resources', checker.check_system_resources, required=True)
    checker.add_check('python_environment', checker.check_python_environment, required=True)
    checker.add_check('application_files', checker.check_application_files, required=True)
    checker.add_check('display_availability', checker.check_display_availability, required=False)
    checker.add_check('network_connectivity', checker.check_network_connectivity, required=False)
    
    # Only check MATLAB engine during runtime checks (not startup)
    if not args.startup_check:
        checker.add_check('matlab_engine', checker.check_matlab_engine_availability, required=True)
    
    # Run health checks
    results = checker.run_checks()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"Health Check Results - Status: {results['overall_status'].upper()}")
        print(f"Timestamp: {time.ctime(results['timestamp'])}")
        print()
        
        for check_name, check_result in results['checks'].items():
            status_icon = "✓" if check_result['status'] else "✗"
            required_text = " (REQUIRED)" if check_result['required'] else " (optional)"
            
            print(f"{status_icon} {check_name}{required_text}: {check_result['message']}")
            
            if args.verbose and check_result['details']:
                for key, value in check_result['details'].items():
                    print(f"    {key}: {value}")
            
            if args.verbose:
                print(f"    Duration: {check_result['duration']:.3f}s")
            
            print()
        
        if results['errors']:
            print("ERRORS:")
            for error in results['errors']:
                print(f"  - {error}")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'healthy' else 1)


if __name__ == '__main__':
    main()