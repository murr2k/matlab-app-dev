#!/bin/bash
# MATLAB Engine API Production Entrypoint Script
# Handles initialization, health checks, and graceful shutdown

set -euo pipefail

# Configuration
MATLAB_ENGINE_LOG_LEVEL=${MATLAB_ENGINE_LOG_LEVEL:-INFO}
MATLAB_ENGINE_PORT=${MATLAB_ENGINE_PORT:-8000}
MATLAB_ENGINE_WORKERS=${MATLAB_ENGINE_WORKERS:-4}
MATLAB_ENGINE_TIMEOUT=${MATLAB_ENGINE_TIMEOUT:-300}

# Logging setup
exec > >(tee -a /app/logs/matlab-engine.log)
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting MATLAB Engine API container..."
log "Configuration:"
log "  - Log Level: ${MATLAB_ENGINE_LOG_LEVEL}"
log "  - Port: ${MATLAB_ENGINE_PORT}"
log "  - Workers: ${MATLAB_ENGINE_WORKERS}"
log "  - Timeout: ${MATLAB_ENGINE_TIMEOUT}s"

# Initialize virtual display for headless MATLAB
if [ "${DISPLAY:-}" = ":99" ]; then
    log "Starting virtual display for headless mode..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    XVFB_PID=$!
    
    # Function to cleanup Xvfb on exit
    cleanup_xvfb() {
        if [ -n "${XVFB_PID:-}" ]; then
            log "Stopping virtual display..."
            kill $XVFB_PID 2>/dev/null || true
        fi
    }
    trap cleanup_xvfb EXIT
    
    # Wait for Xvfb to start
    sleep 2
fi

# Health check before starting main application
log "Running pre-startup health checks..."
if ! python /app/healthcheck.py --startup-check; then
    log "ERROR: Startup health check failed"
    exit 1
fi

# Initialize MATLAB Engine
log "Initializing MATLAB Engine..."
cd /app/src/python

# Run initialization script
python -c "
import sys
import os
sys.path.append('/app/src/python')

from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
from performance_monitor import PerformanceMonitor

try:
    print('Testing MATLAB Engine connection...')
    config = MATLABConfig(
        startup_options=['-nojvm', '-nodisplay', '-nosplash'],
        headless_mode=True,
        performance_monitoring=True,
        session_timeout=300
    )
    
    with MATLABEngineWrapper(config=config) as engine:
        # Test basic functionality
        result = engine.evaluate('2 + 2')
        print(f'MATLAB Engine test: 2 + 2 = {result}')
        
        if abs(result - 4.0) > 1e-10:
            raise ValueError('MATLAB Engine basic test failed')
        
        print('MATLAB Engine initialization successful')
        
except Exception as e:
    print(f'ERROR: MATLAB Engine initialization failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log "ERROR: MATLAB Engine initialization failed"
    exit 1
fi

log "MATLAB Engine initialized successfully"

# Signal handling for graceful shutdown
shutdown_handler() {
    log "Received shutdown signal, gracefully shutting down..."
    
    # Stop any running MATLAB processes
    pkill -f matlab 2>/dev/null || true
    
    # Give processes time to shutdown gracefully
    sleep 5
    
    log "Shutdown complete"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT

# Determine run mode based on arguments
if [ $# -eq 0 ]; then
    # Default: Run as web service (if implemented)
    log "Starting MATLAB Engine API web service..."
    
    # Check if web service is available
    if python -c "import uvicorn" 2>/dev/null; then
        # Run with Uvicorn (FastAPI)
        exec uvicorn main:app \
            --host 0.0.0.0 \
            --port ${MATLAB_ENGINE_PORT} \
            --workers ${MATLAB_ENGINE_WORKERS} \
            --timeout-keep-alive ${MATLAB_ENGINE_TIMEOUT} \
            --log-level ${MATLAB_ENGINE_LOG_LEVEL,,} \
            --access-log \
            --no-server-header
    else
        # Run interactive Python session for testing
        log "Web service not configured, starting interactive mode..."
        exec python -i -c "
print('MATLAB Engine API Interactive Mode')
print('Available modules:')
print('  - matlab_engine_wrapper')
print('  - hybrid_simulations')
print('  - performance_monitor')
print('  - config_manager')
print('')
print('Example usage:')
print('  from matlab_engine_wrapper import MATLABEngineWrapper')
print('  engine = MATLABEngineWrapper()')
print('')
"
    fi
elif [ "$1" = "test" ]; then
    # Run test suite
    log "Running test suite..."
    exec python -m pytest tests/ -v
    
elif [ "$1" = "benchmark" ]; then
    # Run performance benchmarks
    log "Running performance benchmarks..."
    exec python -c "
from matlab_engine_wrapper import MATLABEngineWrapper, MATLABConfig
from performance_monitor import PerformanceMonitor
import time
import json

config = MATLABConfig(
    startup_options=['-nojvm', '-nodisplay'],
    headless_mode=True,
    performance_monitoring=True
)

monitor = PerformanceMonitor()

with MATLABEngineWrapper(config=config) as engine:
    monitor.start_monitoring()
    
    print('Running benchmarks...')
    
    # Matrix operations benchmark
    start = time.time()
    engine.evaluate('A = randn(1000); B = A * A;')
    matrix_time = time.time() - start
    
    # FFT benchmark  
    start = time.time()
    engine.evaluate('x = randn(1, 50000); y = fft(x);')
    fft_time = time.time() - start
    
    report = monitor.stop_monitoring_and_report()
    
    results = {
        'matrix_benchmark': matrix_time,
        'fft_benchmark': fft_time,
        'system_metrics': report
    }
    
    print(json.dumps(results, indent=2, default=str))
"

elif [ "$1" = "demo" ]; then
    # Run comprehensive demo
    log "Running comprehensive demo..."
    exec python demo_comprehensive.py
    
else
    # Pass through any other commands
    log "Executing custom command: $*"
    exec "$@"
fi