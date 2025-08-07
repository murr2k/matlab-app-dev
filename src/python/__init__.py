"""
MATLAB Engine API for Python Integration
=========================================

This package provides Python integration with MATLAB using the MATLAB Engine API.
It enables direct Python-MATLAB communication for the physics simulation framework.

Modules:
--------
- matlab_engine_wrapper: Core wrapper for MATLAB Engine API
- hybrid_simulations: Python-MATLAB simulation bridge
- computational_interface: Interface for matlab-computational-engineer

Usage:
------
    from matlab_engine_wrapper import MATLABEngineWrapper
    
    engine = MATLABEngineWrapper()
    engine.start()
    result = engine.evaluate("sqrt(64)")
    engine.close()

Author: Murray Kopit
License: MIT
"""

__version__ = "1.1.0"
__author__ = "Murray Kopit"
__email__ = "murr2k@gmail.com"

# Package metadata
__all__ = [
    "MATLABEngineWrapper",
    "HybridSimulations",
    "ComputationalInterface"
]