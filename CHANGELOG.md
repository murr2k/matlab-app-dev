# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive roadmap for MATLAB Engine API integration (#1)
- Planned enhancements for matlab-computational-engineer agent (#2)
- Graphical solution termination detection specifications (#3)
- 10 specialized problem domain implementation plan (#4)
- Detailed wiki documentation for project overview
- REINVOCATION_PROMPT.md for session context restoration

### Changed
- Updated README with current features and comprehensive roadmap
- Enhanced project documentation with planned features

## [1.0.0] - 2025-01-07

### Added
- Core physics simulations implementation
  - Pendulum simulation with ODE45 integration
  - Particle dynamics with 3D motion support
  - Wave equation solver with finite differences
- Simulink mass-spring-damper model
- Comprehensive test suite
  - Unit tests for all physics simulations
  - Integration tests for workflow validation
  - Performance benchmarks
- CI/CD pipeline with GitHub Actions
  - Automated testing with matlab-actions
  - Code quality checks and coverage reports
  - Build system with buildfile.m
- MATLAB Drive integration support
  - Windows mount path access (WSL)
  - Linux symlink configuration
- Project structure and organization
  - Modular source code organization
  - Separated test suites
  - Documentation framework

### Fixed
- Deprecated `caxis` replaced with `clim` for MATLAB R2023b compatibility
- HTML rendering issues with proper sprintf() escape sequences
- Headless environment display issues for CI/CD
- Test identifier corrections for proper error reporting
- Wave propagation test numerical stability

### Changed
- Updated all deprecated MATLAB functions to current API
- Simplified documentation generation for CI environment
- Improved demo scripts for publish-friendly output

## [0.1.0] - 2025-01-06

### Added
- Initial project setup
- Basic MATLAB project structure
- GitHub repository initialization
- MIT License

---

## Version History

- **1.0.0** - First stable release with complete physics simulation framework
- **0.1.0** - Initial project creation

## Upcoming Releases

### [1.1.0] - MATLAB Engine API Integration
- Python-MATLAB bridge implementation
- Session management and workspace persistence
- Async computation support

### [2.0.0] - Intelligent Computational Agent
- Enhanced matlab-computational-engineer
- Smart stopping conditions
- Symbolic math support
- Graphical solution termination

### [3.0.0] - Comprehensive Domain Support
- 10 specialized problem domains
- Domain-specific termination logic
- Cross-domain problem handling
- Plugin architecture for extensibility