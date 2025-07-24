# MATLAB Development Environment - Project Reinvocation Prompt

## Project Overview
You are working on a MATLAB development environment for physics simulations and Simulink modeling. The project is located at `/home/murr2k/projects/matlab` and is pushed to GitHub at `https://github.com/murr2k/matlab-app-dev` (main branch).

## Current State
- ✅ Full MATLAB project structure implemented with src/, tests/, and examples/
- ✅ Physics simulations: pendulum dynamics, particle motion, wave equation solver
- ✅ Simulink model for mass-spring-damper system
- ✅ Comprehensive test suite with unit and integration tests
- ✅ CI/CD pipeline with GitHub Actions using matlab-actions
- ✅ GitHub Pages documentation deployed successfully
- ✅ All CI/CD pipeline issues resolved (deprecated functions, test errors, headless plotting)

## Key Technical Details

### Project Structure
```
matlab/
├── src/
│   ├── simulations/      # Physics simulation implementations
│   ├── utils/            # Helper functions
│   └── +mathutils/       # MATLAB package for utilities
├── tests/               # Unit and integration tests
├── examples/            # Demo scripts and documentation generation
├── models/              # Simulink models
├── buildfile.m          # MATLAB build configuration
└── .github/workflows/   # CI/CD pipeline
```

### Important Files
- `buildfile.m`: Defines build tasks (test, check, package, release)
- `startup.m`: Initializes project paths
- `generate_docs.m`: Custom documentation generator with plot support
- `.github/workflows/matlab-ci.yml`: CI/CD pipeline configuration

### Recent Fixes Applied
1. **HTML Rendering Issue**: Fixed literal \n characters in documentation by wrapping HTML content in `sprintf()`
2. **Deprecated Functions**: Replaced `caxis` with `clim` for MATLAB R2023b compatibility
3. **Test Error Handling**: Updated error identifiers from 'error:' to 'MATLAB:' format
4. **Headless Plotting**: Used virtual display (Xvfb) for generating plots in CI environment

### MATLAB Drive Access
- **Windows Mount**: `/mnt/c/Users/murr2/MATLAB Drive`
- **Linux Symlink**: `~/matlab-cloud`
- Contains: semiprime generation scripts, FileExchange packages, and other MATLAB files

## Key Commands
```bash
# Run tests locally
matlab -batch "buildtool test"

# Generate documentation
matlab -batch "addpath('examples'); generate_docs"

# Check code quality
matlab -batch "buildtool check"

# View CI/CD status
# Visit: https://github.com/murr2k/matlab-app-dev/actions
```

## GitHub Pages Documentation
- URL: https://murr2k.github.io/matlab-app-dev/
- Auto-deployed on push to main branch
- Contains examples with plots and mathematical formulas

## Next Steps / Potential Enhancements
1. Consider integrating MATLAB Drive files with the project
2. Add more physics simulations (electromagnetics, fluid dynamics)
3. Implement interactive GUI using App Designer
4. Add performance benchmarking dashboard
5. Integrate with MATLAB Online or MATLAB Production Server

## Important Notes
- Always use `sprintf()` when generating HTML with escape sequences
- Use `clim` instead of `caxis` for R2023b+ compatibility
- Ensure all error identifiers follow 'MATLAB:' convention
- Virtual display required for headless plot generation in CI

---
*Last Updated: Current session*
*GitHub Repository: https://github.com/murr2k/matlab-app-dev*