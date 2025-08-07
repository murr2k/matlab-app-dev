# MATLAB Engine API CI/CD Pipeline Documentation

## Overview

This document describes the comprehensive CI/CD pipeline implemented for the MATLAB Engine API Python integration project (Issue #1). The pipeline provides automated testing, security scanning, containerization, deployment, and monitoring for rapid development cycles.

## Architecture

### Pipeline Stages

1. **Security & Quality Gates**
   - Code security scanning (Bandit, Safety, Semgrep)
   - Code quality checks (Black, isort, Flake8, MyPy, Pylint)
   - Dependency vulnerability scanning

2. **Test Matrix**
   - Multi-version testing (Python 3.9-3.11, MATLAB R2023a-R2024a)
   - Unit tests with coverage reporting
   - Integration tests
   - Performance benchmarks

3. **Container Build**
   - Multi-stage Docker builds
   - Security-hardened containers
   - Multi-architecture support (amd64, arm64)

4. **Deployment**
   - Staging environment deployment
   - Production deployment with approval gates
   - Health checks and smoke tests

5. **Monitoring & Observability**
   - Prometheus metrics collection
   - Grafana dashboards
   - Alerting rules configuration

## Files Structure

### GitHub Actions Workflows

```
.github/workflows/
├── matlab-python-integration.yml    # Main production pipeline
├── matlab-engine-api-tests.yml     # Existing comprehensive tests
└── matlab-ci.yml                   # MATLAB-specific CI
```

### Docker Configuration

```
docker/
├── Dockerfile                      # Multi-stage production container
├── docker-compose.yml             # Full stack deployment
├── .env.example                   # Environment configuration template
├── entrypoint.sh                  # Container initialization script
├── healthcheck.py                 # Health check script
├── prometheus.yml                 # Metrics configuration
└── grafana/
    ├── dashboards/               # Pre-built dashboards
    └── datasources/             # Data source configurations
```

### Kubernetes Manifests

```
k8s/
├── base/
│   ├── deployment.yaml           # Core Kubernetes resources
│   ├── ingress.yaml             # Traffic routing and SSL
│   └── monitoring.yaml          # Monitoring configuration
├── staging/
│   └── kustomization.yaml       # Staging-specific overrides
└── production/
    ├── kustomization.yaml       # Production configuration
    └── production-config.yaml   # Production-specific settings
```

### Deployment Scripts

```
deployment/
└── deploy.sh                    # Automated deployment script
```

## Pipeline Features

### Security First
- **SAST/DAST scanning** with multiple tools
- **Dependency vulnerability scanning** 
- **Container security** with non-root users and minimal attack surface
- **Network policies** in Kubernetes
- **Secret management** with proper encryption

### Performance Optimized
- **Parallel test execution** across matrix
- **Docker layer caching** for fast builds
- **Resource optimization** with proper limits and requests
- **Auto-scaling** configuration

### Reliability Built-in  
- **Health checks** at multiple levels
- **Rollback mechanisms** for failed deployments
- **Circuit breakers** and retry logic
- **Zero-downtime deployments** with rolling updates

### Observability
- **Comprehensive metrics** collection
- **Structured logging** with proper log levels  
- **Distributed tracing** support
- **Custom dashboards** for key metrics

## Usage

### Quick Start

1. **Environment Setup**
   ```bash
   # Copy environment template
   cp docker/.env.example docker/.env.staging
   cp docker/.env.example docker/.env.production
   
   # Edit with your configuration
   vim docker/.env.staging
   ```

2. **Local Development**
   ```bash
   # Start development stack
   docker-compose -f docker/docker-compose.yml up -d
   
   # Run tests
   docker exec matlab-engine-api python -m pytest tests/ -v
   ```

3. **Staging Deployment**
   ```bash
   ./deployment/deploy.sh --environment staging --version latest
   ```

4. **Production Deployment**
   ```bash
   ./deployment/deploy.sh --environment production --version v1.0.0
   ```

### GitHub Actions Triggers

- **Push to main/develop**: Full pipeline execution
- **Pull requests**: Security and test validation
- **Manual trigger**: Custom environment deployment
- **Scheduled**: Nightly regression tests
- **Release**: Production deployment and tagging

### Deployment Environments

#### Staging
- **Purpose**: Integration testing and validation
- **Resources**: 2 CPU, 4GB RAM per pod
- **Replicas**: 3 pods
- **Features**: All monitoring enabled, cache enabled

#### Production
- **Purpose**: Live application serving
- **Resources**: 4 CPU, 8GB RAM per pod  
- **Replicas**: 5 pods with auto-scaling
- **Features**: Full stack with monitoring, caching, and database

## Monitoring & Alerts

### Key Metrics
- **Response time** (95th percentile < 2s)
- **Error rate** (< 0.1% for 5xx errors)
- **Active MATLAB sessions**
- **Resource utilization** (CPU, memory, disk)
- **Request throughput**

### Alert Rules
- High response time (> 2s for 5 minutes)
- High error rate (> 0.1 req/s for 2 minutes)
- Pod down (> 1 minute)
- High memory usage (> 90% for 5 minutes)
- Too many active sessions (> 50 for 5 minutes)

### Dashboards
- **API Performance**: Response times, throughput, errors
- **System Resources**: CPU, memory, disk usage
- **MATLAB Engine**: Active sessions, computation times
- **Infrastructure**: Kubernetes cluster health

## Security Considerations

### Container Security
- **Non-root user** (UID 1000)
- **Read-only root filesystem** where possible
- **Minimal base image** (Alpine/distroless)
- **No unnecessary privileges**
- **Security context** constraints

### Network Security
- **TLS termination** at ingress
- **Network policies** for pod-to-pod communication
- **Rate limiting** to prevent abuse
- **CORS configuration** for web access

### Secrets Management
- **Kubernetes secrets** for sensitive data
- **Environment-specific** configuration
- **Encrypted at rest** storage
- **Rotation policies** for credentials

## Troubleshooting

### Common Issues

1. **MATLAB Engine Startup Failures**
   ```bash
   # Check logs
   docker logs matlab-engine-api
   
   # Verify display configuration
   docker exec matlab-engine-api echo $DISPLAY
   ```

2. **Memory Issues**
   ```bash
   # Check resource usage
   kubectl top pods -l app=matlab-engine-api
   
   # Increase memory limits if needed
   kubectl patch deployment matlab-engine-api -p '...'
   ```

3. **Health Check Failures**
   ```bash
   # Run health check manually
   docker exec matlab-engine-api python /app/healthcheck.py --verbose
   
   # Check specific components
   docker exec matlab-engine-api python /app/healthcheck.py --startup-check
   ```

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
export MATLAB_ENGINE_LOG_LEVEL=DEBUG

# Restart services
docker-compose restart matlab-engine-api
```

## Performance Tuning

### Resource Optimization
- **CPU**: Start with 1 CPU, scale based on load
- **Memory**: MATLAB requires 2-4GB minimum
- **Storage**: Use SSD for better I/O performance

### MATLAB Optimization
- **Headless mode**: `-nojvm -nodisplay` for server deployment
- **Parallel pools**: Configure based on CPU cores
- **Memory management**: Regular cleanup of large variables

### Container Optimization
- **Multi-stage builds** to reduce image size
- **Layer caching** for faster builds
- **Resource requests/limits** to prevent resource contention

## Maintenance

### Regular Tasks
- **Update dependencies** monthly
- **Security patches** as needed
- **Performance review** quarterly
- **Capacity planning** based on usage

### Backup Strategy
- **Configuration backups** stored in Git
- **Data backups** for persistent volumes
- **Container registry** for image versions
- **Database backups** if using persistent storage

## Support and Contacts

- **Repository**: https://github.com/murr2k/matlab-app-dev
- **Issues**: GitHub Issues for bug reports
- **Monitoring**: Grafana dashboards for system health
- **Logs**: Centralized logging via Docker/Kubernetes

## Version History

- **v1.0.0**: Initial production pipeline
- **Future**: Planned enhancements for multi-cloud support