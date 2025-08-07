#!/bin/bash
# MATLAB Engine API Deployment Script
# Supports staging and production deployments with rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/matlab-engine-deploy-${TIMESTAMP}.log"

# Default values
ENVIRONMENT="staging"
VERSION="latest"
SKIP_TESTS=false
FORCE_DEPLOY=false
ROLLBACK_VERSION=""
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $*${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $*${NC}" | tee -a "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
MATLAB Engine API Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Deployment environment (staging|production) [default: staging]
    -v, --version VERSION    Version to deploy [default: latest]
    -s, --skip-tests         Skip running tests before deployment
    -f, --force              Force deployment even if tests fail
    -r, --rollback VERSION   Rollback to specified version
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

EXAMPLES:
    $0 --environment staging --version v1.2.3
    $0 --environment production --skip-tests
    $0 --rollback v1.2.2
    $0 --dry-run --environment production

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log "Validating deployment environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        staging|production)
            log "Environment validated: $ENVIRONMENT"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker docker-compose git; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check project structure
    local required_files=(
        "docker/Dockerfile"
        "docker/docker-compose.yml"
        "src/python/requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_ROOT/$file" ]; then
            error "Required file not found: $file"
            exit 1
        fi
    done
    
    success "Prerequisites check completed"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        warn "Skipping tests as requested"
        return 0
    fi
    
    log "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would run: pytest src/python/tests/ -v --tb=short"
        return 0
    fi
    
    # Run tests with timeout
    if timeout 600 python -m pytest src/python/tests/ -v --tb=short; then
        success "All tests passed"
    else
        error "Tests failed"
        if [ "$FORCE_DEPLOY" = true ]; then
            warn "Continuing deployment despite test failures (--force specified)"
        else
            error "Deployment aborted due to test failures. Use --force to override"
            exit 1
        fi
    fi
}

# Build Docker image
build_image() {
    log "Building Docker image for version: $VERSION"
    
    cd "$PROJECT_ROOT"
    
    local image_tag="matlab-engine-api:$VERSION"
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would build: $image_tag"
        return 0
    fi
    
    # Build with build args based on environment
    local build_args=""
    if [ "$ENVIRONMENT" = "production" ]; then
        build_args="--build-arg PYTHON_VERSION=3.11 --build-arg MATLAB_VERSION=R2023b"
    fi
    
    if docker build $build_args -f docker/Dockerfile -t "$image_tag" .; then
        success "Docker image built successfully: $image_tag"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy application
deploy_application() {
    log "Deploying to $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export ENVIRONMENT
    export VERSION
    
    # Create environment file if it doesn't exist
    local env_file="docker/.env.${ENVIRONMENT}"
    if [ ! -f "$env_file" ]; then
        warn "Environment file not found: $env_file"
        log "Creating from template..."
        cp "docker/.env.example" "$env_file"
        warn "Please review and update $env_file with appropriate values"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would deploy with docker-compose using $env_file"
        return 0
    fi
    
    # Deploy using docker-compose
    local compose_file="docker/docker-compose.yml"
    local profiles=""
    
    # Set profiles based on environment
    case $ENVIRONMENT in
        staging)
            profiles="--profile with-cache"
            ;;
        production)
            profiles="--profile full"
            ;;
    esac
    
    if docker-compose -f "$compose_file" --env-file "$env_file" $profiles up -d; then
        success "Application deployed successfully"
    else
        error "Deployment failed"
        exit 1
    fi
}

# Health check
health_check() {
    log "Performing health check..."
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would perform health check"
        return 0
    fi
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts..."
        
        if docker exec matlab-engine-api python /app/healthcheck.py; then
            success "Health check passed"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Rollback function
rollback() {
    local rollback_version="$1"
    
    log "Rolling back to version: $rollback_version"
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would rollback to: $rollback_version"
        return 0
    fi
    
    # Stop current deployment
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.yml down
    
    # Deploy rollback version
    export VERSION="$rollback_version"
    docker-compose -f docker/docker-compose.yml up -d
    
    # Health check
    if health_check; then
        success "Rollback completed successfully"
    else
        error "Rollback failed health check"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would perform cleanup"
        return 0
    fi
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove old log files (keep last 10)
    find /tmp -name "matlab-engine-deploy-*.log" -type f | sort -r | tail -n +11 | xargs rm -f
    
    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting MATLAB Engine API deployment"
    log "Log file: $LOG_FILE"
    
    parse_args "$@"
    
    # Handle rollback
    if [ -n "$ROLLBACK_VERSION" ]; then
        rollback "$ROLLBACK_VERSION"
        exit 0
    fi
    
    validate_environment
    check_prerequisites
    
    # Skip tests and build for rollback
    if [ -z "$ROLLBACK_VERSION" ]; then
        run_tests
        build_image
    fi
    
    deploy_application
    
    if health_check; then
        success "Deployment completed successfully!"
        log "Environment: $ENVIRONMENT"
        log "Version: $VERSION"
        log "Log file: $LOG_FILE"
    else
        error "Deployment failed health check"
        warn "Consider rolling back to previous version"
        exit 1
    fi
    
    cleanup
}

# Trap for cleanup on exit
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"