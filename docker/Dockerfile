# Multi-stage Docker build for MATLAB Engine API Python Integration
# Optimized for production deployment with minimal attack surface

# Build Arguments
ARG PYTHON_VERSION=3.11
ARG MATLAB_VERSION=R2023b
ARG UBUNTU_VERSION=22.04

# Stage 1: Base image with MATLAB installation
FROM ubuntu:${UBUNTU_VERSION} as matlab-base

# Install system dependencies and MATLAB
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    ca-certificates \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Download and install MATLAB (placeholder - actual installation requires license)
# In production, this would mount a pre-built MATLAB image or use MATLAB containers
RUN echo "MATLAB installation placeholder" && \
    mkdir -p /opt/matlab

# Stage 2: Python environment with dependencies
FROM ubuntu:${UBUNTU_VERSION} as python-deps

ARG PYTHON_VERSION

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY src/python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Production image
FROM ubuntu:${UBUNTU_VERSION} as production

ARG PYTHON_VERSION
ARG MATLAB_VERSION

# Labels for metadata
LABEL maintainer="murr2k@gmail.com"
LABEL org.opencontainers.image.title="MATLAB Engine API for Python"
LABEL org.opencontainers.image.description="Production-ready MATLAB Engine API Python integration"
LABEL org.opencontainers.image.source="https://github.com/murr2k/matlab-app-dev"
LABEL org.opencontainers.image.version="${MATLAB_VERSION}-python${PYTHON_VERSION}"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    xvfb \
    dumb-init \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r matlabapp && useradd -r -g matlabapp -m -s /bin/bash matlabuser

# Copy Python environment from deps stage
COPY --from=python-deps /usr/local/lib/python${PYTHON_VERSION} /usr/local/lib/python${PYTHON_VERSION}
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy MATLAB from base stage (in production, this would be actual MATLAB)
COPY --from=matlab-base /opt/matlab /opt/matlab

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=matlabuser:matlabapp src/python/ ./src/python/
COPY --chown=matlabuser:matlabapp docker/entrypoint.sh ./entrypoint.sh
COPY --chown=matlabuser:matlabapp docker/healthcheck.py ./healthcheck.py

# Make scripts executable
RUN chmod +x ./entrypoint.sh && \
    chmod +x ./healthcheck.py

# Set environment variables
ENV PYTHONPATH=/app/src/python:$PYTHONPATH \
    MATLAB_ENGINE_ENV=production \
    MATLAB_ROOT=/opt/matlab \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R matlabuser:matlabapp /app

# Switch to non-root user
USER matlabuser

# Expose application port (if running web service)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Default command
CMD ["./entrypoint.sh"]