# VTK Visual Testing Docker Image
# 
# Provides secure, isolated environment for executing VTK code
# with offscreen rendering capability for visual regression testing

FROM python:3.13-slim

# Install system dependencies for VTK offscreen rendering
# VTK will try X (not available), EGL (not available), then fall back to OSMesa
RUN apt-get update \
    && apt-get install -y \
    libxrender1 \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 vtkuser && \
    mkdir -p /workspace /output /data && \
    chown -R vtkuser:vtkuser /workspace /output /data

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    vtk>=9.5.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0

# Switch to non-root user
USER vtkuser

# Declare volume mount points (for documentation)
VOLUME ["/workspace", "/output", "/data"]

# Default command - will be overridden at runtime
CMD ["python3", "--version"]
