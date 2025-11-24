FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for Python packages
RUN python3 -m venv /opt/venv

# Ensure the venv's Python and pip are first in PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install TensorFlow + Sphinx once
RUN pip install --upgrade pip && \
    pip install tensorflow sphinx sphinx-rtd-theme

# Set working directory
WORKDIR /workspace
