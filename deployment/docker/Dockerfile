FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip3 install -e .

# Expose port for distributed training
EXPOSE 29500

# Default command
CMD ["/bin/bash"]
