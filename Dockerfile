# ACE-Step Music Generator - RunPod Deployment
FROM nvidia/cuda:12.6-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements.txt .
COPY runpod_setup.py .
COPY simple_ace_api.py .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir -r requirements.txt

# Clone ACE-Step repository
RUN git clone https://github.com/ace-step/ACE-Step.git && \
    cd ACE-Step && \
    pip3 install --no-cache-dir -e .

# Create directories for models and outputs
RUN mkdir -p /app/checkpoints /app/generated_music /app/logs

# Copy the API files to the ACE-Step directory
RUN cp /app/simple_ace_api.py /app/ACE-Step/ && \
    cp /app/runpod_setup.py /app/ACE-Step/

# Set working directory to ACE-Step
WORKDIR /app/ACE-Step

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command - will be overridden by RunPod
CMD ["python3", "runpod_setup.py"]
