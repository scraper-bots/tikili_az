# DeepSeek AZE Docker Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for evaluation
RUN pip install --no-cache-dir \
    evaluate \
    rouge-score \
    sacrebleu \
    bert-score

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/synthetic \
             checkpoints outputs logs model_cache

# Set permissions
RUN chmod +x train.py evaluate.py

# Expose ports for API and monitoring
EXPOSE 8000 6006

# Default command
CMD ["python", "train.py", "--help"]