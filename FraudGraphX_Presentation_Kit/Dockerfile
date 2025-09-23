FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY demo.py .

# Create necessary directories
RUN mkdir -p data models demo_output logs

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd -m -u 1000 fraudgraphx && chown -R fraudgraphx:fraudgraphx /app
USER fraudgraphx

# Expose ports
EXPOSE 8000 8765 8001 8888

# Default command
CMD ["uvicorn", "src.serve.advanced_api:app", "--host", "0.0.0.0", "--port", "8000"]