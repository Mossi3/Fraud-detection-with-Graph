# FraudGraphX Makefile

.PHONY: help install demo test api monitor docker clean

# Default target
help:
	@echo "FraudGraphX - Advanced Graph-Based Fraud Detection"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  demo        - Run complete demo"
	@echo "  test        - Run tests"
	@echo "  api         - Start API server"
	@echo "  monitor     - Start real-time monitor"
	@echo "  docker      - Build and run with Docker"
	@echo "  clean       - Clean up generated files"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run complete demo
demo:
	@echo "Running FraudGraphX demo..."
	python demo.py --transactions 10000 --fraud-rate 0.05

# Run demo with API
demo-api:
	@echo "Running FraudGraphX demo with API..."
	python demo.py --transactions 10000 --fraud-rate 0.05 --run-api

# Generate sample data
data:
	@echo "Generating sample fraud data..."
	python -m src.data.generate_fraud_data \
		--n_transactions 50000 \
		--fraud_rate 0.05 \
		--n_fraud_rings 10 \
		--output data/raw/fraud_transactions.csv

# Train GNN model
train:
	@echo "Training GNN model..."
	python -m src.models.gnn_models \
		--data data/raw/fraud_transactions.csv \
		--model graphsage \
		--hidden_dim 128 \
		--epochs 100

# Detect fraud rings
rings:
	@echo "Detecting fraud rings..."
	python -m src.graph.community_detection \
		--data data/raw/fraud_transactions.csv \
		--method ensemble \
		--output detected_rings.csv

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Run API tests
test-api:
	@echo "Running API tests..."
	./tests/curl_tests.sh

# Start API server
api:
	@echo "Starting API server..."
	uvicorn src.serve.advanced_api:app --reload --host 0.0.0.0 --port 8000

# Start real-time monitor
monitor:
	@echo "Starting real-time monitor..."
	python -m src.features.real_time_monitor

# Start anomaly detection
anomaly:
	@echo "Running anomaly detection..."
	python -m src.features.anomaly_detector \
		--data data/raw/fraud_transactions.csv \
		--output models/anomaly

# Create visualizations
viz:
	@echo "Creating visualizations..."
	python -m src.visual.fraud_ring_viz \
		--data data/raw/fraud_transactions.csv \
		--mode static \
		--output fraud_rings_report.html

# Docker operations
docker-build:
	@echo "Building Docker image..."
	docker build -t fraudgraphx:latest .

docker-run:
	@echo "Running with Docker Compose..."
	docker-compose up -d

docker-stop:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f

# Development tools
lint:
	@echo "Running linters..."
	flake8 src/ --max-line-length=100
	pylint src/ --disable=C0103,R0903,W0613

format:
	@echo "Formatting code..."
	black src/ --line-length=100
	isort src/

# Jupyter notebook
jupyter:
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Performance benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	python -m tests.benchmark_performance

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf demo_output/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.log
	rm -rf models/*.pt
	rm -rf models/*.joblib

# Create documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/

# Setup development environment
dev-setup:
	@echo "Setting up development environment..."
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements-dev.txt

# Health check
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "API not running"
	redis-cli ping || echo "Redis not running"

# Monitor system
monitor-system:
	@echo "System monitoring dashboard available at:"
	@echo "  - API Health: http://localhost:8000/health"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Jupyter: http://localhost:8888"

# All-in-one setup
setup-all: install data demo docker-run
	@echo "Complete FraudGraphX setup finished!"
	@echo "Services available:"
	@echo "  - API: http://localhost:8000"
	@echo "  - WebSocket: ws://localhost:8765"
	@echo "  - Monitoring: http://localhost:3000"
	@echo "  - Jupyter: http://localhost:8888"