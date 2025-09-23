.PHONY: help install test train api docker-build docker-up docker-down clean

# Default target
help:
	@echo "Fraud Detection Graph System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install all dependencies"
	@echo "  make setup        - Set up environment and directories"
	@echo ""
	@echo "Data & Training:"
	@echo "  make generate-data - Generate synthetic training data"
	@echo "  make train        - Train all GNN models"
	@echo ""
	@echo "Running the System:"
	@echo "  make api          - Run the API server locally"
	@echo "  make test         - Run all tests"
	@echo "  make test-api     - Run API tests with curl"
	@echo "  make test-scenarios - Run fraud scenario tests"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start all services with Docker Compose"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make docker-logs  - View Docker logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean generated files and caches"
	@echo "  make format       - Format Python code"
	@echo "  make lint         - Run linting checks"

# Installation
install:
	pip install -r requirements.txt

setup:
	mkdir -p models data logs visualizations
	cp .env.example .env
	@echo "Setup complete. Please edit .env with your configuration."

# Data generation
generate-data:
	@echo "Generating synthetic fraud detection dataset..."
	python -c "from src.utils.data_generator import TransactionDataGenerator; \
		g = TransactionDataGenerator(); \
		df = g.generate_dataset(50000); \
		g.save_dataset(df, 'data/training_data.csv')"
	@echo "Dataset generated: data/training_data.csv"

# Training
train:
	@echo "Training GNN models..."
	python train.py --epochs 100 --lr 0.001

train-quick:
	@echo "Quick training (fewer epochs)..."
	python train.py --epochs 20 --lr 0.001

# API Server
api:
	@echo "Starting Fraud Detection API..."
	cd src/api && python fraud_api.py

api-dev:
	@echo "Starting API in development mode with auto-reload..."
	cd src/api && uvicorn fraud_api:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	@echo "Running all tests..."
	pytest tests/ -v

test-api:
	@echo "Running API tests..."
	chmod +x tests/test_api.sh
	./tests/test_api.sh

test-scenarios:
	@echo "Running fraud scenario tests..."
	python tests/test_scenarios.py

# Docker
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started. API available at http://localhost:8000"
	@echo "Grafana available at http://localhost:3000 (admin/admin)"
	@echo "Neo4j browser available at http://localhost:7474"

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-logs:
	docker-compose logs -f fraud-api

docker-clean:
	@echo "Removing all containers and volumes..."
	docker-compose down -v

# Development utilities
format:
	@echo "Formatting Python code..."
	black src/ tests/ train.py
	isort src/ tests/ train.py

lint:
	@echo "Running linting checks..."
	flake8 src/ tests/ --max-line-length=100
	pylint src/

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API Docs: http://localhost:8000/docs"

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

clean-data:
	@echo "Cleaning generated data..."
	rm -rf data/*.csv
	rm -rf visualizations/*.html
	rm -rf logs/*.log

clean-models:
	@echo "Cleaning trained models..."
	rm -rf models/*.pt
	rm -rf models/*.json

# Full workflow
demo: setup generate-data train api

# Production deployment
deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployment/k8s/

deploy-aws:
	@echo "Deploying to AWS..."
	@echo "Please run: ./deployment/aws/deploy.sh"

# Benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	python benchmarks/performance_test.py

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html

# Database operations
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec postgres psql -U postgres -d fraud_detection -f /docker-entrypoint-initdb.d/init.sql

db-backup:
	@echo "Backing up database..."
	docker-compose exec postgres pg_dump -U postgres fraud_detection > backups/fraud_detection_$(date +%Y%m%d_%H%M%S).sql

# Development shortcuts
dev: docker-up
	@echo "Development environment is ready!"
	@echo "API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000"
	@echo "Run 'make test-api' to test the API"

stop: docker-down

restart: docker-down docker-up

logs: docker-logs