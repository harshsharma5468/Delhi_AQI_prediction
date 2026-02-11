.PHONY: help setup build train run stop clean deploy test logs

# Variables
APP_NAME=delhi-aqi-predictor
DOCKER_COMPOSE=docker-compose
DOCKER_COMPOSE_PROD=docker-compose -f docker-compose.prod.yml

# Default target
help:
	@echo "Delhi AQI Predictor - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  setup     - Setup the project (install dependencies, train model)"
	@echo "  build     - Build Docker image"
	@echo "  train     - Train the machine learning model"
	@echo "  run       - Start the application in development mode"
	@echo "  run-prod  - Start the application in production mode"
	@echo "  stop      - Stop the application"
	@echo "  clean     - Clean up Docker resources"
	@echo "  deploy    - Deploy to production"
	@echo "  test      - Run tests"
	@echo "  logs      - View application logs"
	@echo "  shell     - Access container shell"
	@echo "  backup    - Backup data and models"
	@echo "  monitor   - Start monitoring stack"

# Setup the project
setup: build train
	@echo "✅ Setup complete! Run 'make run' to start the application."

# Build Docker image
build:
	@echo "🔨 Building Docker image..."
	@$(DOCKER_COMPOSE) build
	@echo "✅ Docker image built successfully."

# Train the model
train:
	@echo "🤖 Training machine learning model..."
	@if [ -f "models/trained_model.pkl" ] && [ -f "models/scaler.pkl" ]; then \
		echo "📦 Model already exists. Skipping training."; \
	else \
		$(DOCKER_COMPOSE) run --rm aqi-predictor python -c "from src.model_training import train_complete_pipeline; train_complete_pipeline()"; \
		echo "✅ Model trained successfully."; \
	fi

# Run in development mode
run: build
	@echo "🚀 Starting Delhi AQI Predictor..."
	@$(DOCKER_COMPOSE) up -d
	@echo "✅ Application started at http://localhost:8501"

# Run in production mode
run-prod:
	@echo "🚀 Starting Delhi AQI Predictor in production mode..."
	@$(DOCKER_COMPOSE_PROD) up -d
	@echo "✅ Production application started at http://localhost:8501"
	@echo "📊 Monitoring at http://localhost:9090"
	@echo "📈 Grafana at http://localhost:3000"

# Stop the application
stop:
	@echo "🛑 Stopping application..."
	@$(DOCKER_COMPOSE) down
	@echo "✅ Application stopped."

# Clean up Docker resources
clean:
	@echo "🧹 Cleaning up Docker resources..."
	@$(DOCKER_COMPOSE) down -v --rmi all --remove-orphans
	@docker system prune -f
	@echo "✅ Cleanup complete."

# Deploy to production
deploy:
	@echo "🚀 Deploying to production..."
	@./deploy.sh
	@echo "✅ Deployment complete."

# Run tests
test:
	@echo "🧪 Running tests..."
	@$(DOCKER_COMPOSE) run --rm aqi-predictor python -m pytest tests/ -v
	@echo "✅ Tests completed."

# View logs
logs:
	@$(DOCKER_COMPOSE) logs -f aqi-predictor

# Access container shell
shell:
	@$(DOCKER_COMPOSE) exec aqi-predictor bash

# Backup data and models
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ logs/
	@echo "✅ Backup created in backups/ directory."

# Start monitoring stack
monitor:
	@echo "📊 Starting monitoring stack..."
	@$(DOCKER_COMPOSE_PROD) up -d prometheus grafana
	@echo "✅ Monitoring stack started:"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3000 (admin/admin)"

# Check application health
health:
	@echo "🏥 Checking application health..."
	@if curl -s http://localhost:8501/_stcore/health > /dev/null; then \
		echo "✅ Application is healthy"; \
	else \
		echo "❌ Application health check failed"; \
	fi

# Show container status
status:
	@echo "📋 Container Status:"
	@$(DOCKER_COMPOSE) ps

# Show resource usage
resources:
	@echo "📊 Resource Usage:"
	@docker stats --no-stream $(shell docker ps -q --filter name=aqi-predictor)