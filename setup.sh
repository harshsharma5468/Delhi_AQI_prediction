#!/bin/bash

# Delhi AQI Predictor - Setup Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up Delhi AQI Predictor${NC}"
echo "======================================"

# Function to print status
print_status() {
    echo -e "${YELLOW}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# Check if data file exists
check_data() {
    print_status "Checking dataset..."
    if [ ! -f "data/delhi_aqi.csv" ]; then
        print_error "Dataset not found at data/delhi_aqi.csv"
        echo "Please ensure the dataset is in the correct location."
        exit 1
    fi
    print_success "Dataset found"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data models logs visualizations monitoring/grafana/dashboards monitoring/grafana/datasources
    print_success "Directories created"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_success "Docker image built"
}

# Train the model
train_model() {
    print_status "Training machine learning model..."
    
    # Check if model already exists
    if [ -f "models/trained_model.pkl" ] && [ -f "models/scaler.pkl" ]; then
        print_success "Model already trained. Skipping training."
        return
    fi
    
    # Run training in Docker
    docker-compose run --rm aqi-predictor python -c "
from src.model_training import train_complete_pipeline
train_complete_pipeline()
"
    
    if [ $? -eq 0 ]; then
        print_success "Model trained successfully"
    else
        print_error "Model training failed"
        exit 1
    fi
}

# Start the application
start_application() {
    print_status "Starting Delhi AQI Predictor..."
    docker-compose up -d
    
    # Wait for application to be ready
    print_status "Waiting for application to start..."
    sleep 10
    
    # Check if application is running
    if curl -s http://localhost:8501/_stcore/health > /dev/null; then
        print_success "Application is running!"
        echo ""
        echo -e "${GREEN}✅ Setup complete!${NC}"
        echo ""
        echo "🌐 Access the application at: http://localhost:8501"
        echo ""
        echo "📋 Useful commands:"
        echo "   docker-compose up -d          # Start application"
        echo "   docker-compose down           # Stop application"
        echo "   docker-compose logs -f        # View logs"
        echo "   docker-compose exec aqi-predictor bash  # Access container shell"
        echo ""
    else
        print_error "Application failed to start"
        docker-compose logs aqi-predictor
        exit 1
    fi
}

# Main setup process
main() {
    echo ""
    print_status "Starting setup process..."
    
    # Run all steps
    check_docker
    check_data
    create_directories
    build_image
    train_model
    start_application
    
    echo ""
    print_success "Delhi AQI Predictor is ready to use!"
}

# Run main function
main "$@"