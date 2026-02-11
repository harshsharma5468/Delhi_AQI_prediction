#!/bin/bash

# Delhi AQI Predictor - Deployment Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_NAME="delhi-aqi-predictor"
TAG=${1:-"latest"}
REGISTRY=${2:-"your-registry"}  # Change to your Docker registry

echo -e "${GREEN}🚀 Deploying Delhi AQI Predictor${NC}"
echo "========================================"

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not found"
        exit 1
    fi
    
    print_success "Prerequisites checked"
}

# Build production image
build_production_image() {
    print_status "Building production Docker image..."
    
    # Build with production Dockerfile
    docker build -f Dockerfile.prod -t ${REGISTRY}/${APP_NAME}:${TAG} .
    
    if [ $? -eq 0 ]; then
        print_success "Production image built"
    else
        print_error "Failed to build production image"
        exit 1
    fi
}

# Push to registry (optional)
push_to_registry() {
    if [ "$REGISTRY" != "your-registry" ]; then
        print_status "Pushing image to registry..."
        docker push ${REGISTRY}/${APP_NAME}:${TAG}
        print_success "Image pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down
    
    # Pull latest image if using registry
    if [ "$REGISTRY" != "your-registry" ]; then
        docker-compose -f docker-compose.prod.yml pull
    fi
    
    # Start containers
    docker-compose -f docker-compose.prod.yml up -d
    
    print_success "Application deployed"
}

# Run health check
health_check() {
    print_status "Running health check..."
    
    # Wait for application to start
    sleep 15
    
    # Check application health
    if curl -s http://localhost:8501/_stcore/health > /dev/null; then
        print_success "Application is healthy"
        echo ""
        echo -e "${GREEN}✅ Deployment successful!${NC}"
        echo ""
        echo "🌐 Application URL: http://localhost:8501"
        echo "📊 Monitoring: http://localhost:9090 (Prometheus)"
        echo "📈 Dashboard: http://localhost:3000 (Grafana)"
        echo ""
    else
        print_error "Application health check failed"
        docker-compose -f docker-compose.prod.yml logs aqi-predictor
        exit 1
    fi
}

# Show deployment info
show_info() {
    echo ""
    echo -e "${GREEN}📋 Deployment Information${NC}"
    echo "================================"
    echo "Application: ${APP_NAME}"
    echo "Tag: ${TAG}"
    echo "Registry: ${REGISTRY}"
    echo ""
    echo "📦 Container Status:"
    docker-compose -f docker-compose.prod.yml ps
    echo ""
    echo "📊 Resource Usage:"
    docker stats --no-stream ${APP_NAME} | tail -n +2
}

# Main deployment process
main() {
    echo ""
    print_status "Starting deployment process..."
    
    # Run deployment steps
    check_prerequisites
    build_production_image
    push_to_registry
    deploy_compose
    health_check
    show_info
    
    echo ""
    print_success "Deployment completed successfully!"
    echo ""
}

# Run main function
main "$@"