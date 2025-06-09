#!/bin/bash

# ==============================================================================
# MARKET INTELLIGENCE PLATFORM - LOCAL DEPLOYMENT GUIDE
# Step-by-step deployment for local development environment
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}MIP LOCAL DEPLOYMENT GUIDE${NC}"
echo -e "${BLUE}============================================${NC}"

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found."
    exit 1
fi

print_success "Prerequisites check passed"

# Setup environment
print_step "Setting up environment..."

if [[ ! -f ".env" ]]; then
    cp .env.example .env
    print_warning "Created .env file. Please update with your API keys."
fi

# Start deployment
print_step "Starting database services..."
docker-compose up -d postgres redis

print_step "Waiting for databases..."
sleep 30

# Apply database schema
print_step "Applying database schema..."
if [[ -f "database/schema.sql" ]]; then
    docker exec -i $(docker-compose ps -q postgres) psql -U mip_user -d market_intelligence < database/schema.sql
fi

print_step "Starting all services..."
docker-compose up -d

print_step "Waiting for services to start..."
sleep 60

# Health checks
print_step "Performing health checks..."

services=("8000:API Gateway" "8001:Agent Orchestration" "8002:Sentiment Analysis")

for service in "${services[@]}"; do
    port=$(echo $service | cut -d: -f1)
    name=$(echo $service | cut -d: -f2)
    
    if curl -f -s "http://localhost:$port/health" >/dev/null 2>&1; then
        print_success "$name is healthy"
    else
        print_warning "$name not responding"
    fi
done

print_success "Local deployment completed!"

echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo "  API Gateway:  http://localhost:8000"
echo "  Frontend:     http://localhost:3001" 
echo "  Grafana:      http://localhost:3000 (admin/admin123)"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Update API keys in .env file"
echo "  2. Run tests: python3 scripts/comprehensive_tests.py --env local --test-type sanity"
echo "  3. Open frontend at http://localhost:3001"
