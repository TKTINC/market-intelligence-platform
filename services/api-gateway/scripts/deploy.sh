#!/bin/bash

# Enhanced FastAPI Gateway Deployment Script

set -e

echo "🚀 Starting MIP Enhanced FastAPI Gateway Deployment..."

# Environment setup
export COMPOSE_PROJECT_NAME=mip
export DOCKER_BUILDKIT=1

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs nginx/ssl nginx/logs monitoring/grafana/{dashboards,datasources} logging

# Environment variables
echo "🔧 Setting up environment..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Database
POSTGRES_DB=mip_database
POSTGRES_USER=mip_user
POSTGRES_PASSWORD=mip_secure_password

# Redis
REDIS_PASSWORD=redis_secure_password

# JWT
JWT_SECRET_KEY=$(openssl rand -base64 32)

# External APIs
POLYGON_API_KEY=your_polygon_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
OPENAI_API_KEY=your_openai_api_key

# Monitoring
GRAFANA_PASSWORD=grafana_admin_password

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
EOF
    echo "✅ .env file created. Please update with your actual API keys."
fi

# Pull images
echo "📥 Pulling Docker images..."
docker-compose pull

# Build services
echo "🔨 Building services..."
docker-compose build --parallel

# Start infrastructure services first
echo "🏗️ Starting infrastructure services..."
docker-compose up -d redis postgres kafka zookeeper elasticsearch

# Wait for infrastructure
echo "⏳ Waiting for infrastructure services..."
sleep 30

# Start AI services
echo "🤖 Starting AI services..."
docker-compose up -d finbert-sentiment-service tft-forecasting-service gpt4-strategy-service llama-explanation-service

# Wait for AI services
echo "⏳ Waiting for AI services..."
sleep 20

# Start processing services
echo "⚡ Starting processing services..."
docker-compose up -d realtime-processing-service

# Start API Gateway
echo "🌐 Starting API Gateway..."
docker-compose up -d api-gateway

# Start monitoring
echo "📊 Starting monitoring services..."
docker-compose up -d prometheus grafana kibana logstash

# Start frontend and load balancer
echo "🎨 Starting frontend services..."
docker-compose up -d react-dashboard nginx

# Health checks
echo "🔍 Running health checks..."
sleep 10

services=("api-gateway" "redis" "postgres" "kafka")
for service in "${services[@]}"; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        docker-compose logs $service
    fi
done

# Display service URLs
echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Service URLs:"
echo "📱 Dashboard:     http://localhost"
echo "🔌 API Gateway:   http://localhost/api"
echo "📊 Grafana:       http://localhost:3000"
echo "🔍 Kibana:        http://localhost:5601"
echo "📈 Prometheus:    http://localhost:9090"
echo ""
echo "API Documentation: http://localhost/api/docs"
echo "WebSocket Test:    ws://localhost/ws/user/test"
echo ""
echo "🔧 To view logs: docker-compose logs -f [service-name]"
echo "🛑 To stop: docker-compose down"
echo "🔄 To restart: docker-compose restart [service-name]"
