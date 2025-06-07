#!/bin/bash

echo "🚀 Starting Enhanced Monitoring Stack..."

# Navigate to monitoring directory
cd infrastructure/monitoring

# Load environment variables if they exist
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=("prometheus:9090" "grafana:3000" "alertmanager:9093")
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s http://localhost:$port > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done

echo ""
echo "🎉 Monitoring stack started successfully!"
echo ""
echo "📊 Access URLs:"
echo "   Grafana:      http://localhost:3000 (admin/admin123)"
echo "   Prometheus:   http://localhost:9090"
echo "   AlertManager: http://localhost:9093"
