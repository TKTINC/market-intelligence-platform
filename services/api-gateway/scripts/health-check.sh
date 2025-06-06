#!/bin/bash

# Health Check Script for MIP Services

echo "🏥 MIP Health Check Report"
echo "========================="

services=(
    "api-gateway:8000:/health"
    "finbert-sentiment-service:8005:/health" 
    "tft-forecasting-service:8008:/health"
    "gpt4-strategy-service:8007:/health"
    "llama-explanation-service:8006:/health"
    "realtime-processing-service:8009:/health"
)

for service_info in "${services[@]}"; do
    IFS=':' read -r service port path <<< "$service_info"
    
    echo -n "Checking $service... "
    
    if curl -s -f "http://localhost:$port$path" > /dev/null; then
        echo "✅ Healthy"
    else
        echo "❌ Unhealthy"
        
        # Check if container is running
        if docker-compose ps $service | grep -q "Up"; then
            echo "   Container is running but service is not responding"
            echo "   Logs:"
            docker-compose logs --tail=5 $service | sed 's/^/   /'
        else
            echo "   Container is not running"
        fi
    fi
done

echo ""
echo "Database Connections:"
echo -n "Redis... "
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Connection failed"
fi

echo -n "PostgreSQL... "
if docker-compose exec postgres pg_isready > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Connection failed"
fi

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker-compose ps -q)
