#!/bin/bash

# Enhanced Real-time Processing Agent Deployment Script

set -e

# Configuration
NAMESPACE="mip"
SERVICE_NAME="realtime-processing"
IMAGE_TAG=${1:-"latest"}
ENVIRONMENT=${2:-"development"}

echo "🚀 Deploying Enhanced Real-time Processing Agent..."
echo "   Environment: $ENVIRONMENT"
echo "   Image Tag: $IMAGE_TAG"
echo "   Namespace: $NAMESPACE"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Build Docker image
echo "📦 Building Docker image..."
docker build -t mip/${SERVICE_NAME}:${IMAGE_TAG} .

# Tag for registry
if [ "$ENVIRONMENT" = "production" ]; then
    echo "🏷️  Tagging for production registry..."
    docker tag mip/${SERVICE_NAME}:${IMAGE_TAG} your-registry.com/mip/${SERVICE_NAME}:${IMAGE_TAG}
    docker push your-registry.com/mip/${SERVICE_NAME}:${IMAGE_TAG}
fi

# Deploy using Helm
echo "⚙️  Deploying with Helm..."
helm upgrade --install ${SERVICE_NAME} ./helm \
    --namespace $NAMESPACE \
    --set image.tag=${IMAGE_TAG} \
    --set env.ENVIRONMENT=${ENVIRONMENT} \
    --wait \
    --timeout 600s

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/${SERVICE_NAME} -n $NAMESPACE

# Get service status
echo "📊 Service Status:"
kubectl get pods -n $NAMESPACE -l app=${SERVICE_NAME}
kubectl get svc -n $NAMESPACE -l app=${SERVICE_NAME}

# Health check
echo "🔍 Health Check:"
sleep 10
if kubectl get pods -n $NAMESPACE -l app=${SERVICE_NAME} | grep -q "Running"; then
    echo "✅ Deployment successful!"
    
    # Port forward for local testing
    if [ "$ENVIRONMENT" = "development" ]; then
        echo "🔧 Setting up port forwarding for local access..."
        echo "   Service available at: http://localhost:8008"
        echo "   Use: kubectl port-forward svc/${SERVICE_NAME}-service 8008:8008 -n $NAMESPACE"
    fi
else
    echo "❌ Deployment failed!"
    kubectl logs -n $NAMESPACE -l app=${SERVICE_NAME} --tail=50
    exit 1
fi

echo "🎉 Enhanced Real-time Processing Agent deployed successfully!"
