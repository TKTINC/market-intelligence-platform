#!/bin/bash
# GPT-4 Strategy Service Deployment Script

set -e

# Configuration
SERVICE_NAME="gpt4-strategy"
NAMESPACE="mip"
IMAGE_TAG=${1:-latest}
ENVIRONMENT=${2:-staging}

echo "🚀 Deploying GPT-4 Strategy Service"
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }

# Build and push image
echo "🔨 Building Docker image..."
docker build -t mip/$SERVICE_NAME:$IMAGE_TAG .

if [ "$ENVIRONMENT" != "local" ]; then
    echo "📤 Pushing image to registry..."
    docker tag mip/$SERVICE_NAME:$IMAGE_TAG your-registry.com/mip/$SERVICE_NAME:$IMAGE_TAG
    docker push your-registry.com/mip/$SERVICE_NAME:$IMAGE_TAG
fi

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (ensure they exist)
echo "🔐 Checking secrets..."
if ! kubectl get secret openai-secrets -n $NAMESPACE >/dev/null 2>&1; then
    echo "❌ OpenAI secrets not found. Please create them first:"
    echo "kubectl create secret generic openai-secrets --from-literal=api-key=YOUR_API_KEY -n $NAMESPACE"
    exit 1
fi

if ! kubectl get secret db-secrets -n $NAMESPACE >/dev/null 2>&1; then
    echo "❌ Database secrets not found. Please create them first:"
    echo "kubectl create secret generic db-secrets --from-literal=url=YOUR_DB_URL -n $NAMESPACE"
    exit 1
fi

# Update image tag in deployment
echo "📝 Updating deployment configuration..."
sed -i.bak "s|mip/$SERVICE_NAME:.*|mip/$SERVICE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml

# Apply Kubernetes manifests
echo "🚢 Applying Kubernetes manifests..."
kubectl apply -f k8s/ -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$SERVICE_NAME-service -n $NAMESPACE

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -l app=$SERVICE_NAME -n $NAMESPACE
kubectl get services -l app=$SERVICE_NAME -n $NAMESPACE

# Health check
echo "🏥 Performing health check..."
SERVICE_IP=$(kubectl get service $SERVICE_NAME-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
if kubectl exec -n $NAMESPACE deployment/$SERVICE_NAME-service -- curl -f http://$SERVICE_IP:8006/health; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    exit 1
fi

# Show logs
echo "📋 Recent logs:"
kubectl logs -l app=$SERVICE_NAME -n $NAMESPACE --tail=20

echo "🎉 Deployment completed successfully!"
echo "Service is available at: $SERVICE_NAME-service.$NAMESPACE.svc.cluster.local:8006"

# Restore original deployment file
mv k8s/deployment.yaml.bak k8s/deployment.yaml
