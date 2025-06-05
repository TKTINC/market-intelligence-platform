#!/bin/bash
# TFT Forecasting Service Deployment Script

set -e

# Configuration
SERVICE_NAME="tft-forecasting"
NAMESPACE="mip"
IMAGE_TAG=${1:-latest}
ENVIRONMENT=${2:-staging}

echo "🚀 Deploying TFT Forecasting Service"
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }

# Check for GPU support
echo "🔍 Checking GPU support..."
if kubectl get nodes -o json | grep -q "nvidia.com/gpu"; then
    echo "✅ GPU nodes found"
else
    echo "⚠️  No GPU nodes found - service will run in CPU mode"
fi

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
if ! kubectl get secret db-secrets -n $NAMESPACE >/dev/null 2>&1; then
    echo "❌ Database secrets not found. Please create them first:"
    echo "kubectl create secret generic db-secrets --from-literal=url=YOUR_DB_URL -n $NAMESPACE"
    exit 1
fi

# Create model storage PVC
echo "💾 Setting up model storage..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tft-model-storage
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd
EOF

# Update image tag in deployment
echo "📝 Updating deployment configuration..."
sed -i.bak "s|mip/$SERVICE_NAME:.*|mip/$SERVICE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml

# Apply Kubernetes manifests
echo "🚢 Applying Kubernetes manifests..."
kubectl apply -f k8s/ -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/$SERVICE_NAME-service -n $NAMESPACE

# Check for GPU allocation
echo "🎮 Checking GPU allocation..."
kubectl get pods -l app=$SERVICE_NAME -n $NAMESPACE -o custom-columns=NAME:.metadata.name,GPU:.spec.containers[0].resources.requests.'nvidia\.com/gpu'

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -l app=$SERVICE_NAME -n $NAMESPACE
kubectl get services -l app=$SERVICE_NAME -n $NAMESPACE

# Health check
echo "🏥 Performing health check..."
SERVICE_IP=$(kubectl get service $SERVICE_NAME-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
if kubectl exec -n $NAMESPACE deployment/$SERVICE_NAME-service -- curl -f http://$SERVICE_IP:8007/health; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    echo "📋 Recent logs:"
    kubectl logs -l app=$SERVICE_NAME -n $NAMESPACE --tail=50
    exit 1
fi

# Check model status
echo "🤖 Checking model status..."
kubectl exec -n $NAMESPACE deployment/$SERVICE_NAME-service -- curl -s http://$SERVICE_IP:8007/model/status | jq '.'

# Show logs
echo "📋 Recent logs:"
kubectl logs -l app=$SERVICE_NAME -n $NAMESPACE --tail=20

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service endpoints:"
echo "  - Health: http://$SERVICE_NAME-service.$NAMESPACE.svc.cluster.local:8007/health"
echo "  - Forecasting: http://$SERVICE_NAME-service.$NAMESPACE.svc.cluster.local:8007/forecast/generate"
echo "  - Metrics: http://$SERVICE_NAME-service.$NAMESPACE.svc.cluster.local:8007/metrics"
echo ""
echo "🔧 Management commands:"
echo "  - View logs: kubectl logs -l app=$SERVICE_NAME -n $NAMESPACE -f"
echo "  - Port forward: kubectl port-forward svc/$SERVICE_NAME-service 8007:8007 -n $NAMESPACE"
echo "  - Scale: kubectl scale deployment $SERVICE_NAME-service --replicas=3 -n $NAMESPACE"

# Restore original deployment file
mv k8s/deployment.yaml.bak k8s/deployment.yaml
