#!/bin/bash

# ==============================================================================
# MARKET INTELLIGENCE PLATFORM - AWS DEPLOYMENT GUIDE
# Complete AWS cloud deployment automation
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

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
CLUSTER_NAME="mip-production-cluster"
NAMESPACE="mip-production"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}MIP AWS DEPLOYMENT GUIDE${NC}"
echo -e "${BLUE}============================================${NC}"

# Check AWS prerequisites
print_step "Checking AWS prerequisites..."

required_tools=("aws" "kubectl" "eksctl" "helm" "docker")
missing_tools=()

for tool in "${required_tools[@]}"; do
    if ! command -v $tool &> /dev/null; then
        missing_tools+=($tool)
    fi
done

if [[ ${#missing_tools[@]} -gt 0 ]]; then
    print_error "Missing tools: ${missing_tools[*]}"
    print_info "Please install missing tools and try again"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured"
    print_info "Run 'aws configure' to set up credentials"
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_success "AWS Account: $AWS_ACCOUNT_ID"

# Create ECR repositories
print_step "Creating ECR repositories..."

services=("api-gateway" "agent-orchestration" "sentiment-analysis" "gpt4-strategy" "virtual-trading")

for service in "${services[@]}"; do
    if ! aws ecr describe-repositories --repository-names "mip-$service" --region $AWS_REGION &> /dev/null; then
        aws ecr create-repository --repository-name "mip-$service" --region $AWS_REGION
        print_success "Created ECR repository: mip-$service"
    else
        print_success "ECR repository exists: mip-$service"
    fi
done

# Create EKS cluster
print_step "Creating EKS cluster..."

if ! eksctl get cluster --name $CLUSTER_NAME --region $AWS_REGION &> /dev/null; then
    cat > cluster-config.yaml << EOF_CLUSTER
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $AWS_REGION
  version: "1.28"

managedNodeGroups:
  - name: worker-nodes
    instanceType: m5.large
    minSize: 2
    maxSize: 6
    desiredCapacity: 3
    volumeSize: 50
EOF_CLUSTER
    
    eksctl create cluster -f cluster-config.yaml
    print_success "EKS cluster created"
else
    print_success "EKS cluster exists"
fi

# Update kubeconfig
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME
print_success "Kubeconfig updated"

# Build and push images
print_step "Building and pushing images..."

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

for service in "${services[@]}"; do
    if [[ -d "services/$service" ]]; then
        print_step "Building $service..."
        cd "services/$service"
        docker build -t "mip-$service:latest" .
        docker tag "mip-$service:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mip-$service:latest"
        docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mip-$service:latest"
        cd ../..
        print_success "$service pushed to ECR"
    fi
done

# Deploy to Kubernetes
print_step "Deploying to Kubernetes..."

kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Create basic deployment for API Gateway
kubectl create deployment api-gateway \
    --image="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mip-api-gateway:latest" \
    --namespace=$NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

kubectl expose deployment api-gateway \
    --port=8000 --target-port=8000 --type=LoadBalancer \
    --namespace=$NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

print_step "Waiting for deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n $NAMESPACE

# Get service URL
API_URL=$(kubectl get svc api-gateway -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

print_success "AWS deployment completed!"
echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo "  API Gateway: http://$API_URL"
echo ""
echo -e "${BLUE}Management commands:${NC}"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -f deployment/api-gateway -n $NAMESPACE"

# Save deployment info
cat > aws_deployment_info.txt << EOF_INFO
AWS Deployment Information
========================

Cluster: $CLUSTER_NAME
Region: $AWS_REGION
Namespace: $NAMESPACE
API Gateway: http://$API_URL

Commands:
- Update kubeconfig: aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME
- View pods: kubectl get pods -n $NAMESPACE
- View logs: kubectl logs -f deployment/api-gateway -n $NAMESPACE
EOF_INFO

print_success "Deployment info saved to aws_deployment_info.txt"
