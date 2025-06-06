#!/bin/bash
# =============================================================================
# MIP PLATFORM DEPLOYMENT SCRIPTS
# Enhanced Kubernetes Deployment with GPU Support and Multi-Agent Architecture
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="mip-platform"
HELM_RELEASE_NAME="mip-platform"
KEDA_VERSION="2.14.0"
GPU_OPERATOR_VERSION="v23.9.1"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if NVIDIA GPU nodes are available
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-v100 --no-headers 2>/dev/null | wc -l)
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "No GPU nodes found. Llama agent will use CPU fallback."
    else
        log_success "Found $GPU_NODES GPU-enabled nodes"
    fi
    
    log_success "Prerequisites check completed"
}

# =============================================================================
# INFRASTRUCTURE SETUP
# =============================================================================

setup_gpu_operator() {
    log_info "Setting up NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repository
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update
    
    # Install GPU Operator
    helm upgrade --install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator-resources \
        --create-namespace \
        --version $GPU_OPERATOR_VERSION \
        --set operator.defaultRuntime=containerd
    
    # Wait for GPU operator to be ready
    kubectl wait --for=condition=ready pod -l app=gpu-operator -n gpu-operator-resources --timeout=300s
    
    log_success "GPU Operator installed successfully"
}

setup_keda() {
    log_info "Setting up KEDA for advanced autoscaling..."
    
    # Add KEDA Helm repository
    helm repo add kedacore https://kedacore.github.io/charts
    helm repo update
    
    # Install KEDA
    helm upgrade --install keda kedacore/keda \
        --namespace keda \
        --create-namespace \
        --version $KEDA_VERSION
    
    # Wait for KEDA to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=keda-operator -n keda --timeout=300s
    
    log_success "KEDA installed successfully"
}

setup_prometheus_operator() {
    log_info "Setting up Prometheus Operator for monitoring..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus Operator
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
    
    log_success "Prometheus Operator installed successfully"
}

# =============================================================================
# STORAGE SETUP
# =============================================================================

setup_storage_classes() {
    log_info "Setting up storage classes..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: model-storage
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
volumeBindingMode: Immediate
EOF
    
    log_success "Storage classes created"
}

# =============================================================================
# MIP PLATFORM DEPLOYMENT
# =============================================================================

create_namespace() {
    log_info "Creating MIP namespace..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    log_success "Namespace $NAMESPACE created"
}

deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Generate random passwords if not provided
    POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}
    JWT_SECRET=${JWT_SECRET:-$(openssl rand -base64 64)}
    REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
    
    kubectl create secret generic mip-secrets \
        --namespace=$NAMESPACE \
        --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --from-literal=JWT_SECRET="$JWT_SECRET" \
        --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY:-your_openai_api_key}" \
        --from-literal=DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-your_deepseek_api_key}" \
        --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets deployed"
}

deploy_mip_platform() {
    log_info "Deploying MIP Platform..."
    
    # Apply main deployment
    kubectl apply -f k8s-enhanced-deployment.yaml
    
    # Apply KEDA configurations
    kubectl apply -f keda-autoscaling.yaml
    
    # Wait for core services to be ready
    log_info "Waiting for core services to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgresql -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    # Wait for agent services
    log_info "Waiting for AI agents to be ready..."
    kubectl wait --for=condition=ready pod -l tier=ai-agent -n $NAMESPACE --timeout=600s
    
    # Wait for gateway and frontend
    log_info "Waiting for gateway and frontend to be ready..."
    kubectl wait --for=condition=ready pod -l app=fastapi-gateway -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=react-dashboard -n $NAMESPACE --timeout=300s
    
    log_success "MIP Platform deployed successfully"
}

# =============================================================================
# POST-DEPLOYMENT VERIFICATION
# =============================================================================

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all pods are running
    FAILED_PODS=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [ "$FAILED_PODS" -gt 0 ]; then
        log_error "Some pods are not running. Check with: kubectl get pods -n $NAMESPACE"
        return 1
    fi
    
    # Check services are accessible
    log_info "Checking service endpoints..."
    
    # FastAPI Gateway
    GATEWAY_IP=$(kubectl get service fastapi-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -n "$GATEWAY_IP" ]; then
        log_success "FastAPI Gateway available at: http://$GATEWAY_IP:8000"
    else
        log_warning "FastAPI Gateway LoadBalancer IP not yet assigned"
    fi
    
    # React Dashboard
    DASHBOARD_IP=$(kubectl get service react-dashboard-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -n "$DASHBOARD_IP" ]; then
        log_success "React Dashboard available at: http://$DASHBOARD_IP:3000"
    else
        log_warning "React Dashboard LoadBalancer IP not yet assigned"
    fi
    
    # Check KEDA ScaledObjects
    SCALED_OBJECTS=$(kubectl get scaledobjects -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
    log_success "KEDA ScaledObjects deployed: $SCALED_OBJECTS"
    
    # Check HPA status
    kubectl get hpa -n $NAMESPACE
    
    log_success "Deployment verification completed"
}

# =============================================================================
# MONITORING SETUP
# =============================================================================

setup_monitoring_dashboards() {
    log_info "Setting up monitoring dashboards..."
    
    # Create ServiceMonitor for MIP agents
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mip-agents-monitor
  namespace: $NAMESPACE
  labels:
    app: mip-platform
spec:
  selector:
    matchLabels:
      tier: ai-agent
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mip-gateway-monitor
  namespace: $NAMESPACE
  labels:
    app: mip-platform
spec:
  selector:
    matchLabels:
      app: fastapi-gateway
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
EOF
    
    # Apply Prometheus rules
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: mip-platform-rules
  namespace: $NAMESPACE
  labels:
    app: mip-platform
spec:
  groups:
  - name: mip-agents
    rules:
    - alert: HighAgentLatency
      expr: avg(agent_response_time_ms) > 5000
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected in {{ \$labels.agent_type }} agent"
        description: "Agent {{ \$labels.agent_type }} response time is {{ \$value }}ms"
    
    - alert: AgentDown
      expr: up{job=~".*-agent"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Agent {{ \$labels.job }} is down"
        description: "Agent {{ \$labels.job }} has been down for more than 1 minute"
EOF
    
    log_success "Monitoring dashboards configured"
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup_deployment() {
    log_warning "Starting cleanup process..."
    
    # Delete MIP platform
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    # Delete KEDA
    helm uninstall keda -n keda || true
    kubectl delete namespace keda --ignore-not-found=true
    
    # Delete Prometheus
    helm uninstall prometheus -n monitoring || true
    kubectl delete namespace monitoring --ignore-not-found=true
    
    # Delete GPU Operator
    helm uninstall gpu-operator -n gpu-operator-resources || true
    kubectl delete namespace gpu-operator-resources --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# =============================================================================
# MAIN DEPLOYMENT FUNCTIONS
# =============================================================================

deploy_full_platform() {
    log_info "Starting full MIP platform deployment..."
    
    check_prerequisites
    
    # Infrastructure setup
    setup_storage_classes
    setup_keda
    setup_prometheus_operator
    
    # Setup GPU support if available
    if [ "$GPU_NODES" -gt 0 ]; then
        setup_gpu_operator
    fi
    
    # Deploy MIP platform
    create_namespace
    deploy_secrets
    deploy_mip_platform
    
    # Setup monitoring
    setup_monitoring_dashboards
    
    # Verify deployment
    verify_deployment
    
    log_success "MIP platform deployment completed successfully!"
    
    # Display access information
    echo ""
    echo "==============================================================================="
    echo "MIP PLATFORM DEPLOYMENT SUMMARY"
    echo "==============================================================================="
    echo "Namespace: $NAMESPACE"
    echo "Services deployed:"
    kubectl get services -n $NAMESPACE
    echo ""
    echo "To access the platform:"
    echo "1. FastAPI Gateway: kubectl port-forward service/fastapi-gateway-service 8000:8000 -n $NAMESPACE"
    echo "2. React Dashboard: kubectl port-forward service/react-dashboard-service 3000:3000 -n $NAMESPACE"
    echo "3. Monitoring: kubectl port-forward service/prometheus-operated 9090:9090 -n monitoring"
    echo ""
    echo "To check agent status: kubectl get pods -l tier=ai-agent -n $NAMESPACE"
    echo "==============================================================================="
}

deploy_development() {
    log_info "Starting development deployment (no GPU, minimal resources)..."
    
    # Set environment for development
    export DEVELOPMENT_MODE=true
    
    check_prerequisites
    setup_storage_classes
    create_namespace
    deploy_secrets
    
    # Deploy with development configurations
    sed 's/replicas: [0-9]*/replicas: 1/g' k8s-enhanced-deployment.yaml | \
    sed 's/nvidia.com\/gpu: 1//g' | \
    kubectl apply -f -
    
    verify_deployment
    log_success "Development deployment completed!"
}

# =============================================================================
# HELM CHART STRUCTURE
# =============================================================================

create_helm_chart() {
    log_info "Creating Helm chart structure..."
    
    mkdir -p mip-platform-chart/{templates,charts}
    
    cat > mip-platform-chart/Chart.yaml <<EOF
apiVersion: v2
name: mip-platform
description: Market Intelligence Platform with Multi-Agent AI Architecture
type: application
version: 1.0.0
appVersion: "1.0.0"
dependencies:
- name: postgresql
  version: 12.1.9
  repository: https://charts.bitnami.com/bitnami
- name: redis
  version: 17.4.3
  repository: https://charts.bitnami.com/bitnami
EOF
    
    cat > mip-platform-chart/values.yaml <<EOF
# MIP Platform Configuration
global:
  imageRegistry: ""
  imagePullSecrets: []

# Agent configurations
agents:
  finbert:
    replicas: 3
    image:
      repository: mip-platform/finbert-agent
      tag: latest
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"
  
  llama:
    replicas: 2
    image:
      repository: mip-platform/llama-agent
      tag: latest
    gpu:
      enabled: true
      count: 1
    resources:
      requests:
        memory: "8Gi"
        cpu: "2000m"
      limits:
        memory: "16Gi"
        cpu: "4000m"

# Gateway configuration
gateway:
  replicas: 4
  image:
    repository: mip-platform/fastapi-gateway
    tag: latest

# Dashboard configuration
dashboard:
  replicas: 3
  image:
    repository: mip-platform/react-dashboard
    tag: latest

# Autoscaling
autoscaling:
  enabled: true
  keda:
    enabled: true

# Monitoring
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

# Security
security:
  networkPolicies:
    enabled: true
  rbac:
    create: true
EOF
    
    # Copy templates
    cp k8s-enhanced-deployment.yaml mip-platform-chart/templates/
    cp keda-autoscaling.yaml mip-platform-chart/templates/
    
    log_success "Helm chart created in mip-platform-chart/"
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

case "${1:-deploy}" in
    "deploy")
        deploy_full_platform
        ;;
    "dev")
        deploy_development
        ;;
    "cleanup")
        cleanup_deployment
        ;;
    "verify")
        verify_deployment
        ;;
    "helm")
        create_helm_chart
        ;;
    "monitoring")
        setup_monitoring_dashboards
        ;;
    *)
        echo "Usage: $0 {deploy|dev|cleanup|verify|helm|monitoring}"
        echo ""
        echo "Commands:"
        echo "  deploy     - Full production deployment"
        echo "  dev        - Development deployment (minimal resources)"
        echo "  cleanup    - Remove all MIP platform components"
        echo "  verify     - Verify existing deployment"
        echo "  helm       - Create Helm chart structure"
        echo "  monitoring - Setup monitoring dashboards"
        exit 1
        ;;
esac
