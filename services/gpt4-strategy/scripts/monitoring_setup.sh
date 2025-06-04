#!/bin/bash
# Setup monitoring for GPT-4 Strategy Service

set -e

NAMESPACE="mip"
SERVICE_NAME="gpt4-strategy"

echo "📊 Setting up monitoring for GPT-4 Strategy Service"

# Create monitoring namespace if it doesn't exist
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Install Prometheus if not exists
if ! kubectl get deployment prometheus-server -n monitoring >/dev/null 2>&1; then
    echo "🔍 Installing Prometheus..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/prometheus \
        --namespace monitoring \
        --set server.service.type=LoadBalancer \
        --set server.persistentVolume.size=10Gi
fi

# Install Grafana if not exists
if ! kubectl get deployment grafana -n monitoring >/dev/null 2>&1; then
    echo "📈 Installing Grafana..."
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    helm install grafana grafana/grafana \
        --namespace monitoring \
        --set service.type=LoadBalancer \
        --set persistence.enabled=true \
        --set persistence.size=5Gi \
        --set adminPassword=admin123
fi

# Create ServiceMonitor for GPT-4 Strategy Service
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: $SERVICE_NAME-monitor
  namespace: monitoring
  labels:
    app: $SERVICE_NAME
spec:
  selector:
    matchLabels:
      app: $SERVICE_NAME
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
  namespaceSelector:
    matchNames:
    - $NAMESPACE
EOF

# Create PrometheusRule for alerting
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: $SERVICE_NAME-alerts
  namespace: monitoring
  labels:
    app: $SERVICE_NAME
    prometheus: kube-prometheus
spec:
  groups:
  - name: gpt4-strategy.rules
    rules:
    - alert: GPT4HighErrorRate
      expr: (rate(gpt4_strategy_errors_total[5m]) / rate(gpt4_strategy_requests_total[5m])) > 0.1
      for: 2m
      labels:
        severity: warning
        service: gpt4-strategy
      annotations:
        summary: "High error rate in GPT-4 Strategy Service"
        description: "Error rate is {{ \$value | humanizePercentage }} for the last 5 minutes"
        
    - alert: GPT4HighResponseTime
      expr: histogram_quantile(0.95, rate(gpt4_strategy_response_time_seconds_bucket[5m])) > 5
      for: 3m
      labels:
        severity: warning
        service: gpt4-strategy
      annotations:
        summary: "High response time in GPT-4 Strategy Service"
        description: "95th percentile response time is {{ \$value }}s"
        
    - alert: GPT4ServiceDown
      expr: up{job="gpt4-strategy"} == 0
      for: 1m
      labels:
        severity: critical
        service: gpt4-strategy
      annotations:
        summary: "GPT-4 Strategy Service is down"
        description: "GPT-4 Strategy Service has been down for more than 1 minute"
EOF

# Create Grafana dashboard ConfigMap
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: $SERVICE_NAME-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  gpt4-strategy-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "GPT-4 Strategy Service",
        "tags": ["mip", "gpt4", "strategy"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(gpt4_strategy_requests_total[5m])",
                "legendFormat": "Requests/sec"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Error Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(gpt4_strategy_errors_total[5m]) / rate(gpt4_strategy_requests_total[5m])",
                "legendFormat": "Error Rate"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Response Time",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.50, rate(gpt4_strategy_response_time_seconds_bucket[5m]))",
                "legendFormat": "p50"
              },
              {
                "expr": "histogram_quantile(0.95, rate(gpt4_strategy_response_time_seconds_bucket[5m]))",
                "legendFormat": "p95"
              }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
          },
          {
            "id": 4,
            "title": "Cost Tracking",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(gpt4_strategy_cost_usd_total[1h])",
                "legendFormat": "Cost/hour"
              }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
          }
        ],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "5s"
      }
    }
EOF

echo "✅ Monitoring setup completed!"
echo ""
echo "🔍 Prometheus UI: kubectl port-forward svc/prometheus-server 9090:80 -n monitoring"
echo "📈 Grafana UI: kubectl port-forward svc/grafana 3000:80 -n monitoring"
echo "   Default login: admin/admin123"
echo ""
echo "📊 Dashboard will be automatically imported into Grafana"
