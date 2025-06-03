#!/bin/bash

# Market Intelligence Platform - Repository Setup Script
# Creates the complete directory structure for MIP

echo "🚀 Setting up Market Intelligence Platform Repository..."

# Create root directories
mkdir -p docs
mkdir -p infrastructure/{terraform/{modules/{vpc,eks,rds,kafka},environments/{dev,staging,prod}},kubernetes/{namespaces,secrets,configmaps,deployments},monitoring/{prometheus,grafana,alerts}}
mkdir -p services/{data-ingestion,sentiment-analysis,price-prediction,options-strategy-engine,virtual-trading,brokerage-integration,api-gateway,explanation-service}
mkdir -p frontend/{public,src/{components/{Dashboard,Options,Watchlist,VirtualTrading,Brokerage},store/{slices,middleware},hooks,utils,styles}}
mkdir -p database/{migrations,seeds,schemas,triggers}
mkdir -p ml-models/{notebooks,training,evaluation,saved_models,deployment/{torchserve,model_configs}}
mkdir -p data-pipeline/{flink/{src,jobs,configs},kafka,streaming}
mkdir -p tests/{unit,integration,performance,e2e,fixtures}
mkdir -p scripts/{deployment,data,ml,monitoring}
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
mkdir -p helm/{templates/{deployments,services,configmaps,secrets,ingress}}

# Create service subdirectories
for service in data-ingestion sentiment-analysis price-prediction options-strategy-engine virtual-trading brokerage-integration api-gateway explanation-service; do
    mkdir -p "services/$service"/{src,tests}
done

# Create specific subdirectories for each service
mkdir -p services/data-ingestion/src/{data_sources,validators}
mkdir -p services/sentiment-analysis/src/{models,processors}
mkdir -p services/price-prediction/src/{models,features,training}
mkdir -p services/options-strategy-engine/src/{strategies,backtesting,risk_management}
mkdir -p services/virtual-trading/src/{execution,performance,analytics}
mkdir -p services/brokerage-integration/src/{oauth,brokers,security}
mkdir -p services/api-gateway/src/{routers,middleware,websocket,models}
mkdir -p services/explanation-service/src/{explanations,visualizations}

# Create frontend component subdirectories
mkdir -p frontend/src/components/Dashboard
mkdir -p frontend/src/components/Options
mkdir -p frontend/src/components/Watchlist
mkdir -p frontend/src/components/VirtualTrading
mkdir -p frontend/src/components/Brokerage

# Create root files
touch README.md
touch TECHNICAL_SPEC.md
touch SPRINT_GUIDE.md
touch LICENSE
touch .gitignore
touch docker-compose.yml
touch docker-compose.test.yml
touch requirements.txt
touch package.json
touch pyproject.toml

# Create documentation files
touch docs/API.md
touch docs/DEPLOYMENT.md
touch docs/DEVELOPMENT.md
touch docs/ARCHITECTURE.md

# Create Dockerfile for each service
for service in data-ingestion sentiment-analysis price-prediction options-strategy-engine virtual-trading brokerage-integration api-gateway explanation-service; do
    touch "services/$service/Dockerfile"
    touch "services/$service/requirements.txt"
done

# Create frontend files
touch frontend/Dockerfile
touch frontend/package.json
touch frontend/README.md

# Create database files
touch database/migrations/V1__initial_schema.sql
touch database/migrations/V2__options_tables.sql
touch database/migrations/V3__user_customization.sql
touch database/migrations/V4__virtual_trading.sql
touch database/seeds/sample_data.sql
touch database/seeds/test_data.sql

# Create ML model files
touch ml-models/notebooks/sentiment_analysis.ipynb
touch ml-models/notebooks/price_prediction.ipynb
touch ml-models/notebooks/options_strategies.ipynb
touch ml-models/notebooks/feature_engineering.ipynb
touch ml-models/training/train_sentiment.py
touch ml-models/training/train_price_prediction.py
touch ml-models/training/hyperparameter_tuning.py

# Create infrastructure files
touch infrastructure/terraform/main.tf
touch infrastructure/terraform/variables.tf
touch infrastructure/terraform/outputs.tf

# Create monitoring files
touch infrastructure/monitoring/prometheus/rules.yml
touch infrastructure/monitoring/grafana/dashboards.json
touch infrastructure/monitoring/alerts/trading-alerts.yml

# Create GitHub workflow files
touch .github/workflows/ci.yml
touch .github/workflows/cd.yml
touch .github/workflows/security-scan.yml
touch .github/workflows/model-training.yml
touch .github/PULL_REQUEST_TEMPLATE.md

# Create Helm files
touch helm/Chart.yaml
touch helm/values.yaml
touch helm/values-dev.yaml
touch helm/values-prod.yaml

# Create test files
touch tests/unit/test_models.py
touch tests/integration/test_api.py
touch tests/performance/test_load.py
touch tests/e2e/test_workflows.py

# Create script files
touch scripts/deployment/deploy.sh
touch scripts/deployment/rollback.sh
touch scripts/deployment/verify_deployment.py
touch scripts/data/seed_database.py
touch scripts/data/generate_test_data.py
touch scripts/ml/train_models.py
touch scripts/ml/evaluate_models.py
touch scripts/ml/model_drift_detection.py
touch scripts/monitoring/health_check.py
touch scripts/monitoring/performance_metrics.py

echo "✅ Repository structure created successfully!"
echo "📁 Total directories created: $(find . -type d | wc -l)"
echo "📄 Total files created: $(find . -type f | wc -l)"
echo ""
echo "Next steps:"
echo "1. Initialize git repository: git init"
echo "2. Add remote: git remote add origin https://github.com/yourusername/market-intelligence-platform"
echo "3. Start implementing Sprint 1 files"
echo ""
echo "🎯 Ready for Sprint 1 implementation!"
