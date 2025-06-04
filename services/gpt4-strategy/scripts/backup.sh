#!/bin/bash
# Backup GPT-4 Strategy Service data

set -e

BACKUP_DIR="/backups/gpt4-strategy"
DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACE="mip"

echo "💾 Starting GPT-4 Strategy Service backup - $DATE"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup database tables
echo "📄 Backing up database tables..."
pg_dump $DATABASE_URL \
    --table=gpt4_usage_log \
    --table=budget_alerts \
    --table=user_llm_settings \
    --table=strategy_performance \
    --table=agent_audit_log \
    --no-owner --no-privileges \
    > $BACKUP_DIR/$DATE/gpt4_strategy_tables.sql

# Backup Kubernetes manifests
echo "☸️  Backing up Kubernetes resources..."
kubectl get all -l app=gpt4-strategy -n $NAMESPACE -o yaml > $BACKUP_DIR/$DATE/k8s_resources.yaml
kubectl get configmap -l app=gpt4-strategy -n $NAMESPACE -o yaml > $BACKUP_DIR/$DATE/configmaps.yaml
kubectl get secrets -l app=gpt4-strategy -n $NAMESPACE -o yaml > $BACKUP_DIR/$DATE/secrets.yaml

# Backup service configuration
echo "⚙️  Backing up service configuration..."
cp -r ../k8s/ $BACKUP_DIR/$DATE/
cp -r ../helm/ $BACKUP_DIR/$DATE/
cp ../docker-compose.yml $BACKUP_DIR/$DATE/
cp ../requirements.txt $BACKUP_DIR/$DATE/

# Create backup metadata
cat > $BACKUP_DIR/$DATE/backup_info.txt <<EOF
GPT-4 Strategy Service Backup
============================
Date: $DATE
Service Version: $(kubectl get deployment gpt4-strategy-service -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
Kubernetes Namespace: $NAMESPACE
Database: $DATABASE_URL
Backup Type: Full service backup

Files included:
- gpt4_strategy_tables.sql (Database tables)
- k8s_resources.yaml (Kubernetes resources)
- configmaps.yaml (Configuration maps)
- secrets.yaml (Kubernetes secrets)
- k8s/ (Kubernetes manifests)
- helm/ (Helm charts)
- docker-compose.yml (Docker compose file)
- requirements.txt (Python dependencies)

Restore instructions:
1. Restore database: psql $DATABASE_URL < gpt4_strategy_tables.sql
2. Apply K8s resources: kubectl apply -f k8s_resources.yaml
3. Deploy service: kubectl apply -f k8s/
EOF

# Compress backup
echo "🗜️  Compressing backup..."
cd $BACKUP_DIR
tar -czf gpt4-strategy-backup-$DATE.tar.gz $DATE/
rm -rf $DATE/

# Upload to cloud storage (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    echo "☁️  Uploading to S3..."
    aws s3 cp gpt4-strategy-backup-$DATE.tar.gz s3://$AWS_S3_BUCKET/backups/gpt4-strategy/
fi

# Cleanup old backups (keep last 7 days)
echo "🧹 Cleaning up old backups..."
find $BACKUP_DIR -name "gpt4-strategy-backup-*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_DIR/gpt4-strategy-backup-$DATE.tar.gz"
