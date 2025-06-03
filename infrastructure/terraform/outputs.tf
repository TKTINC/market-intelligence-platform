output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    region           = var.aws_region
    environment      = var.environment
    vpc_id          = module.vpc.vpc_id
    vpc_cidr        = module.vpc.vpc_cidr_block
    availability_zones = module.vpc.azs
    
    networking = {
      private_subnets = module.vpc.private_subnets
      public_subnets  = module.vpc.public_subnets
      intra_subnets   = module.vpc.intra_subnets
      nat_gateway_ids = module.vpc.natgw_ids
    }
    
    eks = {
      cluster_name     = local.cluster_name
      cluster_endpoint = module.eks.cluster_endpoint
      cluster_version  = module.eks.cluster_version
      oidc_issuer_url = module.eks.cluster_oidc_issuer_url
    }
    
    databases = {
      rds_primary_endpoint = aws_db_instance.main.endpoint
      rds_read_replicas   = [for replica in aws_db_instance.read_replica : replica.endpoint]
      redis_endpoint      = aws_elasticache_replication_group.main.primary_endpoint_address
    }
    
    streaming = {
      kafka_bootstrap_brokers_tls = aws_msk_cluster.main.bootstrap_brokers_tls
      kafka_zookeeper_connect    = aws_msk_cluster.main.zookeeper_connect_string
    }
    
    storage = {
      model_artifacts_bucket = aws_s3_bucket.model_artifacts.id
      raw_data_bucket       = aws_s3_bucket.raw_data.id
      options_data_bucket   = aws_s3_bucket.options_data.id
      backups_bucket        = aws_s3_bucket.backups.id
    }
    
    security = {
      kms_eks_key_id          = aws_kms_key.eks.key_id
      kms_trading_secrets_key = aws_kms_key.trading_secrets.key_id
      vpc_endpoints = {
        secrets_manager = aws_vpc_endpoint.secrets_manager.id
        kms            = aws_vpc_endpoint.kms.id
      }
    }
    
    secrets_management = {
      database_credentials    = aws_secretsmanager_secret.db_credentials.arn
      redis_credentials      = aws_secretsmanager_secret.redis_credentials.arn
      brokerage_secrets = {
        td_ameritrade       = aws_secretsmanager_secret.td_ameritrade.arn
        interactive_brokers = aws_secretsmanager_secret.interactive_brokers.arn
        schwab             = aws_secretsmanager_secret.schwab.arn
      }
    }
  }
}

# Connection strings for applications
output "database_connection_info" {
  description = "Database connection information"
  value = {
    primary_endpoint = aws_db_instance.main.endpoint
    port            = aws_db_instance.main.port
    database_name   = aws_db_instance.main.db_name
    secret_arn      = aws_secretsmanager_secret.db_credentials.arn
  }
  sensitive = true
}

output "redis_connection_info" {
  description = "Redis connection information"
  value = {
    primary_endpoint = aws_elasticache_replication_group.main.primary_endpoint_address
    port            = aws_elasticache_replication_group.main.port
    secret_arn      = aws_secretsmanager_secret.redis_credentials.arn
  }
  sensitive = true
}

output "kafka_connection_info" {
  description = "Kafka connection information"
  value = {
    bootstrap_brokers_tls = aws_msk_cluster.main.bootstrap_brokers_tls
    zookeeper_connect    = aws_msk_cluster.main.zookeeper_connect_string
    cluster_arn          = aws_msk_cluster.main.arn
  }
  sensitive = true
}

# Kubernetes configuration
output "kubeconfig_command" {
  description = "Command to update kubeconfig"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${local.cluster_name}"
}

# Trading-specific outputs
output "brokerage_integration_config" {
  description = "Configuration for brokerage integration"
  value = {
    vpc_endpoints = {
      secrets_manager = aws_vpc_endpoint.secrets_manager.dns_entry[0].dns_name
    }
    security_groups = {
      brokerage_integration = aws_security_group.brokerage_integration.id
    }
    secrets = {
      td_ameritrade       = aws_secretsmanager_secret.td_ameritrade.name
      interactive_brokers = aws_secretsmanager_secret.interactive_brokers.name
      schwab             = aws_secretsmanager_secret.schwab.name
    }
    kms_key_arn = aws_kms_key.trading_secrets.arn
  }
  sensitive = true
}

# Cost optimization information
output "cost_optimization_info" {
  description = "Information for cost optimization"
  value = {
    spot_instances = {
      ml_workloads = "c5.4xlarge SPOT instances for ML training"
    }
    storage_classes = {
      s3_lifecycle = "Raw data transitions to IA after 30 days, Glacier after 90 days"
    }
    right_sizing_recommendations = {
      redis = "Monitor usage and consider smaller instance if CPU < 50%"
      rds   = "Monitor connections and consider read replicas scaling"
    }
  }
}

# Monitoring and observability
output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    cloudwatch_log_groups = {
      msk   = aws_cloudwatch_log_group.msk.name
      redis = aws_cloudwatch_log_group.redis_slow.name
    }
    performance_insights = {
      rds_enabled = aws_db_instance.main.performance_insights_enabled
    }
  }
}

# Security compliance information
output "security_compliance" {
  description = "Security compliance information"
  value = {
    encryption = {
      at_rest = {
        rds   = "Encrypted with customer-managed KMS key"
        redis = "Encrypted with customer-managed KMS key"
        s3    = "Encrypted with customer-managed KMS key"
        eks   = "Secrets encrypted with customer-managed KMS key"
      }
      in_transit = {
        rds   = "TLS 1.2+"
        redis = "TLS encryption enabled"
        kafka = "TLS encryption enabled"
        vpc   = "All internal traffic uses TLS"
      }
    }
    access_control = {
      iam_least_privilege = "All roles follow least privilege principle"
      vpc_isolation      = "Private subnets with no direct internet access"
      security_groups    = "Restrictive security groups with specific port access"
    }
    secrets_management = {
      rotation_enabled = aws_secretsmanager_secret.td_ameritrade.rotation_enabled
      rotation_interval = "${var.secrets_rotation_days} days"
      kms_encryption   = "All secrets encrypted with customer-managed keys"
    }
  }
}
