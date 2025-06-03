# Market Intelligence Platform - Enhanced Infrastructure
# Sprint 1, Prompt 1: Enhanced with Trading Security

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  backend "s3" {
    bucket = "mip-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "mip-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project = "MarketIntelligencePlatform"
      Environment = var.environment
      ManagedBy = "Terraform"
      CostCenter = "TradingIntelligence"
      SecurityLevel = "High"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name = "mip-${var.environment}"
  cluster_name = "${local.name}-eks"
  
  vpc_cidr = "10.0.0.0/16"
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
  
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  intra_subnets   = ["10.0.51.0/24", "10.0.52.0/24", "10.0.53.0/24"]
  
  # Trading-specific configurations
  brokerage_endpoints = [
    "api.tdameritrade.com",
    "api.interactivebrokers.com", 
    "api.schwab.com"
  ]
  
  common_tags = {
    Project = "MIP"
    Environment = var.environment
    Terraform = "true"
    TradingEnabled = "true"
  }
}

#################################
# VPC and Networking
#################################

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name}-vpc"
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets
  intra_subnets   = local.intra_subnets

  enable_nat_gateway = true
  enable_vpn_gateway = false
  single_nat_gateway = false
  one_nat_gateway_per_az = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Enhanced networking for trading
  enable_dhcp_options = true
  dhcp_options_domain_name = "mip.internal"
  dhcp_options_domain_name_servers = ["AmazonProvidedDNS"]

  # VPC Flow Logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # VPC Endpoints for brokerage APIs
  enable_s3_endpoint = true
  enable_dynamodb_endpoint = true
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  })

  public_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
    "SubnetType" = "public"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb"             = "1"
    "SubnetType" = "private"
  }
  
  intra_subnet_tags = {
    "SubnetType" = "database"
    "TradingData" = "true"
  }
}

#################################
# VPC Endpoints for Brokerage APIs
#################################

# VPC Endpoint for Secrets Manager (enhanced trading security)
resource "aws_vpc_endpoint" "secrets_manager" {
  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  
  private_dns_enabled = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name}-secrets-manager-endpoint"
    Purpose = "TradingCredentials"
  })
}

# VPC Endpoint for KMS
resource "aws_vpc_endpoint" "kms" {
  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.kms"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  
  private_dns_enabled = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name}-kms-endpoint"
    Purpose = "CredentialEncryption"
  })
}

#################################
# Security Groups
#################################

# Enhanced security group for VPC endpoints
resource "aws_security_group" "vpc_endpoints" {
  name_prefix = "${local.name}-vpc-endpoints-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for VPC endpoints"

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
    description = "HTTPS from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-vpc-endpoints-sg"
  })
}

# Trading-specific security group for brokerage integration
resource "aws_security_group" "brokerage_integration" {
  name_prefix = "${local.name}-brokerage-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for brokerage API integration"

  # HTTPS outbound to brokerage APIs
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS to brokerage APIs"
  }

  # Internal communication
  ingress {
    from_port = 8080
    to_port   = 8090
    protocol  = "tcp"
    self      = true
    description = "Internal service communication"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-brokerage-sg"
    Purpose = "BrokerageIntegration"
  })
}

#################################
# EKS Cluster with Enhanced Security
#################################

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"

  cluster_name    = local.cluster_name
  cluster_version = "1.28"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true
  
  # Enhanced security settings
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]  # Restrict in production
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Cluster logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # Primary node group for general workloads
    primary = {
      name = "primary"

      instance_types = ["t3.2xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 3
      max_size     = 10
      desired_size = 5

      ami_type = "AL2_x86_64"
      
      # Enhanced for trading workloads
      labels = {
        Environment = var.environment
        NodeGroup = "primary"
        WorkloadType = "general"
      }

      taints = []

      update_config = {
        max_unavailable_percentage = 33
      }

      # Enhanced storage for models and data
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 150
            encrypted             = true
            kms_key_id           = aws_kms_key.eks.arn
            delete_on_termination = true
          }
        }
      }
    }

    # High-performance node group for ML workloads
    ml_workloads = {
      name = "ml-workloads"

      instance_types = ["c5.4xlarge", "c5.2xlarge"]
      capacity_type  = "SPOT"

      min_size     = 0
      max_size     = 20
      desired_size = 2

      labels = {
        Environment = var.environment
        NodeGroup = "ml-workloads"
        WorkloadType = "ml-training"
      }

      taints = [
        {
          key    = "ml-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 4000
            throughput            = 250
            encrypted             = true
            kms_key_id           = aws_kms_key.eks.arn
            delete_on_termination = true
          }
        }
      }
    }
  }

  # Enhanced IRSA configuration
  enable_irsa = true

  # Cluster security group additional rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Node to cluster API"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  # Node security group additional rules
  node_security_group_additional_rules = {
    ingress_cluster_to_node_all_traffic = {
      description                   = "Cluster API to node all traffic"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
    
    # Trading-specific rules
    ingress_brokerage_apis = {
      description = "Access to brokerage APIs"
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
      type        = "egress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  tags = local.common_tags
}

#################################
# KMS Keys for Enhanced Encryption
#################################

resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "${local.name}-eks-encryption-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

resource "aws_kms_key" "trading_secrets" {
  description             = "Trading Secrets Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Secrets Manager access"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:Encrypt",
          "kms:GenerateDataKey*",
          "kms:ReEncrypt*"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name}-trading-secrets-key"
    Purpose = "TradingCredentials"
  })
}

resource "aws_kms_alias" "trading_secrets" {
  name          = "alias/${local.name}-trading-secrets"
  target_key_id = aws_kms_key.trading_secrets.key_id
}

#################################
# RDS PostgreSQL with Read Replicas
#################################

# DB subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${local.name}-db-subnet-group"
  subnet_ids = module.vpc.intra_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name}-db-subnet-group"
  })
}

# Enhanced security group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for RDS PostgreSQL"

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "PostgreSQL from EKS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-rds-sg"
  })
}

# Primary RDS instance
resource "aws_db_instance" "main" {
  identifier = "${local.name}-primary"

  # Engine configuration
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.m6g.4xlarge"
  allocated_storage    = 1000
  max_allocated_storage = 5000
  storage_type         = "gp3"
  storage_encrypted    = true
  kms_key_id          = aws_kms_key.trading_secrets.arn

  # Database configuration
  db_name  = "mip_production"
  username = "mip_admin"
  password = random_password.db_password.result
  port     = 5432

  # Networking
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Backup and maintenance
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  # Enhanced logging
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  # High availability and performance
  multi_az = true
  auto_minor_version_upgrade = false
  apply_immediately = false

  # Deletion protection for production
  deletion_protection = var.environment == "prod" ? true : false
  skip_final_snapshot = var.environment == "prod" ? false : true
  final_snapshot_identifier = var.environment == "prod" ? "${local.name}-final-snapshot" : null

  tags = merge(local.common_tags, {
    Name = "${local.name}-primary-db"
    Purpose = "TradingData"
  })
}

# Read replica for read-only queries
resource "aws_db_instance" "read_replica" {
  count = var.environment == "prod" ? 2 : 1

  identifier = "${local.name}-read-replica-${count.index + 1}"

  # Replica configuration
  replicate_source_db = aws_db_instance.main.id
  instance_class      = "db.m6g.2xlarge"
  
  # Same networking as primary
  publicly_accessible = false
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  # Performance Insights
  performance_insights_enabled = true

  auto_minor_version_upgrade = false
  
  tags = merge(local.common_tags, {
    Name = "${local.name}-read-replica-${count.index + 1}"
    Purpose = "ReadOnlyQueries"
  })
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store DB credentials in Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name                    = "${local.name}/db/credentials"
  description             = "Database credentials for MIP"
  kms_key_id             = aws_kms_key.trading_secrets.arn
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name}-db-credentials"
    Purpose = "DatabaseAccess"
  })
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = random_password.db_password.result
    endpoint = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
  })
}

#################################
# ElastiCache Redis Cluster
#################################

# Redis subnet group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name}-redis-subnet-group"
  })
}

# Security group for Redis
resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for Redis cluster"

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Redis from EKS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-redis-sg"
  })
}

# Redis replication group
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${local.name}-redis"
  description               = "Redis cluster for MIP caching and sessions"

  # Engine configuration
  engine               = "redis"
  engine_version       = "7.0"
  node_type           = "cache.r6g.2xlarge"
  port                = 6379
  parameter_group_name = "default.redis7"

  # Cluster configuration
  num_cache_clusters = 6
  
  # Networking
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth.result
  kms_key_id                = aws_kms_key.trading_secrets.arn

  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  # Maintenance
  maintenance_window = "sun:05:00-sun:07:00"
  auto_minor_version_upgrade = false

  # Enhanced logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format      = "text"
    log_type        = "slow-log"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-redis-cluster"
    Purpose = "CachingAndSessions"
  })
}

# Redis auth token
resource "random_password" "redis_auth" {
  length  = 32
  special = false
}

# Store Redis credentials in Secrets Manager
resource "aws_secretsmanager_secret" "redis_credentials" {
  name                    = "${local.name}/redis/credentials"
  description             = "Redis credentials for MIP"
  kms_key_id             = aws_kms_key.trading_secrets.arn
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name}-redis-credentials"
  })
}

resource "aws_secretsmanager_secret_version" "redis_credentials" {
  secret_id = aws_secretsmanager_secret.redis_credentials.id
  secret_string = jsonencode({
    primary_endpoint = aws_elasticache_replication_group.main.primary_endpoint_address
    port            = aws_elasticache_replication_group.main.port
    auth_token      = random_password.redis_auth.result
  })
}

#################################
# MSK Kafka Cluster
#################################

# Kafka configuration
resource "aws_msk_configuration" "main" {
  kafka_versions = ["3.5.1"]
  name           = "${local.name}-kafka-config"
  description    = "Kafka configuration for MIP"

  server_properties = <<PROPERTIES
auto.create.topics.enable=false
default.replication.factor=3
min.insync.replicas=2
num.partitions=50
offsets.topic.replication.factor=3
transaction.state.log.replication.factor=3
transaction.state.log.min.isr=2
log.retention.hours=168
log.segment.bytes=1073741824
compression.type=lz4
PROPERTIES
}

# Security group for MSK
resource "aws_security_group" "msk" {
  name_prefix = "${local.name}-msk-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for MSK cluster"

  # Kafka broker communication
  ingress {
    from_port       = 9092
    to_port         = 9092
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Kafka broker from EKS"
  }

  # Kafka broker communication (TLS)
  ingress {
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Kafka broker TLS from EKS"
  }

  # Zookeeper
  ingress {
    from_port       = 2181
    to_port         = 2181
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Zookeeper from EKS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-msk-sg"
  })
}

# MSK cluster
resource "aws_msk_cluster" "main" {
  cluster_name           = "${local.name}-kafka"
  kafka_version         = "3.5.1"
  number_of_broker_nodes = 6  # 2 per AZ for high availability

  broker_node_group_info {
    instance_type   = "kafka.m5.4xlarge"
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = 1000  # 1TB per broker
        provisioned_throughput {
          enabled           = true
          volume_throughput = 250
        }
      }
    }

    connectivity_info {
      public_access {
        type = "DISABLED"
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  encryption_info {
    encryption_at_rest_kms_key_id = aws_kms_key.trading_secrets.arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  client_authentication {
    tls {}
    sasl {
      iam = true
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
      s3 {
        enabled = true
        bucket  = aws_s3_bucket.kafka_logs.id
        prefix  = "kafka-logs"
      }
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-kafka-cluster"
    Purpose = "DataStreaming"
  })
}

#################################
# S3 Buckets for Data and Models
#################################

# Model artifacts bucket
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${local.name}-model-artifacts-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.name}-model-artifacts"
    Purpose = "MLModels"
  })
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.trading_secrets.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Raw data bucket
resource "aws_s3_bucket" "raw_data" {
  bucket = "${local.name}-raw-data-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.name}-raw-data"
    Purpose = "DataIngestion"
  })
}

resource "aws_s3_bucket_lifecycle_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    id     = "transition_to_ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 2555  # 7 years retention
    }
  }
}

# Options data bucket (NEW)
resource "aws_s3_bucket" "options_data" {
  bucket = "${local.name}-options-data-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.name}-options-data"
    Purpose = "OptionsIntelligence"
  })
}

resource "aws_s3_bucket_server_side_encryption_configuration" "options_data" {
  bucket = aws_s3_bucket.options_data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.trading_secrets.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Backups bucket
resource "aws_s3_bucket" "backups" {
  bucket = "${local.name}-backups-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.name}-backups"
    Purpose = "DataBackup"
  })
}

# Kafka logs bucket
resource "aws_s3_bucket" "kafka_logs" {
  bucket = "${local.name}-kafka-logs-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.name}-kafka-logs"
    Purpose = "LoggingAndMonitoring"
  })
}

# Random suffix for bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

#################################
# Secrets Manager for Brokerage Credentials
#################################

# TD Ameritrade credentials
resource "aws_secretsmanager_secret" "td_ameritrade" {
  name                    = "${local.name}/brokerage/td-ameritrade"
  description             = "TD Ameritrade API credentials"
  kms_key_id             = aws_kms_key.trading_secrets.arn
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name}-td-ameritrade-creds"
    Purpose = "BrokerageIntegration"
    Broker = "TDAmeritrade"
  })
}

# Interactive Brokers credentials
resource "aws_secretsmanager_secret" "interactive_brokers" {
  name                    = "${local.name}/brokerage/interactive-brokers"
  description             = "Interactive Brokers API credentials"
  kms_key_id             = aws_kms_key.trading_secrets.arn
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name}-interactive-brokers-creds"
    Purpose = "BrokerageIntegration"
    Broker = "InteractiveBrokers"
  })
}

# Schwab credentials
resource "aws_secretsmanager_secret" "schwab" {
  name                    = "${local.name}/brokerage/schwab"
  description             = "Charles Schwab API credentials"
  kms_key_id             = aws_kms_key.trading_secrets.arn
  recovery_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.name}-schwab-creds"
    Purpose = "BrokerageIntegration"
    Broker = "CharlesSchwab"
  })
}

# Auto-rotation for secrets (4-hour rotation as specified)
resource "aws_secretsmanager_secret_rotation" "brokerage_secrets" {
  for_each = toset([
    aws_secretsmanager_secret.td_ameritrade.id,
    aws_secretsmanager_secret.interactive_brokers.id,
    aws_secretsmanager_secret.schwab.id
  ])

  secret_id           = each.value
  rotation_lambda_arn = aws_lambda_function.secrets_rotation.arn

  rotation_rules {
    automatically_after_days = 1  # Daily rotation for enhanced security
  }
}

#################################
# IAM Roles and Policies
#################################

# EKS service role
resource "aws_iam_role" "eks_service_role" {
  name = "${local.name}-eks-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Enhanced IAM role for brokerage integration service
resource "aws_iam_role" "brokerage_service_role" {
  name = "${local.name}-brokerage-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Purpose = "BrokerageIntegration"
  })
}

# Policy for brokerage service (read-only Phase 1)
resource "aws_iam_role_policy" "brokerage_service_policy" {
  name = "${local.name}-brokerage-service-policy"
  role = aws_iam_role.brokerage_service_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.td_ameritrade.arn,
          aws_secretsmanager_secret.interactive_brokers.arn,
          aws_secretsmanager_secret.schwab.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.trading_secrets.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# RDS monitoring role
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"]

  tags = local.common_tags
}

#################################
# Lambda for Secrets Rotation
#################################

# Lambda function for secrets rotation
resource "aws_lambda_function" "secrets_rotation" {
  filename         = "secrets_rotation.zip"
  function_name    = "${local.name}-secrets-rotation"
  role            = aws_iam_role.lambda_secrets_rotation.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.11"
  timeout         = 60

  environment {
    variables = {
      ENVIRONMENT = var.environment
    }
  }

  tags = merge(local.common_tags, {
    Purpose = "SecretsRotation"
  })
}

# IAM role for Lambda secrets rotation
resource "aws_iam_role" "lambda_secrets_rotation" {
  name = "${local.name}-lambda-secrets-rotation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Policy for Lambda secrets rotation
resource "aws_iam_role_policy" "lambda_secrets_rotation" {
  name = "${local.name}-lambda-secrets-rotation-policy"
  role = aws_iam_role.lambda_secrets_rotation.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:*"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.trading_secrets.arn
      }
    ]
  })
}

#################################
# CloudWatch Log Groups
#################################

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${local.name}"
  retention_in_days = 14
  kms_key_id       = aws_kms_key.trading_secrets.arn

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/redis/${local.name}/slow-log"
  retention_in_days = 7
  kms_key_id       = aws_kms_key.trading_secrets.arn

  tags = local.common_tags
}

#################################
# Outputs
#################################

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = local.cluster_name
}

output "rds_endpoint" {
  description = "RDS primary endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}

output "kafka_bootstrap_brokers" {
  description = "MSK bootstrap brokers"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    model_artifacts = aws_s3_bucket.model_artifacts.id
    raw_data       = aws_s3_bucket.raw_data.id
    options_data   = aws_s3_bucket.options_data.id
    backups        = aws_s3_bucket.backups.id
  }
}

output "brokerage_secrets" {
  description = "Brokerage secrets ARNs"
  value = {
    td_ameritrade       = aws_secretsmanager_secret.td_ameritrade.arn
    interactive_brokers = aws_secretsmanager_secret.interactive_brokers.arn
    schwab             = aws_secretsmanager_secret.schwab.arn
  }
  sensitive = true
}
