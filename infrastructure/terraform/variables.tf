variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "market-intelligence-platform"
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway for all private subnets"
  type        = bool
  default     = false
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "eks_node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size      = number
    max_size      = number
    desired_size  = number
  }))
  default = {
    primary = {
      instance_types = ["t3.2xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size      = 3
      max_size      = 10
      desired_size  = 5
    }
    ml_workloads = {
      instance_types = ["c5.4xlarge"]
      capacity_type  = "SPOT"
      min_size      = 0
      max_size      = 20
      desired_size  = 2
    }
  }
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.m6g.4xlarge"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 1000
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.2xlarge"
}

variable "redis_num_cache_clusters" {
  description = "Number of Redis cache clusters"
  type        = number
  default     = 6
}

variable "kafka_instance_type" {
  description = "Kafka broker instance type"
  type        = string
  default     = "kafka.m5.4xlarge"
}

variable "kafka_ebs_volume_size" {
  description = "Kafka EBS volume size in GB"
  type        = number
  default     = 1000
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS and Redis"
  type        = bool
  default     = true
}

variable "brokerage_api_endpoints" {
  description = "List of brokerage API endpoints for VPC endpoints"
  type        = list(string)
  default = [
    "api.tdameritrade.com",
    "api.interactivebrokers.com",
    "api.schwab.com"
  ]
}

variable "trading_security_level" {
  description = "Trading security level (standard, enhanced, maximum)"
  type        = string
  default     = "enhanced"
  
  validation {
    condition     = contains(["standard", "enhanced", "maximum"], var.trading_security_level)
    error_message = "Trading security level must be standard, enhanced, or maximum."
  }
}

variable "enable_secrets_rotation" {
  description = "Enable automatic rotation of trading secrets"
  type        = bool
  default     = true
}

variable "secrets_rotation_days" {
  description = "Number of days between secret rotations"
  type        = number
  default     = 1
}

variable "options_data_retention_days" {
  description = "Number of days to retain options data"
  type        = number
  default     = 90
}

variable "enable_multi_az" {
  description = "Enable Multi-AZ for RDS"
  type        = bool
  default     = true
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}
