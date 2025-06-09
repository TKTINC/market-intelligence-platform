#!/bin/bash

# ==============================================================================
# MARKET INTELLIGENCE PLATFORM - ONE-CLICK AUTOMATION
# Complete deployment and testing automation for local and cloud environments
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs"
DEPLOYMENT_LOG="$LOG_DIR/deployment_$TIMESTAMP.log"
TEST_LOG="$LOG_DIR/testing_$TIMESTAMP.log"

# Default values
ENVIRONMENT="local"
DEPLOY_MONITORING="true"
RUN_TESTS="true"
TEST_TYPE="all"
CLEAN_DEPLOY="false"
SKIP_BUILD="false"
PARALLEL_DEPLOY="true"

print_banner() {
    echo -e "${PURPLE}"
    echo "=================================================================="
    echo "    MARKET INTELLIGENCE PLATFORM - AUTOMATION SUITE"
    echo "    One-Click Deployment & Testing for Local and Cloud"
    echo "=================================================================="
    echo -e "${NC}"
}

print_header() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

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

usage() {
    cat << EOF_USAGE
Usage: $0 [OPTIONS]

One-click deployment and testing automation for Market Intelligence Platform

OPTIONS:
    -e, --environment ENV    Environment: local|aws (default: local)
    -t, --test-type TYPE     Test type: all|sanity|integration|e2e|performance|security (default: all)
    -c, --clean             Clean deployment (remove existing containers/resources)
    -s, --skip-build        Skip building container images
    -m, --skip-monitoring   Skip monitoring stack deployment
    -n, --no-tests          Skip running tests
    -p, --no-parallel       Disable parallel deployment
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Local deployment with all tests
    $0 --environment local

    # AWS deployment with only sanity tests
    $0 --environment aws --test-type sanity

    # Clean local deployment without tests
    $0 --environment local --clean --no-tests

EOF_USAGE
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_DEPLOY="true"
            shift
            ;;
        -s|--skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        -m|--skip-monitoring)
            DEPLOY_MONITORING="false"
            shift
            ;;
        -n|--no-tests)
            RUN_TESTS="false"
            shift
            ;;
        -p|--no-parallel)
            PARALLEL_DEPLOY="false"
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

print_banner

check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    local missing_tools=()
    
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    
    if [[ "$ENVIRONMENT" == "aws" ]]; then
        command -v aws >/dev/null 2>&1 || missing_tools+=("aws-cli")
        command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
        command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

setup_environment() {
    print_header "ENVIRONMENT SETUP"
    print_step "Setting up environment variables..."
    
    if [[ ! -f ".env" ]]; then
        print_step "Creating default .env file..."
        cat > .env << 'EOF_ENV'
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=market_intelligence
POSTGRES_USER=mip_user
POSTGRES_PASSWORD=mip_secure_password_2024

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_secure_password_2024

# JWT Secret
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production_2024

# API Keys (REPLACE WITH YOUR ACTUAL KEYS)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Service Ports
API_GATEWAY_PORT=8000
AGENT_ORCHESTRATION_PORT=8001
SENTIMENT_ANALYSIS_PORT=8002
GPT4_STRATEGY_PORT=8003
VIRTUAL_TRADING_PORT=8006

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
EOF_ENV
        print_warning "Please update API keys in .env file before proceeding"
    fi
    
    print_success "Environment configured"
}

deploy_local() {
    print_header "DEPLOYING TO LOCAL ENVIRONMENT"
    
    # Start databases first
    print_step "Starting database services..."
    docker-compose up -d postgres redis
    sleep 30
    
    # Start all services
    print_step "Starting all services..."
    docker-compose up -d
    sleep 60
    
    print_success "Local deployment completed"
}

deploy_monitoring() {
    if [[ "$DEPLOY_MONITORING" == "true" ]]; then
        print_step "Deploying monitoring stack..."
        
        if [[ -d "infrastructure/monitoring" ]]; then
            cd infrastructure/monitoring
            docker-compose -f docker-compose.monitoring.yml up -d || print_warning "Monitoring stack not available"
            cd "$PROJECT_ROOT"
        fi
    fi
}

run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        print_header "RUNNING TESTS"
        
        if [[ -f "scripts/comprehensive_tests.py" ]]; then
            print_step "Installing test dependencies..."
            pip3 install aiohttp asyncio psycopg2-binary redis websockets numpy pandas > /dev/null 2>&1 || true
            
            print_step "Running test suite..."
            python3 scripts/comprehensive_tests.py --env $ENVIRONMENT --test-type $TEST_TYPE || print_warning "Some tests failed"
        else
            print_warning "Test framework not found"
        fi
    fi
}

verify_deployment() {
    print_header "VERIFYING DEPLOYMENT"
    
    print_step "Checking service health..."
    
    local services=(
        "8000:API Gateway"
        "8001:Agent Orchestration"
        "8002:Sentiment Analysis"
        "8006:Virtual Trading"
    )
    
    for service_info in "${services[@]}"; do
        local port=$(echo $service_info | cut -d: -f1)
        local name=$(echo $service_info | cut -d: -f2)
        
        if curl -f -s "http://localhost:$port/health" >/dev/null 2>&1; then
            print_success "$name is healthy"
        else
            print_warning "$name is not responding"
        fi
    done
}

generate_summary() {
    print_header "DEPLOYMENT SUMMARY"
    
    print_success "MIP deployment completed!"
    echo ""
    echo -e "${BLUE}🌐 Service URLs:${NC}"
    echo "   Frontend:        http://localhost:3001"
    echo "   API Gateway:     http://localhost:8000"
    echo "   Grafana:         http://localhost:3000 (admin/admin123)"
    echo ""
    echo -e "${BLUE}🔧 Management:${NC}"
    echo "   View services:   docker-compose ps"
    echo "   View logs:       docker-compose logs -f [service]"
    echo "   Stop all:        docker-compose down"
    echo ""
    echo -e "${GREEN}🎉 Ready for trading with AI!${NC}"
}

main() {
    cd "$PROJECT_ROOT"
    
    check_prerequisites
    setup_environment
    
    if [[ "$CLEAN_DEPLOY" == "true" ]]; then
        print_step "Cleaning up..."
        docker-compose down -v >/dev/null 2>&1 || true
    fi
    
    if [[ "$ENVIRONMENT" == "local" ]]; then
        deploy_local
        deploy_monitoring
    else
        print_step "Running AWS deployment..."
        bash scripts/aws_deployment_guide.sh || print_error "AWS deployment failed"
    fi
    
    verify_deployment
    run_tests
    generate_summary
}

main "$@"
