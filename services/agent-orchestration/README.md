# services/agent-orchestration/README.md
# Agent Orchestration Service

The Agent Orchestration Service is the central coordinator for the MIP (Market Intelligence Platform) multi-agent system. It intelligently routes requests to specialized AI agents, manages fallbacks, tracks costs, and provides performance monitoring.

## Features

### 🤖 Multi-Agent Coordination
- **Intelligent Routing**: Automatically routes requests to appropriate agents based on request type and user tier
- **Fallback Management**: Graceful degradation to cheaper models when primary agents fail
- **Circuit Breakers**: Prevents cascading failures with automatic recovery
- **Load Balancing**: Distributes requests across multiple agent instances

### 💰 Cost Management
- **Budget Tracking**: Real-time monitoring of user spending across all agents
- **Tier-Based Limits**: Different budget limits for free, premium, and enterprise users
- **Auto-Downgrade**: Automatically switches to cheaper models when budget limits approached
- **Cost Optimization**: Intelligent suggestions for reducing costs

### 📊 Performance Monitoring
- **Real-Time Metrics**: Prometheus metrics for latency, success rates, and costs
- **Health Checks**: Continuous monitoring of agent availability and performance
- **Audit Logging**: Complete audit trail of all agent interactions for compliance
- **Performance Analytics**: Historical performance tracking and trend analysis

### 🔒 Security & Compliance
- **JWT Authentication**: Secure user authentication with tier-based permissions
- **Input Validation**: Comprehensive validation of all requests
- **Audit Trails**: SHA-256 hashing of inputs/outputs for compliance
- **Rate Limiting**: Protection against abuse and cost overruns

## Architecture

### Agent Types
- **FinBERT Agent**: Financial sentiment analysis (110M parameters, 12ms latency)
- **GPT-4 Turbo Agent**: Options strategy generation (850ms latency, rate limited)
- **Llama 2-7B Agent**: Explanations and insights (210ms latency)
- **TFT Agent**: Temporal Fusion Transformer for price forecasting (28ms latency)
- **Risk Analysis Agent**: Portfolio and strategy risk assessment (50ms latency)

### Request Types
- `news_analysis`: Sentiment analysis + explanations
- `options_recommendation`: Strategy generation + risk analysis
- `price_prediction`: TFT forecasting + explanations
- `portfolio_analysis`: Comprehensive risk and sentiment analysis
- `comprehensive_analysis`: All agents for complete analysis

### User Tiers
- **Free Tier**: Basic models, $5/month limit, 100 requests/day
- **Premium Tier**: Advanced models, $100/month limit, 1000 requests/day
- **Enterprise Tier**: All models, $1000/month limit, unlimited requests

## Quick Start

### Local Development
```bash
# Clone and setup
git clone <repository>
cd services/agent-orchestration

# Install dependencies
pip install -r requirements.txt

# Start dependencies
docker-compose up -d postgres redis

# Run service
python -m uvicorn src.main:app --reload --port 8000
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Example Request
```bash
# Analyze market sentiment
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "news_analysis",
    "data": {
      "text": "Apple stock surged 5% after strong quarterly earnings report"
    }
  }'
```

## API Endpoints

### Core Endpoints
- `POST /analyze` - Execute multi-agent analysis
- `GET /health` - Service health check
- `GET /agents/health` - Individual agent health status
- `POST /agents/preferences` - Update user agent preferences

### Management Endpoints
- `GET /agents/performance/{agent_type}` - Agent performance metrics
- `POST /agents/test` - Test specific agent
- `GET /stats` - Orchestration statistics

### Cost Management
- `GET /cost/breakdown` - User cost breakdown
- `GET /cost/suggestions` - Cost optimization suggestions
- `POST /budget/update` - Update budget settings

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/mip
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_hf_key

# Service URLs
FINBERT_SERVICE_URL=http://finbert-service:8000
LLAMA_SERVICE_URL=http://llama-service:8000
GPT4_SERVICE_URL=http://gpt4-service:8000

# Security
JWT_SECRET_KEY=your_secret_key
CORS_ORIGINS=["http://localhost:3000"]

# Performance
MAX_CONCURRENT_REQUESTS=100
CIRCUIT_BREAKER_THRESHOLD=5
AGENT_TIMEOUT_DEFAULT=30
```

## Monitoring

### Prometheus Metrics
- `orchestrator_requests_total` - Total requests by type and status
- `orchestrator_request_duration_seconds` - Request processing time
- `agent_invocations_total` - Agent usage by type
- `cost_usd_total` - Cost tracking by agent and user tier
- `circuit_breaker_state` - Circuit breaker status
- `system_health_score` - Overall system health

### Grafana Dashboards
Access Grafana at `http://localhost:3000` (when using full MIP stack) to view:
- Agent performance metrics
- Cost breakdown by user tier
- System health overview
- Request volume and latency trends

## Development

### Running Tests
```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests
pytest tests/integration/ -v
```

### Adding New Agents
1. Create agent class inheriting from `BaseAgent`
2. Implement required methods: `initialize()`, `validate_input()`, `_process_internal()`
3. Add agent to `orchestrator.py` configuration
4. Update routing table for new request types
5. Add tests and documentation

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Production Deployment

### Prerequisites
- PostgreSQL 15+
- Redis 7+
- Kubernetes cluster with GPU support (for Llama agent)

### Helm Deployment
```bash
# Install with Helm
helm install agent-orchestration ./helm/ \
  --set image.tag=latest \
  --set database.url="postgresql://..." \
  --set secrets.openaiKey="..."
```

### Scaling Considerations
- **CPU**: 2-4 cores recommended for orchestrator
- **Memory**: 4-8GB depending on concurrent users
- **Database**: Connection pooling recommended (20+ connections)
- **Redis**: Cluster mode for high availability

### Security Checklist
- [ ] JWT secret key rotated regularly
- [ ] Database credentials secured
- [ ] API keys in secrets management
- [ ] Network policies configured
- [ ] HTTPS/TLS enabled
- [ ] Rate limiting configured

## Troubleshooting

### Common Issues

**Agent timeouts:**
```bash
# Check agent health
curl http://localhost:8000/agents/health

# View circuit breaker status
curl http://localhost:8000/stats
```

**High costs:**
```bash
# Check user cost breakdown
curl http://localhost:8000/cost/breakdown

# Get optimization suggestions
curl http://localhost:8000/cost/suggestions
```

**Performance issues:**
```bash
# Monitor metrics
curl http://localhost:9090/metrics

# Check system stats
curl http://localhost:8000/stats
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug
python -m uvicorn src.main:app --reload --log-level debug
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-agent`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit pull request with clear description

## License

This project is part of the MIP platform and follows the main project licensing.
