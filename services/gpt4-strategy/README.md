# GPT-4 Strategy Agent Service

Enterprise-grade options strategy generation service using GPT-4 Turbo with portfolio awareness, rate limiting, cost controls, and comprehensive security.

## 🎯 Overview

The GPT-4 Strategy Agent is a critical component of the MIP multi-agent architecture, responsible for generating sophisticated options strategies with:

- **Portfolio Context Awareness**: Considers existing positions and Greeks
- **Risk Management**: Validates strategies against user profiles and market conditions
- **Cost Controls**: Real-time budget tracking and rate limiting
- **Security**: Prompt injection protection and input validation
- **Fallback Mechanisms**: Graceful degradation during service issues

## 🏗️ Architecture

```
GPT-4 Strategy Agent
├── Rate Limiting Layer (Redis-based)
├── Cost Tracking & Budget Controls
├── Portfolio Context Enrichment
├── Strategy Validation Engine
├── Security & Prompt Protection
├── Fallback Mechanisms
└── Performance Monitoring
```

## 🚀 Features

### Core Capabilities
- **GPT-4 Turbo Integration**: Latest model for sophisticated strategy generation
- **Portfolio-Aware Strategies**: Considers existing positions and Greeks exposure
- **Multi-Tier Rate Limiting**: Different limits for free/premium users
- **Real-Time Cost Tracking**: Budget management and usage analytics
- **Strategy Validation**: Comprehensive risk and compliance checks
- **Batch Processing**: Generate multiple strategies efficiently

### Security & Compliance
- **Prompt Injection Protection**: Advanced security validation
- **Financial Compliance**: FINRA/SEC guideline adherence
- **Input Sanitization**: Prevent harmful or malicious requests
- **Output Validation**: Ensure safe and accurate strategy recommendations

### Performance & Reliability
- **Fallback Strategies**: Multiple fallback layers for reliability
- **Circuit Breakers**: Automatic failover during high error rates
- **Performance Monitoring**: Real-time metrics and alerting
- **Auto-scaling**: Kubernetes HPA for demand management

## 📊 Performance Targets

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Response Time** | <2s (p95) | Prometheus alerts |
| **Error Rate** | <5% | Real-time monitoring |
| **Cost Efficiency** | <$0.50/request | Budget tracking |
| **Availability** | 99.9% | Health checks |

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo-preview
MAX_TOKENS=4000
TEMPERATURE=0.3

# Database & Redis
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_HOST=localhost
REDIS_PORT=6379

# Service Settings
ENABLE_SECURITY_VALIDATION=true
ENABLE_RATE_LIMITING=true
ENABLE_COST_TRACKING=true
REQUESTS_PER_MINUTE=50
COST_PER_HOUR_LIMIT=100.0

# Performance
REQUEST_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=100
```

### Rate Limits by Tier

| Tier | Requests/Hour | Cost/Hour | Features |
|------|---------------|-----------|----------|
| **Free** | 10 | $1.00 | Basic strategies |
| **Basic** | 50 | $10.00 | Advanced strategies |
| **Premium** | 200 | $50.00 | Portfolio analysis |
| **Enterprise** | 1000 | $200.00 | Custom strategies |

## 🔌 API Endpoints

### Generate Strategy
```http
POST /strategy/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "user_id": "user123",
  "user_intent": "I want a conservative income strategy for SPY",
  "market_context": {
    "symbol": "SPY",
    "current_price": 450.0,
    "vix": 18.5,
    "trend": "bullish"
  },
  "portfolio_context": {
    "positions": [...],
    "portfolio_greeks": {...}
  },
  "risk_preferences": {
    "risk_tolerance": "medium",
    "max_loss": 1000
  },
  "max_cost_usd": 0.50
}
```

### Batch Generation
```http
POST /strategy/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "requests": [
    { /* strategy request 1 */ },
    { /* strategy request 2 */ }
  ],
  "batch_priority": "normal"
}
```

### Usage Analytics
```http
GET /user/{user_id}/usage
Authorization: Bearer <token>
```

### Service Metrics
```http
GET /metrics
Authorization: Bearer <token>
```

## 🏃‍♂️ Quick Start

### Development Setup

```bash
# Clone and navigate
cd services/gpt4-strategy

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export DATABASE_URL=your_db_url

# Run service
python main.py
```

### Docker Deployment

```bash
# Build image
docker build -t mip/gpt4-strategy .

# Run with docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Or use Helm
helm install gpt4-strategy ./helm
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_gpt4_engine.py -v
```

### Integration Tests
```bash
pytest tests/test_api.py -v
```

### Load Testing
```bash
# Install dependencies
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8006
```

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8006/health
```

### Metrics Collection
- **Prometheus**: Scrapes `/metrics` endpoint
- **Grafana**: Dashboards for visualization
- **AlertManager**: Automated alerting

### Key Metrics
- `gpt4_strategy_requests_total`
- `gpt4_strategy_response_time_seconds`
- `gpt4_strategy_cost_usd_total`
- `gpt4_strategy_errors_total`

## 🔍 Troubleshooting

### Common Issues

**High Error Rate**
- Check OpenAI API status
- Verify API key validity
- Review request validation logs

**Slow Response Times**
- Monitor OpenAI API latency
- Check database connection pool
- Review resource utilization

**Rate Limit Exceeded**
- Check user tier settings
- Review rate limit configuration
- Monitor Redis connectivity

### Debugging

```bash
# View logs
docker logs gpt4-strategy

# Check service status
kubectl get pods -l app=gpt4-strategy

# Monitor metrics
curl http://localhost:8006/metrics
```

## 🔐 Security

### Input Validation
- Prompt injection detection
- Content filtering
- Length limits

### Output Sanitization
- Response validation
- Information leakage prevention
- Error message sanitization

### Compliance
- FINRA guidelines adherence
- SEC regulation compliance
- Audit trail maintenance

## 🚀 Deployment Guide

### Production Checklist

- [ ] Environment variables configured
- [ ] Secrets properly managed
- [ ] Database migrations applied
- [ ] Redis cluster configured
- [ ] Monitoring stack deployed
- [ ] SSL/TLS certificates installed
- [ ] Rate limiting configured
- [ ] Backup procedures verified

### Scaling Considerations

- **Horizontal Scaling**: Use Kubernetes HPA
- **Vertical Scaling**: Adjust resource limits
- **Database**: Connection pooling and read replicas
- **Cache**: Redis cluster for high availability

## 📚 Integration

### With Other MIP Services

```python
# Agent Orchestrator integration
from mip.orchestrator import AgentRequest

request = AgentRequest(
    agent_type="gpt4_strategy",
    user_id="user123",
    payload=strategy_request
)

response = await orchestrator.route_request(request)
```

### Portfolio Enricher

```python
# Get enriched context
enriched_context = await portfolio_enricher.enrich_context(
    market_context=market_data,
    portfolio_context=user_portfolio,
    user_id=user_id
)
```

## 🔄 Updates & Maintenance

### Version Management
- Semantic versioning (v1.0.0)
- Rolling updates with zero downtime
- Automated testing in CI/CD

### Database Migrations
```sql
-- Add new columns for enhanced tracking
ALTER TABLE gpt4_usage_log ADD COLUMN model_version VARCHAR(50);
ALTER TABLE gpt4_usage_log ADD COLUMN validation_score FLOAT;
```

## 📞 Support

For issues or questions:
- Create GitHub issue
- Check monitoring dashboards
- Review service logs
- Contact MIP team

---

**Built with ❤️ for the MIP Multi-Agent Trading Platform**
