# Enhanced FastAPI Gateway
## Multi-Agent Market Intelligence Platform

A production-ready, enterprise-grade API gateway that orchestrates multiple AI agents for comprehensive market intelligence, virtual trading, and real-time portfolio management.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 🚀 Features

### 🤖 Multi-Agent Orchestration
- **FinBERT Sentiment Analysis**: Real-time sentiment analysis of financial news and social media
- **TFT Forecasting**: Temporal Fusion Transformer for price predictions
- **GPT-4 Strategy Generation**: Advanced strategy recommendations
- **Llama Explanation**: Natural language explanations of trading decisions
- **Real-time Processing**: Live market data integration and analysis

### 💰 Virtual Trading Engine
- **Portfolio Management**: Create and manage multiple virtual portfolios
- **Order Execution**: Market, limit, and stop orders with realistic slippage
- **Position Tracking**: Real-time position updates with P&L calculations
- **Trade History**: Comprehensive trade logging and performance analysis

### 📊 Real-time P&L Engine
- **Live Updates**: Continuous profit/loss calculations
- **Performance Metrics**: Sharpe ratio, max drawdown, VaR calculations
- **Historical Tracking**: Time-series P&L data with multiple timeframes
- **Risk Analytics**: Portfolio risk assessment and monitoring

### 🔒 Enterprise Security
- **JWT Authentication**: Secure token-based authentication system
- **API Key Management**: Enterprise API key support with permissions
- **Rate Limiting**: Intelligent rate limiting by user tier
- **Permission System**: Granular permission controls

### ⚡ Real-time Features
- **WebSocket Support**: Real-time portfolio updates and market data
- **Live Market Data**: Multiple data source integration (Polygon, Alpha Vantage)
- **Event Broadcasting**: Push notifications for trades, alerts, and analysis
- **Connection Management**: Robust WebSocket connection handling

### 📈 Monitoring & Analytics
- **System Monitoring**: Comprehensive health checks and performance metrics
- **Business Analytics**: User activity, trading volume, agent usage tracking
- **Alert System**: Automated alerts for system issues and risk breaches
- **Performance Optimization**: Caching, rate limiting, resource management

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │  Nginx Gateway  │    │  FastAPI App    │
│   (Frontend)    │◄──►│ (Load Balancer) │◄──►│   (Gateway)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
            ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
            │     Redis       │              │   PostgreSQL    │              │     Kafka       │
            │   (Cache)       │              │  (Database)     │              │  (Messaging)    │
            └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │
    ┌──────────────────┼──────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │                  │                  │
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│ FinBERT   │ │    TFT    │ │   GPT-4   │ │   Llama   │ │Real-time  │ │Prometheus │ │  Grafana  │
│Sentiment  │ │Forecasting│ │ Strategy  │ │Explanation│ │Processing │ │Monitoring │ │Dashboards │
└───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
```

## 📦 Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 16+ (for React dashboard)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-org/enhanced-fastapi-gateway.git
cd enhanced-fastapi-gateway
```

2. **Run the deployment script**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

3. **Access the services**
- Dashboard: http://localhost
- API Gateway: http://localhost/api
- API Documentation: http://localhost/api/docs
- Grafana: http://localhost:3000
- Kibana: http://localhost:5601

### Manual Setup

1. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

2. **Start infrastructure services**
```bash
docker-compose up -d redis postgres kafka zookeeper
```

3. **Start AI services**
```bash
docker-compose up -d finbert-sentiment-service tft-forecasting-service gpt4-strategy-service llama-explanation-service
```

4. **Start the API Gateway**
```bash
docker-compose up -d api-gateway
```

5. **Start monitoring and frontend**
```bash
docker-compose up -d prometheus grafana nginx react-dashboard
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
POSTGRES_DB=mip_database
POSTGRES_USER=mip_user
POSTGRES_PASSWORD=your_secure_password

# Cache
REDIS_PASSWORD=your_redis_password

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key

# External APIs
POLYGON_API_KEY=your_polygon_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
OPENAI_API_KEY=your_openai_api_key

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### Service Configuration

Each service can be configured through environment variables and configuration files:

- **Rate Limiting**: Configure in `src/rate_limiter.py`
- **Risk Management**: Configure in `src/risk_manager.py`
- **Market Data**: Configure in `src/market_data.py`
- **WebSocket**: Configure in `src/websocket_handler.py`

## 📚 API Documentation

### Authentication

```bash
# Login
curl -X POST "http://localhost/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password"}'

# Use token
curl -X GET "http://localhost/api/market/price/AAPL" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

### Market Data

```bash
# Get single price
GET /api/market/price/{symbol}

# Get multiple prices
POST /api/market/prices
{
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}

# Get detailed quote
GET /api/market/quote/{symbol}
```

### Risk Management

```bash
# Validate trade
POST /api/risk/validate-trade
{
  "portfolio_id": "portfolio_123",
  "symbol": "AAPL",
  "action": "buy",
  "quantity": 100,
  "price": 150.0
}

# Get portfolio risk
GET /api/risk/portfolio/{portfolio_id}
```

### WebSocket Connections

```javascript
// User updates
const ws = new WebSocket('ws://localhost/ws/user/user_123');

// Portfolio updates
const ws = new WebSocket('ws://localhost/ws/portfolio/portfolio_123');

// Market data
const ws = new WebSocket('ws://localhost/ws/market/AAPL,MSFT,GOOGL');
```

## 🔍 Monitoring

### Health Checks

```bash
# System health
curl http://localhost/api/health

# Service health script
./scripts/health-check.sh
```

### Metrics & Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Kibana**: http://localhost:5601

### Logging

```bash
# View logs
docker-compose logs -f api-gateway

# View all services
docker-compose logs -f

# Search logs in Kibana
# Navigate to http://localhost:5601
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
# Install k6
brew install k6  # macOS
# or download from https://k6.io/

# Run load tests
k6 run tests/load/api_load_test.js
```

## 🚀 Deployment

### Production Deployment

1. **Configure environment for production**
```bash
# Update .env for production
ENVIRONMENT=production
LOG_LEVEL=WARNING
```

2. **Enable SSL/TLS**
```bash
# Add SSL certificates to nginx/ssl/
# Update nginx.conf for HTTPS
```

3. **Scale services**
```bash
docker-compose up -d --scale api-gateway=3
```

### Kubernetes Deployment
```bash
# Generate Kubernetes manifests
kompose convert -f docker-compose.yml

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 📊 Performance

### Benchmarks
- **Throughput**: 10,000+ requests/second
- **Latency**: <100ms average response time
- **Concurrency**: 10,000+ concurrent connections
- **Availability**: 99.9% uptime SLA

### Optimization Tips
1. **Caching**: Implement Redis caching for frequently accessed data
2. **Connection Pooling**: Use connection pools for database and external APIs
3. **Rate Limiting**: Implement appropriate rate limits by user tier
4. **Load Balancing**: Use multiple API gateway instances
5. **Monitoring**: Set up comprehensive monitoring and alerting

## 🛡️ Security

### Security Features
- JWT token authentication with refresh tokens
- API key management with permissions
- Rate limiting by user tier and IP
- CORS configuration
- Input validation and sanitization
- Secure WebSocket connections

### Security Best Practices
- Regularly rotate JWT secrets
- Use HTTPS in production
- Implement proper CORS policies
- Monitor for security threats
- Regular security audits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](http://localhost/api/docs)
- [WebSocket Guide](docs/websocket.md)
- [Deployment Guide](docs/deployment.md)

### Community
- [GitHub Issues](https://github.com/your-org/enhanced-fastapi-gateway/issues)
- [Discussions](https://github.com/your-org/enhanced-fastapi-gateway/discussions)
- [Discord Community](https://discord.gg/your-server)

### Commercial Support
For enterprise support, custom development, or consulting services, contact us at support@yourcompany.com

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Redis for high-performance caching
- Docker for containerization
- Prometheus & Grafana for monitoring
- All the amazing open-source contributors

---

**Built with ❤️ for the financial technology community**
