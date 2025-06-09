# 🚀 Market Intelligence Platform (MIP)

## 🎯 Quick Start

### One-Click Local Deployment

```bash
# 1. Clone and setup
git clone <your-repo>
cd market-intelligence-platform
chmod +x scripts/*.sh

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Deploy everything
./scripts/automation_scripts.sh --environment local

# 4. Access your platform
# Frontend: http://localhost:3001
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin123)
```

### Test Your Deployment

```bash
# Quick health check
python3 scripts/comprehensive_tests.py --env local --test-type sanity

# Full test suite
python3 scripts/comprehensive_tests.py --env local --test-type all
```

## 🏗️ Architecture

**Multi-Agent AI System:**
- 🧠 FinBERT (Sentiment Analysis)
- 💡 GPT-4 (Options Strategies)
- 📝 Llama 2-7B (Explanations)
- 📊 TFT (Price Forecasting)
- ⚖️ Risk Validation

**Core Services:**
- 🌐 API Gateway (FastAPI + WebSocket)
- 🤖 Agent Orchestration
- 💰 Virtual Trading Engine
- 📊 Real-time Monitoring
- 🔒 Enterprise Security

## 📋 Prerequisites

- Docker Desktop (8GB+ RAM)
- Python 3.11+
- Node.js 18+ (for frontend)
- API Keys: OpenAI, Anthropic, Alpha Vantage

## 🔧 Management Commands

```bash
# View services
docker-compose ps

# View logs
docker-compose logs -f api-gateway

# Restart service
docker-compose restart virtual-trading

# Stop all
docker-compose down

# Run tests
python3 scripts/comprehensive_tests.py --env local --test-type sanity
```

## 🌐 Service URLs (Local)

- **Frontend**: http://localhost:3001
- **API Gateway**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

## 🚨 Troubleshooting

**Services not starting?**
```bash
docker-compose logs [service-name]
docker-compose restart [service-name]
```

**Need to reset everything?**
```bash
docker-compose down -v
./scripts/automation_scripts.sh --environment local --clean
```

## 🎉 What's Included

✅ **Multi-Agent AI Trading System**  
✅ **Real-time Virtual Trading**  
✅ **Production Monitoring Stack**  
✅ **Enterprise Security**  
✅ **One-Click Deployment**  
✅ **Comprehensive Testing**  
✅ **CI/CD Pipeline**  

**Happy Trading with AI! 🎯📈🤖**
