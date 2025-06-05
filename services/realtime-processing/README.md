# Enhanced Real-time Processing Agent

Enterprise-grade real-time processing engine that coordinates all MIP agents, handles streaming data, manages agent communication, and provides unified multi-agent responses with intelligent caching and load balancing.

## Features

### 🔄 Multi-Agent Coordination
- **Intelligent Agent Orchestration**: Coordinates FinBERT, Llama, GPT-4, and TFT agents
- **Dynamic Routing**: Smart request routing based on load, performance, and requirements
- **Circuit Breakers**: Automatic failover and recovery for unhealthy agents
- **Load Balancing**: Distributes requests optimally across available agents

### 🚀 Real-time Processing
- **Streaming Intelligence**: Live market data and news ingestion
- **WebSocket Support**: Real-time client connections with pub/sub
- **Event Processing**: High-throughput event handling and routing
- **Data Pipelines**: Continuous processing of market feeds

### 💾 Intelligent Caching
- **Multi-Level Cache**: L1 (local) + L2 (Redis) caching strategy
- **Smart Compression**: Automatic compression for large cache entries
- **TTL Management**: Intelligent cache expiration and cleanup
- **Cache Warming**: Proactive caching of frequently accessed data

### 📊 Performance Monitoring
- **Real-time Metrics**: System, agent, and request performance tracking
- **Health Monitoring**: Continuous health checks and alerting
- **Performance Analytics**: Detailed performance trends and insights
- **Custom Dashboards**: Comprehensive monitoring and alerting

## Architecture

```
Enhanced Real-time Processing Engine
├── Multi-Agent Coordinator      # Agent orchestration and communication
├── Stream Processing Pipeline   # Real-time data processing
├── Intelligent Request Router   # Smart routing and load balancing
├── Response Aggregator         # Multi-agent response combination
├── Real-time Data Ingestion    # Market data and news streams
├── Distributed Caching Layer   # Redis-based intelligent caching
├── Load Balancer              # Circuit breakers and health monitoring
├── WebSocket Manager          # Real-time client connections
└── Performance Monitoring     # Metrics collection and alerting
```

## API Endpoints

### Core Intelligence
- `POST /intelligence/unified` - Get unified multi-agent analysis
- `POST /intelligence/stream/start` - Start streaming intelligence
- `DELETE /intelligence/stream/{stream_id}` - Stop streaming

### WebSocket Endpoints
- `WS /ws/intelligence/{stream_id}` - Real-time intelligence streaming
- `WS /ws/market/{symbols}` - Real-time market data streaming

### Management
- `GET /health` - Comprehensive health check
- `GET /agents/status` - Agent status and metrics
- `POST /agents/{agent_name}/restart` - Restart specific agent
- `GET /cache/status` - Cache performance statistics
- `GET /metrics/performance` - Detailed performance metrics

### Alerts & Subscriptions
- `POST /alerts/subscribe` - Subscribe to real-time alerts
- `DELETE /alerts/unsubscribe/{subscription_id}` - Unsubscribe from alerts

## Quick Start

### Docker Compose (Development)
```bash
# Clone repository
git clone <repository-url>
cd services/realtime-processing

# Start services
docker-compose up -d

# Check health
curl http://localhost:8008/health
```

### Kubernetes (Production)
```bash
# Deploy using script
./scripts/deploy.sh latest production

# Or using Helm directly
helm install realtime-processing ./helm \
    --namespace mip \
    --set image.tag=latest
```

## Configuration

### Environment Variables
```bash
REDIS_URL=redis://redis:6379
LOG_LEVEL=INFO
ENVIRONMENT=production
MAX_WORKERS=1
```

### Agent Configuration
```python
# Agent endpoints and timeouts
agents = {
    "sentiment": {
        "url": "http://finbert-sentiment-service:8005",
        "timeout": 10,
        "max_concurrent": 20
    },
    "llama_explanation": {
        "url": "http://llama-explanation-service:8006", 
        "timeout": 30,
        "max_concurrent": 5
    },
    "gpt4_strategy": {
        "url": "http://gpt4-strategy-service:8007",
        "timeout": 20,
        "max_concurrent": 8
    },
    "tft_forecasting": {
        "url": "http://tft-forecasting-service:8008",
        "timeout": 15,
        "max_concurrent": 12
    }
}
```

## Usage Examples

### Unified Intelligence Request
```python
import requests

response = requests.post("http://localhost:8008/intelligence/unified", json={
    "user_id": "user123",
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "analysis_type": "comprehensive",
    "time_horizon": "intraday",
    "include_explanations": True,
    "include_strategies": True,
    "include_forecasts": True,
    "real_time_data": True
})

result = response.json()
print(f"Analysis confidence: {result['confidence_score']}")
print(f"Agents used: {result['agents_used']}")
```

### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:8008/ws/intelligence/stream123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'intelligence_update') {
        console.log('New intelligence:', data.data);
    } else if (data.type === 'alert') {
        console.log('Alert:', data.data);
    }
};

// Subscribe to symbols
ws.send(JSON.stringify({
    type: 'subscribe',
    subscription_type: 'market_data',
    symbols: ['AAPL', 'MSFT']
}));
```

### Performance Monitoring
```python
import requests

# Get current performance stats
stats = requests.get("http://localhost:8008/metrics/performance").json()

print(f"System CPU: {stats['current_stats']['system']['cpu_percent']}%")
print(f"Cache hit ratio: {stats['current_stats']['performance']['cache_hit_ratio']}")
print(f"Average response time: {stats['current_stats']['performance']['avg_response_time_ms']}ms")

# Check agent health
agents = requests.get("http://localhost:8008/agents/status").json()
for agent_name, status in agents['agents'].items():
    print(f"{agent_name}: {status['health_score']:.2f} health, {status['utilization']:.1%} load")
```

## Monitoring & Alerts

### Health Checks
The service provides comprehensive health monitoring:
- System resource utilization (CPU, memory, disk)
- Agent availability and response times
- Cache performance and hit ratios
- WebSocket connection health
- Background task status

### Performance Metrics
- **Request Metrics**: Response time, success rate, throughput
- **Agent Metrics**: Individual agent performance and health
- **System Metrics**: Resource usage and capacity
- **Cache Metrics**: Hit ratios, memory usage, evictions

### Alerting
Automatic alerts for:
- High system resource usage
- Agent failures or degraded performance
- Slow response times
- Low cache hit ratios
- WebSocket connection issues

## Scaling & Performance

### Horizontal Scaling
- **Kubernetes HPA**: Auto-scaling based on CPU/memory
- **Load Distribution**: Intelligent request distribution
- **Stateless Design**: No shared state between instances

### Performance Optimization
- **Connection Pooling**: Reused HTTP connections to agents
- **Response Caching**: Multi-level intelligent caching
- **Async Processing**: Non-blocking I/O operations
- **Resource Management**: Configurable limits and timeouts

### Capacity Planning
- **Agent Concurrency**: Configurable per-agent limits
- **Memory Management**: Automatic cache cleanup
- **Rate Limiting**: Client-side rate limiting
- **Circuit Breakers**: Automatic failure isolation

## Security

### Authentication
- **JWT Tokens**: Bearer token authentication
- **User Tiers**: Role-based access control
- **API Keys**: Service-to-service authentication

### Data Protection
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: DOS protection
- **CORS**: Configurable cross-origin policies
- **TLS**: Encrypted connections in production

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Run service
python main.py
```

### Testing
```bash
# Run unit tests
pytest tests/

# Load testing
locust -f tests/load_test.py --host=http://localhost:8008
```

### Contributing
1. Fork repository
2. Create feature branch
3. Write tests
4. Submit pull request

## Troubleshooting

### Common Issues

**High Memory Usage**
```bash
# Check cache statistics
curl http://localhost:8008/cache/status

# Clear cache if needed
curl -X DELETE http://localhost:8008/cache/clear
```

**Agent Connectivity Issues**
```bash
# Check agent status
curl http://localhost:8008/agents/status

# Restart specific agent
curl -X POST http://localhost:8008/agents/sentiment/restart
```

**Performance Degradation**
```bash
# Check performance metrics
curl http://localhost:8008/metrics/performance

# Monitor resource usage
kubectl top pods -n mip
```

### Logs
```bash
# Application logs
kubectl logs -f deployment/realtime-processing -n mip

# Agent communication logs
kubectl logs -f deployment/realtime-processing -n mip | grep "agent"

# Performance logs
kubectl logs -f deployment/realtime-processing -n mip | grep "metrics"
```

## License

Copyright (c) 2024 MIP Team. All rights reserved.
