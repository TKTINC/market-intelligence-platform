# services/llama-explanation/README.md
# Llama 2-7B Explanation Service

The Llama 2-7B Explanation Service provides GPU-optimized, quantized language model inference for generating financial explanations and insights in the MIP platform.

## Features

### 🚀 High-Performance Inference
- **Quantized Model**: 4-bit quantization (Q4_K_M) for efficient GPU memory usage
- **GPU Optimization**: CUDA-accelerated inference with up to 35 GPU layers
- **Fast Response**: Target 210ms latency for explanation generation
- **Concurrent Processing**: Support for 3+ concurrent requests per GPU

### 📊 Smart Load Balancing
- **Priority Queue**: Urgent/high/normal/low priority request handling
- **Request Batching**: Efficient batch processing for multiple requests
- **Circuit Breaker**: Automatic failover and recovery mechanisms
- **Queue Management**: Configurable queue depth and timeout handling

### 💰 Cost & Performance Optimization
- **Memory Efficient**: 4-bit quantization reduces memory usage by ~4x
- **Token Tracking**: Precise cost calculation based on actual token usage
- **Performance Monitoring**: Real-time latency and throughput metrics
- **Auto-scaling**: KEDA-based scaling based on queue depth and GPU utilization

### 🔒 Production-Ready
- **Health Monitoring**: Comprehensive health checks and metrics
- **GPU Management**: Optimal GPU memory allocation and cleanup
- **Error Handling**: Graceful error handling and request retry logic
- **Security**: Non-root container execution and network policies

## Architecture

### Model Specifications
- **Base Model**: Llama 2-7B Chat (fine-tuned for financial explanations)
- **Quantization**: Q4_K_M GGUF format (~4GB model size)
- **Context Window**: 4096 tokens
- **GPU Layers**: 35 layers on GPU, remainder on CPU
- **Batch Size**: 512 tokens for optimal throughput

### Performance Targets
- **Latency**: 210ms average (95th percentile <500ms)
- **Throughput**: 100+ explanations per minute per GPU
- **Concurrent Requests**: 3 simultaneous generations
- **Queue Capacity**: 100 pending requests
- **Memory Usage**: <12GB GPU memory

### Request Flow
```
Client Request → Load Balancer → Priority Queue → GPU Processing → Response
     ↓              ↓               ↓              ↓           ↓
   Validation → Rate Limiting → Queue Management → Inference → Monitoring
```

## API Endpoints

### Core Endpoints
- `POST /explain` - Generate single explanation
- `POST /explain/batch` - Generate multiple explanations
- `GET /health` - Service health check
- `GET /status` - Detailed service status

### Admin Endpoints
- `POST /admin/warmup` - Manually warm up model
- `POST /admin/reload` - Reload model
- `GET /admin/stats` - Detailed statistics

### Monitoring
- `GET /metrics` - Prometheus metrics

## Quick Start

### Local Development
```bash
# Clone and setup
cd services/llama-explanation

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python[server]

# Run service
python -m uvicorn src.main:app --reload --port 8000
```

### Docker with GPU
```bash
# Build image
docker build -t mip/llama-explanation .

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 9090:9090 \
  -e N_GPU_LAYERS=35 \
  -e MAX_CONCURRENT_REQUESTS=3 \
  mip/llama-explanation
```

### Example Request
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "analysis_type": "sentiment",
      "symbol": "AAPL",
      "sentiment_score": 0.75,
      "current_price": 150.0,
      "iv_rank": 65
    },
    "max_tokens": 200,
    "temperature": 0.1,
    "priority": "normal"
  }'
```

### Example Response
```json
{
  "explanation": "The positive sentiment analysis for AAPL reflects strong market confidence driven by robust institutional support and favorable analyst coverage. Key factors include solid earnings expectations, sector leadership position, and technical momentum indicators showing continued strength...",
  "tokens_used": 87,
  "processing_time_ms": 195,
  "model_info": {
    "model_name": "llama-2-7b-explanations",
    "quantization": "Q4_K_M",
    "gpu_layers": 35,
    "context_size": 4096
  },
  "confidence_score": 0.89
}
```

## Configuration

### Environment Variables
```bash
# Model Configuration
MODEL_PATH=/models/llama-2-7b-explanations.Q4_K_M.gguf
N_CTX=4096                    # Context window size
N_GPU_LAYERS=35               # GPU layers
N_BATCH=512                   # Batch size
N_THREADS=8                   # CPU threads

# Performance
MAX_CONCURRENT_REQUESTS=3     # Concurrent requests
MAX_QUEUE_SIZE=100           # Queue capacity
REQUEST_TIMEOUT=60           # Request timeout (seconds)

# GPU Settings
CUDA_VISIBLE_DEVICES=0       # GPU device
GPU_MEMORY_FRACTION=0.9      # GPU memory usage limit

# Monitoring
PROMETHEUS_ENABLED=true      # Enable metrics
METRICS_PORT=9090           # Metrics port
```

### Model Quantization
```bash
# Download base model
wget https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# Quantize to Q4_K_M format
python -m llama_cpp.quantize \
  ./Llama-2-7b-chat-hf/pytorch_model.bin \
  ./llama-2-7b-explanations.Q4_K_M.gguf \
  Q4_K_M
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster with GPU support
- NVIDIA device plugin installed
- GPU nodes with 12GB+ memory

### Deploy with Helm
```bash
# Install Llama service
helm install llama-explanation ./helm/ \
  --set image.tag=latest \
  --set resources.requests."nvidia\.com/gpu"=1 \
  --set model.gpuLayers=35

# Check deployment
kubectl get pods -l app=llama-explanation
kubectl logs -l app=llama-explanation
```

### Scaling Configuration
```yaml
# Auto-scaling based on queue depth
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  customMetrics:
    - type: Pods
      pods:
        metric:
          name: llama_queue_depth
        target:
          type: AverageValue
          averageValue: "5"
```

## Monitoring & Observability

### Prometheus Metrics
- `llama_requests_total` - Total requests by priority and status
- `llama_processing_duration_seconds` - Processing time histogram
- `llama_tokens_generated_total` - Total tokens generated
- `llama_queue_depth` - Current queue depth
- `llama_gpu_memory_usage_bytes` - GPU memory usage
- `llama_tokens_per_second` - Generation speed

### Health Checks
```bash
# Basic health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status

# Performance metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboard
Key metrics to monitor:
- Request rate and latency
- Queue depth and processing time
- GPU memory and utilization
- Error rates and timeouts
- Token generation speed

## Performance Tuning

### GPU Optimization
```python
# Optimal settings for V100/A100
N_GPU_LAYERS = 35        # All layers on GPU
N_CTX = 4096            # Full context window
N_BATCH = 512           # Optimal batch size
GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory
```

### CPU Optimization
```python
N_THREADS = 8           # CPU threads (adjust based on CPU cores)
USE_MLOCK = True        # Lock model in memory
USE_MMAP = True         # Memory-map model file
F16_KV = True          # Use float16 for key/value cache
```

### Load Balancing
```python
MAX_CONCURRENT_REQUESTS = 3  # Concurrent generations
MAX_QUEUE_SIZE = 100         # Queue capacity
REQUEST_TIMEOUT = 60         # Request timeout
```

## Development

### Running Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v --benchmark-only
```

### Adding Custom Prompts
```python
# Customize system prompt for specific use cases
custom_prompt = """
You are a financial AI assistant specialized in options trading.
Focus on:
- Options strategy explanations
- Risk assessment
- Greeks analysis
- Market conditions
"""

result = await llama_engine.generate_explanation(
    context=context,
    system_prompt=custom_prompt
)
```

### Model Fine-tuning
```bash
# Fine-tune base model on financial data
python scripts/fine_tune.py \
  --base_model llama-2-7b-chat \
  --dataset financial_explanations.jsonl \
  --output_dir ./fine_tuned_model
```

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
# Reduce GPU layers
export N_GPU_LAYERS=20

# Reduce context size
export N_CTX=2048

# Monitor GPU memory
nvidia-smi
```

**Slow Performance:**
```bash
# Check GPU utilization
nvidia-smi

# Monitor queue depth
curl http://localhost:8000/status

# Check processing metrics
curl http://localhost:9090/metrics | grep llama_processing
```

**High Queue Depth:**
```bash
# Scale up replicas
kubectl scale deployment llama-explanation-service --replicas=5

# Check resource limits
kubectl describe pod llama-explanation-xxx

# Monitor KEDA scaling
kubectl get scaledobject
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m uvicorn src.main:app --log-level debug
```

## Production Checklist

### Deployment
- [ ] GPU nodes available and labeled
- [ ] Model downloaded and quantized
- [ ] Resource limits configured
- [ ] Health checks enabled
- [ ] Monitoring configured

### Performance
- [ ] Load testing completed
- [ ] Autoscaling tested
- [ ] Memory usage optimized
- [ ] Latency targets met

### Security
- [ ] Non-root user configured
- [ ] Network policies applied
- [ ] Resource quotas set
- [ ] Image security scanned

### Monitoring
- [ ] Prometheus metrics enabled
- [ ] Grafana dashboards configured
- [ ] Alerting rules set up
- [ ] Log aggregation configured

## License

This service is part of the MIP platform and follows the main project licensing.
