# 🚀 RunPod Serverless Deployment Guide

## ✨ Perfect! You already have a Serverless endpoint!

Serverless is actually better for music generation because:
- ⚡ **Pay per use** - Only charged during generation
- 🔄 **Auto-scaling** - Handles traffic spikes automatically  
- 💰 **Cost effective** - No idle time charges
- 🚀 **Faster cold starts** with proper optimization

## 🛠 Setup Your Serverless Endpoint

### Step 1: Update Your Endpoint Configuration

In your RunPod dashboard for endpoint `9eb182ubs5j0jf`:

1. **Go to "Releases" tab**
2. **Create New Release**:
   - **Docker Image**: Use our serverless image (see build instructions below)
   - **Handler**: `/opt/runpod_serverless_handler.py`
   - **Container Disk**: 50GB (for model caching)
   - **GPU**: 24GB (you already have this ✅)

### Step 2: Build Serverless Docker Image

You need to build and push a custom Docker image:

```bash
# In your local repository
docker build -f Dockerfile.serverless -t your-username/ace-step-serverless .

# Push to Docker Hub (create account at hub.docker.com)
docker push your-username/ace-step-serverless

# Or use GitHub Container Registry (free)
docker tag your-username/ace-step-serverless ghcr.io/idreesaziz/ace-step-serverless
docker push ghcr.io/idreesaziz/ace-step-serverless
```

### Step 3: Configure the Release

In RunPod dashboard:
- **Docker Image**: `ghcr.io/idreesaziz/ace-step-serverless:latest`
- **Container Disk**: 50GB
- **Environment Variables**:
  ```
  HF_HUB_ENABLE_HF_TRANSFER=1
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```

## 🧪 Test Your Serverless API

### Get Your Credentials

From your RunPod dashboard:
- **Endpoint ID**: `9eb182ubs5j0jf` ✅ (you have this)
- **API Key**: Go to Settings → API Keys → Create new key

### Test with Python

```python
import requests
import base64

# Your credentials
ENDPOINT_ID = "9eb182ubs5j0jf"
API_KEY = "your-api-key-here"

# Generate music
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "prompt": "upbeat electronic dance music",
            "duration": 20.0,
            "seed": 42
        }
    },
    timeout=300
)

result = response.json()
if result["status"] == "COMPLETED":
    # Save the generated audio
    audio_data = base64.b64decode(result["output"]["audio_base64"])
    with open("generated_music.wav", "wb") as f:
        f.write(audio_data)
    print("Music generated successfully!")
```

### Test with cURL

```bash
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "peaceful piano melody",
      "duration": 15.0
    }
  }'
```

## 💰 Serverless Pricing Breakdown

Your endpoint shows: **24GB GPU** at **$0.50** (likely per compute unit)

### Expected Costs:
- **Cold start**: ~30-60s = $0.008-$0.017
- **Generation** (20s audio): ~15-30s = $0.004-$0.008  
- **Per request**: ~$0.012-$0.025 total
- **100 generations**: ~$1.20-$2.50

### Cost Optimization Tips:
1. **Keep workers warm**: Set min workers to 1 if you have regular traffic
2. **Batch requests**: Generate multiple clips in one request
3. **Cache models**: Use persistent storage for faster cold starts

## 🔧 Advanced Configuration

### Scaling Settings (in your dashboard):

```
Active Workers: 0 → 1 (keep 1 warm for faster response)
Max Workers: 3 (good for moderate traffic)
Idle Timeout: 30 seconds (balance cost vs speed)
Scale Strategy: Queue Delay = 4s (current setting is good)
```

### Performance Optimization:

1. **Model Caching**: Models are cached between requests
2. **GPU Memory**: Optimized for 24GB GPU
3. **Cold Start**: ~30-60s for first request, ~5-15s subsequent

## 🚨 Troubleshooting

### Common Issues:

1. **"No workers available"**
   - Check if your Docker image is accessible
   - Verify the handler path is correct
   - Check container disk size (need 50GB minimum)

2. **Long cold starts**
   - Increase container disk size
   - Use model pre-loading in the handler
   - Consider keeping 1 worker active

3. **Out of memory errors**
   - Reduce batch size in the handler
   - Enable gradient checkpointing
   - Use mixed precision training

### Debug Commands:

```bash
# Check if your image works locally
docker run --gpus all -p 8000:8000 your-username/ace-step-serverless

# Test the handler directly
python3 runpod_serverless_handler.py
```

## 📊 Monitoring

Track these metrics in your RunPod dashboard:
- **Request latency**: Target <60s total (including cold start)
- **Success rate**: Should be >95%
- **Worker utilization**: Optimize scaling based on usage patterns
- **Cost per request**: Monitor and optimize

## 🎯 Next Steps

1. **Build the Docker image** using `Dockerfile.serverless`
2. **Create a new release** in your RunPod dashboard
3. **Test with the provided Python client**: `python3 test_serverless_api.py`
4. **Monitor performance** and adjust scaling settings
5. **Integrate into your application** using the API

Your serverless endpoint is perfect for music generation! Much more cost-effective than traditional pods. 🎵
