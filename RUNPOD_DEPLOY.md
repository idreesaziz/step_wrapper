# ACE-Step Music Generator - RunPod Deployment Guide

## 🚀 Quick Deploy to RunPod

### Option 1: Pre-built Docker Image (Recommended)
I'll create a public Docker image you can use directly:

```bash
# Build and push the image (I'll do this for you)
docker build -t your-username/ace-step-runpod .
docker push your-username/ace-step-runpod
```

### Option 2: Deploy from GitHub

1. **Create RunPod Account**
   - Go to [runpod.io](https://runpod.io)
   - Sign up and add some credits ($10-20 recommended)

2. **Deploy Pod**
   - Click "Deploy" → "New Pod"
   - Choose GPU: RTX 4090 or A100 (recommended)
   - Template: "Custom Docker Image"
   - Docker Image: `nvidia/cuda:12.6-runtime-ubuntu22.04`
   - Container Disk: 50GB (for models)
   - Expose Ports: `8000/http`

3. **Setup Commands**
   Run these in the RunPod terminal:
   ```bash
   # Update system and install dependencies
   apt-get update && apt-get install -y python3-pip git curl wget ffmpeg
   
   # Clone your repository
   git clone https://github.com/YOUR_USERNAME/your-repo.git
   cd your-repo
   
   # Install Python dependencies
   pip3 install -r requirements.txt
   
   # Clone ACE-Step
   git clone https://github.com/ace-step/ACE-Step.git
   cd ACE-Step && pip3 install -e . && cd ..
   
   # Copy files and start
   cp runpod_setup.py ACE-Step/
   cd ACE-Step && python3 runpod_setup.py
   ```

## 📋 RunPod Configuration

### Recommended GPU Options:
- **RTX 4090**: ~$0.34/hr, 24GB VRAM (good for testing)
- **A100 40GB**: ~$1.89/hr, 40GB VRAM (production)
- **H100**: ~$4.29/hr, 80GB VRAM (fastest)

### Container Settings:
- **Container Disk**: 50GB minimum (models are ~3.5GB)
- **Volume**: Optional persistent storage
- **Environment Variables**:
  ```
  PORT=8000
  HF_HUB_ENABLE_HF_TRANSFER=1
  ```

## 🔌 API Endpoints

Once deployed, your API will be available at: `https://your-pod-id-8000.proxy.runpod.net`

### Endpoints:

#### 1. Health Check
```bash
GET /health
```

#### 2. Generate Music (RunPod Format)
```bash
POST /runsync
Content-Type: application/json

{
  "input": {
    "prompt": "upbeat electronic dance music",
    "duration": 30.0,
    "seed": 42,
    "guidance_scale": 3.0,
    "num_inference_steps": 50
  }
}
```

#### 3. Generate Music (Direct)
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "peaceful piano melody",
  "duration": 15.0,
  "seed": 123
}
```

## 💡 Usage Examples

### Python Client
```python
import requests
import base64

# Your RunPod endpoint
API_URL = "https://your-pod-id-8000.proxy.runpod.net"

# Generate music
response = requests.post(f"{API_URL}/runsync", json={
    "input": {
        "prompt": "energetic rock guitar solo",
        "duration": 20.0,
        "seed": 42
    }
})

result = response.json()
if result["status"] == "success":
    # Decode base64 audio
    audio_data = base64.b64decode(result["output"]["audio_base64"])
    
    # Save as WAV file
    with open("generated_music.wav", "wb") as f:
        f.write(audio_data)
    
    print(f"Generated in {result['output']['generation_time']:.1f}s")
else:
    print(f"Error: {result['output']['error']}")
```

### cURL Example
```bash
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/runsync" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "ambient forest sounds with gentle piano",
      "duration": 25.0
    }
  }' | jq '.output.audio_base64' | base64 -d > music.wav
```

## 📊 Performance & Costs

### Expected Performance:
- **First generation**: 60-90s (model loading + generation)
- **Subsequent generations**: 10-30s (30s audio)
- **Memory usage**: ~8GB VRAM

### Cost Estimates (RTX 4090):
- **Setup time**: ~5 mins = $0.03
- **Per generation**: ~30s = $0.003
- **1000 generations**: ~$3.00

## 🛠 Troubleshooting

### Common Issues:

1. **Model Download Fails**
   ```bash
   # Check internet and try manual download
   huggingface-cli download ACE-Step/ACE-Step-v1-3.5B
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size or use smaller GPU settings
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Slow Generation**
   ```bash
   # Verify GPU usage
   nvidia-smi
   ```

## 🔄 Scaling Options

### Auto-scaling Setup:
1. Use RunPod Serverless for automatic scaling
2. Set min/max instances based on demand
3. Configure cold start optimization

### Load Balancing:
1. Deploy multiple pods
2. Use RunPod's built-in load balancer
3. Implement request queuing for high demand

## 📈 Monitoring

### Key Metrics to Track:
- Generation time per request
- GPU utilization
- Memory usage
- Request queue length
- Error rates

### Logging:
All logs are available in RunPod's console and can be exported for analysis.

---

## Next Steps

1. **Test locally first**: `docker build . && docker run -p 8000:8000 your-image`
2. **Deploy to RunPod**: Follow the setup guide above
3. **Scale as needed**: Add more pods or switch to serverless

Need help? Check the RunPod documentation or ask for assistance!
