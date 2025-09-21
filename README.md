# 🎵 ACE-Step Music Generator API

A production-ready REST API wrapper around the ACE-Step music generation model, optimized for RunPod deployment. Generate high-quality music from text prompts with GPU acceleration.

## ⚡ RunPod Deployment (Recommended)

Deploy to RunPod in minutes with GPU acceleration:

1. **Fork this repository** on GitHub
2. **Deploy on RunPod**:
   - Go to [runpod.io](https://runpod.io)
   - Create new pod with RTX 4090 or A100
   - Use template: Custom Docker Image
   - Image: `nvidia/cuda:12.6-runtime-ubuntu22.04`
   - Expose port: `8000/http`
   - Container disk: 50GB

3. **Setup in RunPod terminal**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ace-step-api.git
   cd ace-step-api
   chmod +x build_runpod.sh
   ./build_runpod.sh
   ```

4. **Start the API**:
   ```bash
   cd ACE-Step && python3 ../runpod_setup.py
   ```

Your API will be live at: `https://your-pod-8000.proxy.runpod.net`

**💰 Cost**: ~$0.34/hour on RTX 4090 (~$0.003 per generation)

## 🚀 Quick Start (Local)

### Option 1: Use the Existing API (Recommended)

ACE-Step already includes a ready-to-use API in `trainer-api.py`:

```bash
# 1. Clone and setup
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step
pip install -e .

# 2. Run the API
python trainer-api.py
```

### Option 2: Use Our Simplified API

```bash
# 1. Run the setup script
./setup_ace_step.sh

# 2. Activate environment and run
source ace_step_env/bin/activate
cd ACE-Step
python simple_ace_api.py
```

### Option 3: Manual Setup

```bash
# Clone repository
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install fastapi uvicorn loguru

# Run API
python trainer-api.py
```

## 📡 API Usage

### Endpoints

- `POST /generate` - Generate music from text prompt
- `GET /health` - Check API health

### Generate Music

**Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "upbeat electronic dance music, 128 bpm, energetic",
       "duration": 60,
       "infer_steps": 27
     }'
```

**Response:**
```json
{
  "audio_path": "generated_audio/generated_20241221_143022_12345.wav",
  "prompt": "upbeat electronic dance music, 128 bpm, energetic",
  "seed": 12345,
  "sample_rate": 48000
}
```

### Python Client

```python
import requests

# Generate music
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "calm acoustic guitar melody, peaceful",
    "duration": 30
})

result = response.json()
print(f"Generated: {result['audio_path']}")
```

## 🎛️ Parameters

### trainer-api.py (Simple API)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the music |
| `duration` | int | 240 | Duration in seconds (10-240) |
| `infer_steps` | int | 60 | Quality vs speed (lower=faster) |
| `guidance_scale` | float | 15.0 | How closely to follow prompt |
| `omega_scale` | float | 10.0 | Additional guidance |
| `seed` | int | random | Seed for reproducible results |

### simple_ace_api.py (Our Version)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the music |
| `duration` | int | 60 | Duration in seconds (10-240) |
| `seed` | int | random | Seed for reproducible results |

## 🎵 Prompt Examples

### Electronic/Dance
```
"upbeat electronic dance music, 128 bpm, synthesizers, energetic, bass drop"
"ambient electronic, atmospheric, slow tempo, ethereal pads, relaxing"
"techno, driving beat, 140 bpm, industrial sounds, dark atmosphere"
```

### Acoustic/Folk
```
"acoustic guitar fingerpicking, folk style, peaceful, campfire vibes"
"classical guitar solo, Spanish style, passionate, moderate tempo"
"banjo and harmonica, bluegrass, lively, traditional American folk"
```

### Orchestral/Cinematic
```
"epic orchestral, cinematic, powerful strings, brass fanfare, heroic"
"romantic violin and piano, classical, emotional, slow tempo"
"dark orchestral, horror movie soundtrack, tension, dissonant"
```

### Jazz/Blues
```
"smooth jazz piano, improvisation, relaxed tempo, sophisticated"
"blues guitar, soulful, emotional, 12-bar progression, minor key"
"big band jazz, swing rhythm, brass section, upbeat, vintage"
```

### Rock/Metal
```
"heavy metal, distorted guitars, powerful drums, aggressive, fast tempo"
"classic rock, electric guitar solo, driving rhythm, anthemic"
"punk rock, fast tempo, raw energy, rebellious, garage band sound"
```

## ⚡ Performance

### Hardware Requirements
- **Minimum**: 8GB VRAM (with optimizations)
- **Recommended**: 16GB+ VRAM
- **CPU**: Any modern CPU (GPU acceleration recommended)

### Generation Speed
- **RTX 4090**: ~1.7s for 1 minute of music
- **RTX 3090**: ~4.7s for 1 minute of music  
- **A100**: ~2.2s for 1 minute of music

### Memory Optimization
```python
# For low VRAM systems
acestep --cpu_offload true --overlapped_decode true
```

## 🛠️ Testing

Use the included test client:

```bash
python test_api.py
```

This provides an interactive interface to test different prompts and see generation results.

## 📁 File Structure

```
music_generator/
├── setup_ace_step.sh          # Automated setup script
├── simple_ace_api.py          # Our simplified API
├── test_api.py               # Test client
├── setup_ace_step.md         # Detailed setup guide
└── ACE-Step/                 # Cloned repository
    ├── trainer-api.py        # Official simple API ⭐
    ├── infer-api.py          # Full-featured API
    └── acestep/              # Main package
```

## 🔧 Configuration

### Environment Variables
```bash
export CHECKPOINT_DIR="./checkpoints"  # Model storage location
export ACE_OUTPUT_DIR="./outputs"      # Output directory
```

### CUDA Setup
For NVIDIA GPUs, ensure CUDA is properly installed:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Use CPU offloading
   export ACE_CPU_OFFLOAD=true
   ```

2. **Models Not Downloading**
   ```bash
   # Manual download
   huggingface-cli download ACE-Step/ACE-Step-v1-3.5B --local-dir ./checkpoints
   ```

3. **Import Errors**
   ```bash
   # Reinstall in development mode
   pip install -e .
   ```

4. **Slow Generation**
   ```bash
   # Reduce inference steps
   # Use: infer_steps=27 instead of 60
   ```

### Docker Setup

```dockerfile
FROM nvidia/cuda:12.6-runtime-ubuntu22.04
# ... (use the provided Dockerfile in ACE-Step repo)
```

## 📚 Additional Resources

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [Technical Paper](https://arxiv.org/abs/2506.00045)
- [Hugging Face Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)
- [Demo Space](https://huggingface.co/spaces/ACE-Step/ACE-Step)

## 📄 License

Apache 2.0 - See [LICENSE](https://github.com/ace-step/ACE-Step/blob/main/LICENSE)
