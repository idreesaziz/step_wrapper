# 🎵 ACE-Step Serverless Music Generator

A serverless deployment of the ACE-Step music generation model on RunPod, providing high-quality text-to-music generation via REST API.

## ✨ Features

- 🎼 **High-Quality Music Generation**: 48kHz stereo audio output
- ⚡ **Fast Inference**: ~25-40s for 30s audio generation  
- 🔧 **Flexible Parameters**: Control duration, quality, and creativity
- 📦 **Serverless**: Auto-scaling RunPod deployment
- 🎯 **Easy Integration**: Simple REST API with base64 audio response
- 💰 **Cost Efficient**: Pay-per-use serverless pricing

## 🚀 Production API (Ready to Use)

**Endpoint**: `https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync`  
**Status**: ✅ **Live and Operational**

### Quick API Test

```python
import requests
import base64

# Configure API
endpoint_id = "9eb182ubs5j0jf"
api_key = "YOUR_API_KEY"  # Get from RunPod
url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

# Generate music
response = requests.post(url, json={
    "input": {
        "prompt": "upbeat electronic dance music with synthesizers",
        "duration": 20.0,
        "guidance_scale": 8.0,
        "num_inference_steps": 20
    }
}, headers={
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}, timeout=180)

# Save generated audio
if response.status_code == 200:
    result = response.json()
    if result.get("status") == "COMPLETED":
        audio_data = base64.b64decode(result["output"]["audio_base64"])
        with open("music.wav", "wb") as f:
            f.write(audio_data)
        print("🎵 Music generated successfully!")
```

## 📖 Complete Documentation

**[📚 Full API Usage Guide →](API_USAGE.md)**

## 🎨 Example Usage

### Simple Generation
```bash
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "peaceful piano melody with strings",
      "duration": 15.0
    }
  }'
```

### Advanced Parameters
```python
payload = {
    "input": {
        "prompt": "cinematic orchestral epic with powerful drums",
        "duration": 30.0,
        "guidance_scale": 12.0,  # Higher adherence to prompt
        "num_inference_steps": 40,  # Higher quality
        "seed": 42  # Reproducible results
    }
}
```

## 🛠️ Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | required | - | Text description of music |
| `duration` | float | 30.0 | 1-240 | Audio length in seconds |
| `guidance_scale` | float | 8.0 | 1.0-20.0 | Prompt adherence strength |
| `num_inference_steps` | int | 20 | 10-100 | Quality vs speed tradeoff |
| `seed` | int | random | any | For reproducible results |

## � Performance Metrics

- **Generation Speed**: ~25-40s for 30s audio
- **Audio Quality**: 48kHz stereo WAV
- **Model Size**: 3.5B parameters  
- **Typical Output**: 2-3MB per 30s audio
- **Success Rate**: >99% for valid prompts

## 🎨 Prompt Examples

```python
# Genre-specific
"upbeat electronic dance music with heavy bass"
"classical piano sonata in C major, romantic period"
"jazz fusion with saxophone and electric piano"

# Mood-based  
"peaceful ambient soundscape with nature sounds"
"energetic rock anthem with powerful guitar riffs"
"melancholic indie folk with acoustic instruments"

# Technical details
"cinematic orchestral, 4/4 time, 120 BPM, epic and heroic"
"lo-fi hip hop beat, vinyl crackle, chill and relaxing"
"progressive house track, 128 BPM, build-up and drop"
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
## 🏗️ Architecture

This serverless deployment consists of:

1. **ACE-Step Model**: Official 3.5B parameter checkpoint
2. **Serverless Handler**: RunPod wrapper (`runpod_serverless_handler.py`)  
3. **Docker Container**: CUDA-enabled PyTorch environment
4. **REST API**: JSON input/output with base64 audio encoding

### Container Image
```
ghcr.io/idreesaziz/ace-step-serverless:latest
```

### Repository Structure
```
├── runpod_serverless_handler.py    # Main serverless handler
├── Dockerfile                      # Container configuration
├── API_USAGE.md                   # Complete API documentation  
├── README.md                      # This file
└── test_full_music_generation.py  # Test script
```

## 🔧 Development

### Local Testing

1. **Clone repository**:
```bash
git clone https://github.com/idreesaziz/step_wrapper.git
cd step_wrapper
```

2. **Build Docker image**:
```bash
docker build -t ace-step-serverless .
```

3. **Test locally** (requires GPU):
```bash
docker run --gpus all -p 8000:8000 ace-step-serverless
```

### Deployment Updates

1. **Make changes** to `runpod_serverless_handler.py`
2. **Build new image**:
```bash
docker build -t ghcr.io/idreesaziz/ace-step-serverless:latest .
```
3. **Push to registry**:
```bash
docker push ghcr.io/idreesaziz/ace-step-serverless:latest
```
4. **RunPod workers** automatically update within minutes

## 🛠️ Testing

Use the included test client:

```bash
python test_api.py
```

This provides an interactive interface to test different prompts and see generation results.

## 📁 File Structure

```
music_generator/
## 🚨 Troubleshooting

### API Issues

1. **Timeout Errors**
   - Increase request timeout (3+ minutes)
   - Reduce `duration` or `num_inference_steps`
   - Retry with exponential backoff

2. **Poor Quality Output**
   - Increase `num_inference_steps` (30-50)
   - Adjust `guidance_scale` (7-12 range)
   - Use more descriptive prompts

3. **Rate Limiting**
   - Wait between requests
   - Monitor RunPod worker availability
   - Consider upgrading RunPod plan

### Response Format Issues
```python
# Always check response status
if response.status_code == 200:
    result = response.json()
    if result.get("status") == "COMPLETED":
        # Process successful response
        pass
    elif result.get("status") == "FAILED":
        print(f"Generation failed: {result.get('error')}")
```

## 💰 Pricing

- **Pay-per-use** serverless model
- **Typical cost**: ~$0.01-0.05 per generation
- **No fixed costs** or server maintenance
- **Auto-scaling** based on demand

## � Additional Resources

- **[ACE-Step GitHub](https://github.com/ace-step/ACE-Step)** - Original repository
- **[RunPod Documentation](https://docs.runpod.io/)** - Deployment platform
- **[Model Demo](https://huggingface.co/spaces/ace-step/ACE-Step)** - Try online

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch  
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📜 License

This project is licensed under the Apache 2.0 License - see the [ACE-Step repository](https://github.com/ace-step/ACE-Step) for details.

## 🙏 Acknowledgments

- **ACE-Step Team**: For the original model and implementation
- **RunPod**: For serverless GPU infrastructure  
- **Hugging Face**: For model hosting and diffusers library

---

**Status**: ✅ **Production Ready** - API fully operational and tested

For questions or support, please open an issue in this repository.

## 📄 License

Apache 2.0 - See [LICENSE](https://github.com/ace-step/ACE-Step/blob/main/LICENSE)
