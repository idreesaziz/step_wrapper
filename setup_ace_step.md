# ACE-Step Simple API Setup Guide

## Quick Setup

### 1. Clone and Install ACE-Step
```bash
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Run the Simple API (trainer-api.py)
```bash
# The trainer-api.py is already in the repository
python trainer-api.py
```

### 3. API Usage Examples

#### Python Client Example:
```python
import requests
import json

# API endpoint
url = "http://localhost:8000/generate"

# Simple request
data = {
    "prompt": "upbeat pop song with guitar and drums, 120 bpm, energetic"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Generated audio: {result['audio_path']}")
print(f"Sample rate: {result['sample_rate']}")
print(f"Seed used: {result['seed']}")
```

#### cURL Example:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "calm acoustic guitar melody, relaxing, folk style",
       "duration": 60,
       "infer_steps": 30
     }'
```

### 4. API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the music |
| `duration` | int | 240 | Duration in seconds |
| `infer_steps` | int | 60 | Inference steps (lower = faster, higher = better quality) |
| `guidance_scale` | float | 15.0 | How closely to follow the prompt |
| `omega_scale` | float | 10.0 | Additional guidance parameter |
| `seed` | int | null | Random seed for reproducible results |

### 5. Environment Configuration

Set environment variable for checkpoint directory:
```bash
export CHECKPOINT_DIR="./checkpoints"  # Will auto-download if not exists
```

## Performance Notes

- **First run**: Models will auto-download (~3.5GB)
- **GPU Memory**: Minimum 8GB VRAM recommended
- **Generation Speed**: 
  - A100: ~2.2s for 1 minute of music
  - RTX 4090: ~1.7s for 1 minute of music
  - RTX 3090: ~4.7s for 1 minute of music

## Health Check

```bash
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```
