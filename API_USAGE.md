# ACE-Step Serverless API Usage Documentation

## Overview

The ACE-Step Serverless API provides high-quality text-to-music generation using the ACE-Step 3.5B parameter model. This API is deployed on RunPod and generates music from text prompts in real-time.

## Endpoint Information

- **Base URL**: `https://api.runpod.ai/v2/9eb182ubs5j0jf`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Authentication**: Bearer token required

## Quick Start

### Basic Request

```bash
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "upbeat electronic dance music",
      "duration": 30.0,
      "guidance_scale": 8.0,
      "num_inference_steps": 20,
      "seed": 42
    }
  }'
```

### Python Example

```python
import requests
import base64
import json

# Configuration
endpoint_id = "9eb182ubs5j0jf"
api_key = "YOUR_API_KEY"
url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "input": {
        "prompt": "peaceful piano melody with nature sounds",
        "duration": 20.0,
        "guidance_scale": 7.5,
        "num_inference_steps": 25,
        "seed": 12345
    }
}

# Make request
response = requests.post(url, json=payload, headers=headers, timeout=180)

if response.status_code == 200:
    result = response.json()
    
    if result.get("status") == "COMPLETED":
        output = result.get("output", {})
        
        # Get audio data
        audio_base64 = output.get("audio_base64")
        if audio_base64:
            # Decode and save audio
            audio_data = base64.b64decode(audio_base64)
            with open("generated_music.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Music generated successfully!")
            print(f"🎵 Prompt: {output.get('prompt')}")
            print(f"⏱️ Generation time: {output.get('generation_time')}s")
            print(f"🎲 Seed: {output.get('seed')}")
        else:
            print("❌ No audio generated")
    else:
        print(f"❌ Generation failed: {result.get('error')}")
else:
    print(f"❌ HTTP Error: {response.status_code}")
```

## Request Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the music to generate |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration` | float | 30.0 | Length of generated audio in seconds (1-240) |
| `guidance_scale` | float | 8.0 | Controls adherence to prompt (1.0-20.0) |
| `num_inference_steps` | integer | 20 | Number of denoising steps (10-100) |
| `seed` | integer | random | Random seed for reproducible results |

### Parameter Guidelines

#### Duration
- **Range**: 1.0 to 240.0 seconds
- **Recommended**: 10-60 seconds for optimal quality
- **Note**: Longer durations increase generation time proportionally

#### Guidance Scale
- **Range**: 1.0 to 20.0
- **Low (1.0-5.0)**: More creative, less adherent to prompt
- **Medium (6.0-10.0)**: Balanced creativity and prompt adherence
- **High (11.0-20.0)**: Strict prompt adherence, less variation

#### Inference Steps
- **Range**: 10 to 100 steps
- **Fast (10-20)**: Quick generation, good quality
- **Balanced (20-40)**: Good quality/speed tradeoff
- **High Quality (40-100)**: Best quality, slower generation

## Response Format

### Successful Response

```json
{
  "status": "COMPLETED",
  "output": {
    "audio_base64": "UklGRjwAAABXQVZFZm10...",
    "generation_time": 25.43,
    "prompt": "peaceful piano melody with nature sounds",
    "duration": 20.0,
    "seed": 12345,
    "sample_rate": 48000,
    "success": true
  },
  "workerId": "worker123"
}
```

### Error Response

```json
{
  "status": "FAILED",
  "error": "Error message describing what went wrong"
}
```

## Example Prompts

### Genre-Based Prompts
```json
// Electronic
{"prompt": "upbeat electronic dance music with heavy bass"}

// Classical
{"prompt": "orchestral symphony with strings and brass"}

// Jazz
{"prompt": "smooth jazz with piano and saxophone"}

// Rock
{"prompt": "energetic rock music with electric guitar"}

// Ambient
{"prompt": "peaceful ambient soundscape with nature sounds"}
```

### Mood-Based Prompts
```json
// Happy
{"prompt": "cheerful and uplifting melody with bright instruments"}

// Sad
{"prompt": "melancholic piano ballad with emotional depth"}

// Energetic
{"prompt": "high-energy music with driving rhythm and powerful drums"}

// Relaxing
{"prompt": "calm and soothing instrumental with gentle flow"}
```

### Detailed Prompts
```json
// Specific instruments and style
{"prompt": "cinematic orchestral piece with epic drums, soaring strings, and heroic brass in C major"}

// Tempo and mood
{"prompt": "fast-paced electronic track with synthesizers, 128 BPM, futuristic and energetic"}

// Cultural style
{"prompt": "traditional folk music with acoustic guitar, warm and nostalgic atmosphere"}
```

## Performance Metrics

### Generation Times
- **10 seconds audio**: ~15-25 seconds
- **30 seconds audio**: ~25-40 seconds  
- **60 seconds audio**: ~45-70 seconds

### Audio Quality
- **Sample Rate**: 48 kHz
- **Format**: WAV (uncompressed)
- **Channels**: Stereo
- **Bit Depth**: 16-bit

## Error Handling

### Common Errors

| Error Type | Description | Solution |
|------------|-------------|----------|
| `Timeout` | Request exceeded time limit | Reduce duration or inference steps |
| `Invalid prompt` | Empty or invalid prompt | Provide descriptive text prompt |
| `Parameter out of range` | Invalid parameter values | Check parameter ranges |
| `Rate limit` | Too many requests | Wait before retrying |

### Retry Logic

```python
import time
import requests

def generate_music_with_retry(payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "COMPLETED":
                    return result
                elif result.get("status") == "FAILED":
                    print(f"Generation failed: {result.get('error')}")
                    return None
            
            # Retry on timeout or server error
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        except requests.exceptions.Timeout:
            print(f"Request timed out (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return None
```

## Advanced Usage

### Batch Generation

```python
prompts = [
    "happy pop song with upbeat rhythm",
    "dark electronic ambient soundscape", 
    "classical piano piece in minor key"
]

results = []
for i, prompt in enumerate(prompts):
    payload = {
        "input": {
            "prompt": prompt,
            "duration": 15.0,
            "seed": 1000 + i  # Different seed for each
        }
    }
    
    result = generate_music_with_retry(payload)
    if result:
        results.append(result)
        print(f"Generated {i+1}/{len(prompts)}: {prompt}")
```

### Save Audio Files

```python
def save_audio_from_response(result, filename):
    """Save audio from API response to file"""
    output = result.get("output", {})
    audio_base64 = output.get("audio_base64")
    
    if audio_base64:
        audio_data = base64.b64decode(audio_base64)
        with open(filename, "wb") as f:
            f.write(audio_data)
        return True
    return False

# Usage
if save_audio_from_response(result, "my_song.wav"):
    print("Audio saved successfully!")
```

## Best Practices

### Prompt Engineering
1. **Be Descriptive**: Include instruments, genre, mood, and style
2. **Use Musical Terms**: Tempo (BPM), key signatures, dynamics
3. **Specify Instruments**: Piano, guitar, drums, strings, etc.
4. **Include Mood/Emotion**: Happy, sad, energetic, peaceful

### Parameter Optimization
1. **Start with defaults** and adjust based on results
2. **Lower guidance_scale** for more creativity
3. **Higher inference_steps** for better quality
4. **Use seeds** for reproducible results

### Performance Tips
1. **Shorter durations** generate faster
2. **Batch similar requests** for efficiency  
3. **Cache results** using seeds for repeatability
4. **Monitor generation times** and adjust accordingly

## Troubleshooting

### Common Issues

1. **Slow Generation**
   - Reduce `duration` or `num_inference_steps`
   - Check RunPod worker availability

2. **Poor Quality Output**
   - Increase `num_inference_steps` (30-50)
   - Adjust `guidance_scale` (7-12 range)
   - Use more descriptive prompts

3. **Timeout Errors**
   - Increase request timeout (3+ minutes)
   - Reduce audio duration
   - Retry with exponential backoff

### Support

For technical issues or questions:
- Check RunPod status page for service availability
- Verify API key and endpoint configuration
- Monitor generation parameters and adjust as needed

## Rate Limits

- **Concurrent requests**: Limited by RunPod worker availability
- **Request frequency**: No explicit rate limits, but consider costs
- **Generation time**: Scales with audio duration and quality settings

---

*This API is powered by ACE-Step, a state-of-the-art music generation model with 3.5B parameters, deployed on RunPod serverless infrastructure.*
