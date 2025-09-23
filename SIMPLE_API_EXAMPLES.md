# Simple ACE-Step API Examples

## Overview
The ACE-Step API now has a simple, clean interface with just 3 parameters:
- `prompt` (required): Description of the music you want
- `lyrics` (optional): Song lyrics with structure tags
- `duration` (optional): Length in seconds (default: 30)

## Basic Examples

### 1. Instrumental Music Only

```python
import requests
import base64

url = "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

# Simple instrumental generation
payload = {
    "input": {
        "prompt": "upbeat electronic dance music with heavy bass"
    }
}

response = requests.post(url, json=payload, headers=headers)
```

### 2. Music with Lyrics

```python
# Generate music with lyrics
payload = {
    "input": {
        "prompt": "acoustic folk song with gentle guitar",
        "lyrics": """[verse]
Walking through the autumn leaves
Colors falling all around me
[chorus]
This is where I want to be
In this moment, feeling free
[verse]
Every step tells a story
Of the paths that led me here""",
        "duration": 45.0
    }
}

response = requests.post(url, json=payload, headers=headers)
```

### 3. Different Music Styles

```bash
# Pop song
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "catchy pop song with upbeat drums and synths",
      "lyrics": "[verse]\nLights are shining bright tonight\n[chorus]\nWe are young and free",
      "duration": 30.0
    }
  }'

# Jazz instrumental
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "smooth jazz with saxophone and piano",
      "duration": 60.0
    }
  }'

# Rock with vocals
curl -X POST "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "energetic rock song with electric guitar and drums",
      "lyrics": "[verse]\nBreaking through the silence\n[chorus]\nRock and roll forever",
      "duration": 40.0
    }
  }'
```

## Complete Python Example with File Saving

```python
import requests
import base64
import json
import time

def generate_music(prompt, lyrics=None, duration=30.0):
    """Generate music with the simplified 3-parameter API"""
    
    url = "https://api.runpod.ai/v2/9eb182ubs5j0jf/runsync"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    
    # Build payload with only the 3 simple parameters
    payload = {
        "input": {
            "prompt": prompt,
            "duration": duration
        }
    }
    
    # Add lyrics if provided
    if lyrics:
        payload["input"]["lyrics"] = lyrics
    
    print(f"🎵 Generating: {prompt}")
    if lyrics:
        print(f"🎤 With lyrics: {len(lyrics)} characters")
    print(f"⏱️ Duration: {duration}s")
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=180)
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            audio_base64 = output.get("audio_base64")
            
            if audio_base64:
                # Save the generated audio
                audio_data = base64.b64decode(audio_base64)
                filename = f"generated_{int(time.time())}.wav"
                
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                gen_time = time.time() - start_time
                print(f"✅ Success! Saved as {filename}")
                print(f"⚡ Generated in {gen_time:.1f}s")
                return filename
            else:
                print("❌ No audio data received")
        else:
            print(f"❌ Error: {result}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
    
    return None

# Example usage
if __name__ == "__main__":
    # Instrumental
    generate_music(
        prompt="peaceful ambient music with soft piano",
        duration=25.0
    )
    
    # With lyrics
    generate_music(
        prompt="indie folk song with acoustic guitar",
        lyrics="[verse]\nMorning light through window pane\n[chorus]\nEverything will be okay",
        duration=35.0
    )
```

## Key Benefits of Simplified API

1. **Easy to Use**: Only 3 parameters to remember
2. **Smart Defaults**: The API uses optimal internal settings automatically
3. **Lyrics Support**: Full lyric generation with structure tags
4. **Quality**: Advanced parameters are auto-tuned for best results
5. **Fast**: Streamlined interface reduces complexity and errors

## Internal Optimizations

When you use this simple API, behind the scenes it automatically:
- Sets `guidance_scale` to 3.0 for good quality
- Enables `guidance_scale_lyric` (2.0) when lyrics are provided  
- Uses 50 inference steps for quality/speed balance
- Enables ERG (Enhanced Representation Guidance) for better results
- Generates random seeds for varied output each time

This gives you professional-quality results with minimal configuration!
