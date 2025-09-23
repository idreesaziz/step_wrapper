# 🎵 ACE-Step API - How to Pass Lyrics

## Overview
The ACE-Step serverless API now supports **full lyrics generation** with song structure tags and multi-language support.

## 🎤 Basic Lyrics Usage

### Simple Lyrics Request
```json
{
  "input": {
    "prompt": "upbeat pop song with electric guitar and drums",
    "lyrics": "[verse]\nWalking down the street tonight\nCity lights are burning bright\n\n[chorus]\nThis is our moment to shine\nEverything's gonna be fine",
    "duration": 60.0,
    "guidance_scale": 8.0,
    "guidance_scale_lyric": 3.0,
    "num_inference_steps": 50,
    "use_erg_lyric": true
  }
}
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

# Request with lyrics
payload = {
    "input": {
        "prompt": "emotional ballad with piano and strings",
        "lyrics": """[verse]
Sometimes I wonder if you think of me
When the night falls and you're all alone
These memories keep calling out to me
Like echoes from a place we used to know

[chorus]
We were young and wild and free
Nothing could tear us apart
Now I'm holding onto what we used to be
These fragments of your heart

[verse]
I drive past all our favorite places
See your ghost in every crowded room
All these unfamiliar faces
Can't replace what we once knew

[bridge]
If I could turn back time
I'd make you mine
One more time""",
        "duration": 120.0,
        "guidance_scale": 10.0,
        "guidance_scale_lyric": 4.0,
        "guidance_scale_text": 2.0,
        "num_inference_steps": 60,
        "use_erg_lyric": True,
        "seed": 42
    }
}

# Make request
response = requests.post(url, headers=headers, json=payload)
result = response.json()

if result.get("success"):
    # Decode and save audio
    audio_data = base64.b64decode(result["audio_base64"])
    with open("song_with_lyrics.wav", "wb") as f:
        f.write(audio_data)
    
    print(f"Generated song with lyrics in {result['generation_time']:.1f}s")
    print(f"Used lyrics: {bool(result.get('lyrics'))}")
else:
    print(f"Error: {result.get('error')}")
```

## 🏷️ Song Structure Tags

### Supported Structure Tags
- `[verse]` - Song verses
- `[chorus]` - Main chorus/hook
- `[bridge]` - Bridge section
- `[outro]` - Song ending
- `[intro]` - Song beginning  
- `[pre-chorus]` - Pre-chorus section
- `[instrumental]` or `[inst]` - Instrumental sections

### Example with Full Song Structure
```json
{
  "input": {
    "prompt": "rock anthem with heavy guitars, 120 BPM",
    "lyrics": "[intro]\nGuitar riff and drums\n\n[verse]\nStanding on the edge of something new\nFighting for a dream that could come true\nEvery step we take, we're getting closer\nTo the life we always knew\n\n[pre-chorus]\nCan you hear it calling?\nCan you feel it in your soul?\n\n[chorus]\nWe are the champions of our destiny\nRising up with fire and energy\nNothing's gonna stop us now\nWe'll show the world somehow\n\n[verse]\nThrough the darkest nights and longest days\nWe will find our own and better ways\nEvery battle won makes us stronger\nIn this game that we must play\n\n[pre-chorus]\nCan you hear it calling?\nCan you feel it in your soul?\n\n[chorus]\nWe are the champions of our destiny\nRising up with fire and energy\nNothing's gonna stop us now\nWe'll show the world somehow\n\n[bridge]\nWhen the world comes crashing down\nWe'll stand our ground\nWhen the odds are stacked against us\nWe won't back down\n\n[chorus]\nWe are the champions of our destiny\nRising up with fire and energy\nNothing's gonna stop us now\nWe'll show the world somehow\n\n[outro]\nChampions forever\nChampions together\n\n[instrumental]",
    "duration": 180.0,
    "guidance_scale": 12.0,
    "guidance_scale_lyric": 5.0,
    "guidance_scale_text": 3.0
  }
}
```

## 🌍 Multi-Language Support

### Supported Languages
English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Russian, Arabic, Hindi, and 50+ more languages.

### Spanish Example
```json
{
  "input": {
    "prompt": "salsa music with brass instruments and percussion",
    "lyrics": "[verso]\nBaila conmigo bajo las estrellas\nLa noche es nuestra, no hay que esperar\nSiente el ritmo en tus venas\nDéjate llevar por este compás\n\n[coro]\nVamos a bailar hasta el amanecer\nCon esta salsa que me hace enloquecer\nTus ojos brillan como diamantes\nEres la reina de todos los amantes",
    "guidance_scale_lyric": 4.0
  }
}
```

### Japanese Example  
```json
{
  "input": {
    "prompt": "J-pop song with synthesizers and electronic drums",
    "lyrics": "[verse]\n君と歩いた夏の日々\n青い空と白い雲\n忘れられない思い出が\n心の中で光ってる\n\n[chorus]\n永遠に続くこの愛を\n信じていたい今も\n君がいれば怖くない\n未来への扉を開こう",
    "guidance_scale_lyric": 3.5
  }
}
```

## 🎛️ Guidance Parameters

### `guidance_scale_lyric` (0.0 - 10.0)
Controls how strongly the model follows the lyrics:
- `0.0`: Lyrics ignored (instrumental)
- `1.0-3.0`: Subtle lyric influence
- `3.0-6.0`: Strong lyric alignment (recommended)
- `6.0-10.0`: Very strict lyric following

### `guidance_scale_text` (0.0 - 10.0)  
Controls how strongly the model follows the text prompt:
- `0.0`: Prompt ignored
- `1.0-3.0`: Subtle prompt influence  
- `3.0-6.0`: Strong prompt alignment
- `6.0-10.0`: Very strict prompt following

### `use_erg_lyric` (boolean)
Enable/disable ERG (Enhanced Representation Generation) for lyrics:
- `true`: Enhanced lyric processing (recommended)
- `false`: Basic lyric processing

## 🎵 Instrumental Generation

### Generate Instrumental Sections
```json
{
  "input": {
    "prompt": "epic orchestral instrumental with strings and brass",
    "lyrics": "[instrumental]",
    "duration": 90.0,
    "guidance_scale_lyric": 0.0
  }
}
```

### Mixed Vocal and Instrumental
```json
{
  "input": {
    "prompt": "progressive rock with guitar solos",
    "lyrics": "[verse]\nIn the silence of the night\n\n[instrumental]\nGuitar solo section\n\n[verse]\nStars are shining bright\n\n[instrumental]\nDrum solo and guitar harmony",
    "guidance_scale_lyric": 3.0
  }
}
```

## 🔧 Best Practices

### 1. Lyric Quality Tips
- Use proper song structure tags
- Keep verses/choruses consistent length
- Use natural, singable phrases
- Avoid overly complex vocabulary

### 2. Parameter Tuning
- Start with `guidance_scale_lyric: 3.0-5.0`
- Use `guidance_scale_text: 1.0-3.0` for style control
- Set `guidance_scale: 8.0-12.0` for overall quality
- Use `num_inference_steps: 50-80` for best results

### 3. Duration Recommendations
- Short songs (30-60s): Simple structure with 1-2 verses + chorus
- Medium songs (60-120s): Full structure with bridge
- Long songs (120-240s): Extended structure with multiple sections

## ❌ Common Mistakes

### Don't Use Genre Tags in Lyrics
```json
// ❌ WRONG - Don't put genre info in lyrics
{
  "lyrics": "[pop] [upbeat] This is a love song"
}

// ✅ CORRECT - Put genre info in prompt
{
  "prompt": "upbeat pop love song",
  "lyrics": "[verse]\nThis is a love song"
}
```

### Don't Skip Structure Tags
```json
// ❌ WRONG - No structure tags
{
  "lyrics": "Walking down the street\nFeeling the beat"
}

// ✅ CORRECT - Use proper structure
{
  "lyrics": "[verse]\nWalking down the street\nFeeling the beat"
}
```

## 🎯 Example Response

```json
{
  "audio_base64": "UklGRv7+/f5XQVZFZm10...",
  "generation_time": 45.2,
  "prompt": "upbeat pop song with guitar",
  "lyrics": "[verse]\nWalking down the street tonight...",
  "duration": 60.0,
  "seed": 42,
  "sample_rate": 48000,
  "guidance_scale_lyric": 4.0,
  "guidance_scale_text": 2.0,
  "use_erg_lyric": true,
  "success": true
}
```

## 📝 Quick Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `lyrics` | string | `""` | Any text | Song lyrics with structure tags |
| `guidance_scale_lyric` | float | `0.0` | 0.0-10.0 | How strongly to follow lyrics |
| `guidance_scale_text` | float | `0.0` | 0.0-10.0 | How strongly to follow text prompt |
| `use_erg_lyric` | boolean | `true` | true/false | Enable enhanced lyric processing |

---

🎵 **Ready to create amazing music with lyrics!** The ACE-Step API now supports full lyric generation with professional song structure and multi-language capabilities.
