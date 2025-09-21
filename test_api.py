"""
Example client for ACE-Step Music Generator API
"""

import requests
import json
import time
import os

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def generate_music(prompt, duration=60, seed=None):
    """Generate music using the API"""
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "duration": duration
    }
    
    if seed is not None:
        data["seed"] = seed
    
    print(f"🎵 Generating music...")
    print(f"📝 Prompt: {prompt}")
    print(f"⏱️  Duration: {duration} seconds")
    
    # Make API request
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/generate", json=data, timeout=300)  # 5 min timeout
        
        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            
            print(f"✅ Generation complete!")
            print(f"🎧 Audio file: {result['audio_path']}")
            print(f"🎯 Seed used: {result['seed']}")
            print(f"🔊 Sample rate: {result['sample_rate']} Hz")
            print(f"⚡ API time: {elapsed_time:.2f}s")
            print(f"🚀 Model time: {result['generation_time']:.2f}s")
            
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. Music generation can take a while...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"🌐 Connection error: {e}")
        return None

def main():
    """Main function with example usage"""
    
    print("🎼 ACE-Step Music Generator Client")
    print("=" * 40)
    
    # Check if API is running
    if not test_api_health():
        print("❌ API is not running!")
        print("Start the API first with: python simple_ace_api.py")
        return
    
    print("✅ API is running")
    print()
    
    # Example prompts
    examples = [
        {
            "prompt": "upbeat electronic dance music, 128 bpm, synthesizers, energetic",
            "duration": 30
        },
        {
            "prompt": "calm acoustic guitar melody, folk style, peaceful, 90 bpm",
            "duration": 45
        },
        {
            "prompt": "epic orchestral music, cinematic, powerful strings, brass section",
            "duration": 60
        },
        {
            "prompt": "jazz piano solo, smooth, improvisation, relaxed tempo",
            "duration": 30
        }
    ]
    
    # Interactive mode
    while True:
        print("\nChoose an option:")
        print("1-4: Generate example music")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example['prompt'][:50]}... ({example['duration']}s)")
        print("5. Custom prompt")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "6":
            print("👋 Goodbye!")
            break
        elif choice in ["1", "2", "3", "4"]:
            example = examples[int(choice) - 1]
            generate_music(example["prompt"], example["duration"])
        elif choice == "5":
            custom_prompt = input("Enter your music prompt: ").strip()
            if custom_prompt:
                try:
                    duration = int(input("Duration in seconds (10-240, default 60): ") or 60)
                    duration = max(10, min(240, duration))  # Clamp between 10-240
                except ValueError:
                    duration = 60
                
                generate_music(custom_prompt, duration)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
