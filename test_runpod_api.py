#!/usr/bin/env python3
"""
Test client for ACE-Step RunPod API
"""

import requests
import json
import base64
import time
import os
from typing import Optional

class RunPodClient:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        
    def health_check(self) -> dict:
        """Check API health"""
        response = requests.get(f"{self.api_url}/health")
        return response.json()
    
    def generate_music(self, prompt: str, duration: float = 30.0, 
                      seed: Optional[int] = None, output_file: str = "generated_music.wav") -> dict:
        """Generate music using RunPod format"""
        
        payload = {
            "input": {
                "prompt": prompt,
                "duration": duration
            }
        }
        
        if seed is not None:
            payload["input"]["seed"] = seed
            
        print(f"🎵 Generating music: '{prompt}' ({duration}s)")
        print(f"🔗 API URL: {self.api_url}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/runsync",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            request_time = time.time() - start_time
            
            if result.get("status") == "success":
                # Save audio file
                audio_base64 = result["output"]["audio_base64"]
                audio_data = base64.b64decode(audio_base64)
                
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                
                print(f"✅ Success! Generated in {result['output']['generation_time']:.1f}s")
                print(f"📁 Saved to: {output_file}")
                print(f"📊 Total request time: {request_time:.1f}s")
                
                return {
                    "success": True,
                    "file": output_file,
                    "generation_time": result["output"]["generation_time"],
                    "total_time": request_time
                }
            else:
                error = result["output"].get("error", "Unknown error")
                print(f"❌ Generation failed: {error}")
                return {"success": False, "error": error}
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (5 minutes)")
            return {"success": False, "error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            print(f"🔗 Request failed: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Interactive test client"""
    
    # Get API URL
    api_url = input("Enter your RunPod API URL (e.g., https://your-pod-8000.proxy.runpod.net): ").strip()
    
    if not api_url:
        print("❌ API URL is required")
        return
    
    client = RunPodClient(api_url)
    
    # Health check
    print("\n🏥 Checking API health...")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health.get('status', 'unknown')}")
        print(f"🤖 Model Loaded: {health.get('model_loaded', False)}")
        print(f"🎮 GPU Available: {health.get('gpu_available', False)}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Continuing anyway...")
    
    # Test prompts
    test_prompts = [
        ("upbeat electronic dance music", 15.0),
        ("peaceful piano melody", 10.0),
        ("energetic rock guitar solo", 20.0),
        ("ambient forest sounds", 25.0),
        ("jazz saxophone improvisation", 30.0)
    ]
    
    while True:
        print("\n" + "="*50)
        print("ACE-Step Music Generator Test Client")
        print("="*50)
        
        print("Options:")
        print("1. Custom prompt")
        print("2. Test with sample prompts")
        print("3. Batch test")
        print("4. Health check")
        print("5. Quit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            # Custom prompt
            prompt = input("Enter music prompt: ").strip()
            if not prompt:
                print("❌ Prompt required")
                continue
                
            try:
                duration = float(input("Duration in seconds (default 30): ") or "30")
            except ValueError:
                duration = 30.0
                
            try:
                seed_input = input("Seed (optional, press enter for random): ").strip()
                seed = int(seed_input) if seed_input else None
            except ValueError:
                seed = None
            
            filename = f"custom_{int(time.time())}.wav"
            client.generate_music(prompt, duration, seed, filename)
            
        elif choice == "2":
            # Sample prompts
            print("\n📋 Sample Prompts:")
            for i, (prompt, duration) in enumerate(test_prompts, 1):
                print(f"{i}. {prompt} ({duration}s)")
            
            try:
                selection = int(input("Select prompt (1-5): ")) - 1
                if 0 <= selection < len(test_prompts):
                    prompt, duration = test_prompts[selection]
                    filename = f"sample_{selection+1}_{int(time.time())}.wav"
                    client.generate_music(prompt, duration, None, filename)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "3":
            # Batch test
            print("\n🔄 Running batch test with all sample prompts...")
            results = []
            
            for i, (prompt, duration) in enumerate(test_prompts, 1):
                print(f"\n--- Test {i}/{len(test_prompts)} ---")
                filename = f"batch_{i}_{int(time.time())}.wav"
                result = client.generate_music(prompt, duration, 42, filename)
                results.append({"prompt": prompt, "result": result})
                
                if not result["success"]:
                    print("❌ Batch test failed, stopping...")
                    break
                    
            # Summary
            print(f"\n📊 Batch Test Summary:")
            successful = sum(1 for r in results if r["result"]["success"])
            print(f"✅ Successful: {successful}/{len(results)}")
            
            if successful > 0:
                avg_time = sum(r["result"]["generation_time"] for r in results if r["result"]["success"]) / successful
                print(f"⏱️  Average generation time: {avg_time:.1f}s")
                
        elif choice == "4":
            # Health check
            print("\n🏥 Health Check:")
            try:
                health = client.health_check()
                print(json.dumps(health, indent=2))
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
