#!/usr/bin/env python3
"""
Test client for RunPod Serverless ACE-Step API
"""

import requests
import json
import base64
import time
from typing import Dict, Any

class RunPodServerlessClient:
    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize RunPod Serverless client
        
        Args:
            endpoint_id: Your RunPod endpoint ID (e.g., "9eb182ubs5j0jf")
            api_key: Your RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
    def generate_music_sync(self, prompt: str, duration: float = 30.0, 
                           seed: int = None, timeout: int = 300) -> Dict[str, Any]:
        """
        Generate music synchronously (wait for completion)
        """
        
        payload = {
            "input": {
                "prompt": prompt,
                "duration": duration
            }
        }
        
        if seed is not None:
            payload["input"]["seed"] = seed
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"🎵 Generating: '{prompt}' ({duration}s)")
        print(f"🔗 Endpoint: {self.endpoint_id}")
        
        start_time = time.time()
        
        try:
            # Submit job
            response = requests.post(
                f"{self.base_url}/runsync",
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            total_time = time.time() - start_time
            
            if result.get("status") == "COMPLETED":
                output = result.get("output", {})
                
                if output.get("success"):
                    print(f"✅ Success! Generated in {output.get('generation_time', 0):.1f}s")
                    print(f"📊 Total time: {total_time:.1f}s")
                    return {
                        "success": True,
                        "audio_base64": output.get("audio_base64"),
                        "generation_time": output.get("generation_time"),
                        "total_time": total_time
                    }
                else:
                    error = output.get("error", "Unknown error")
                    print(f"❌ Generation failed: {error}")
                    return {"success": False, "error": error}
            else:
                error = result.get("error", "Job failed")
                print(f"❌ Job failed: {error}")
                return {"success": False, "error": error}
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out ({timeout}s)")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            print(f"💥 Error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_music_async(self, prompt: str, duration: float = 30.0, 
                            seed: int = None) -> Dict[str, Any]:
        """
        Generate music asynchronously (returns job ID immediately)
        """
        
        payload = {
            "input": {
                "prompt": prompt,
                "duration": duration
            }
        }
        
        if seed is not None:
            payload["input"]["seed"] = seed
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/run",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "IN_QUEUE":
                job_id = result.get("id")
                print(f"🚀 Job submitted: {job_id}")
                return {"success": True, "job_id": job_id}
            else:
                return {"success": False, "error": "Failed to queue job"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of an async job"""
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Interactive test client"""
    
    print("🎵 RunPod Serverless ACE-Step Test Client")
    print("=" * 45)
    
    # Get credentials
    endpoint_id = input("Enter your RunPod Endpoint ID (e.g., 9eb182ubs5j0jf): ").strip()
    api_key = input("Enter your RunPod API Key: ").strip()
    
    if not endpoint_id or not api_key:
        print("❌ Both Endpoint ID and API Key are required")
        return
    
    client = RunPodServerlessClient(endpoint_id, api_key)
    
    # Test prompts
    test_prompts = [
        ("upbeat electronic dance music", 15.0),
        ("peaceful piano melody", 10.0),
        ("energetic rock guitar solo", 20.0),
        ("ambient nature sounds", 25.0),
        ("jazz saxophone solo", 30.0)
    ]
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Custom prompt (synchronous)")
        print("2. Test sample prompts")
        print("3. Async job test")
        print("4. Check job status")
        print("5. Quit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            # Custom prompt
            prompt = input("Enter music prompt: ").strip()
            if not prompt:
                print("❌ Prompt required")
                continue
                
            try:
                duration = float(input("Duration (default 30s): ") or "30")
            except ValueError:
                duration = 30.0
                
            try:
                seed_input = input("Seed (optional): ").strip()
                seed = int(seed_input) if seed_input else None
            except ValueError:
                seed = None
            
            result = client.generate_music_sync(prompt, duration, seed)
            
            if result["success"] and result.get("audio_base64"):
                # Save audio file
                filename = f"generated_{int(time.time())}.wav"
                audio_data = base64.b64decode(result["audio_base64"])
                with open(filename, "wb") as f:
                    f.write(audio_data)
                print(f"💾 Saved to: {filename}")
                
        elif choice == "2":
            # Sample prompts
            print("\n📋 Sample Prompts:")
            for i, (prompt, duration) in enumerate(test_prompts, 1):
                print(f"{i}. {prompt} ({duration}s)")
            
            try:
                selection = int(input("Select (1-5): ")) - 1
                if 0 <= selection < len(test_prompts):
                    prompt, duration = test_prompts[selection]
                    result = client.generate_music_sync(prompt, duration, 42)
                    
                    if result["success"] and result.get("audio_base64"):
                        filename = f"sample_{selection+1}.wav"
                        audio_data = base64.b64decode(result["audio_base64"])
                        with open(filename, "wb") as f:
                            f.write(audio_data)
                        print(f"💾 Saved to: {filename}")
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "3":
            # Async test
            prompt = input("Enter prompt for async job: ").strip()
            if prompt:
                result = client.generate_music_async(prompt, 20.0)
                if result["success"]:
                    print(f"✅ Job ID: {result['job_id']}")
                    print("Use option 4 to check status")
                
        elif choice == "4":
            # Check job status
            job_id = input("Enter Job ID: ").strip()
            if job_id:
                status = client.get_job_status(job_id)
                print(f"📊 Status: {json.dumps(status, indent=2)}")
                
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
