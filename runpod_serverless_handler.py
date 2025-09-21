"""
RunPod Serverless Handler for ACE-Step Music Generator
Optimized for serverless deployment with cold start handling
"""

import runpod
import torch
import logging
import os
import sys
import base64
import tempfile
import time
import traceback
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def download_model():
    """Download and cache the model"""
    try:
        logger.info("Downloading ACE-Step model...")
        from huggingface_hub import snapshot_download
        
        # Download model to cache directory
        model_path = snapshot_download(
            repo_id="ACE-Step/ACE-Step-v1-3.5B",
            cache_dir="/runpod-volume",  # Use persistent storage if available
            resume_download=True
        )
        logger.info(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise

def load_model():
    """Load ACE-Step model (with caching for serverless)"""
    global model
    
    if model is not None:
        logger.info("Using cached model")
        return model
    
    try:
        logger.info("Loading ACE-Step model for serverless...")
        
        # Add ACE-Step to path
        sys.path.append('/opt/ACE-Step')
        
        # Import after path is set
        from ace_step.inference import InferencePipeline
        
        # Initialize pipeline
        model = InferencePipeline()
        logger.info("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load model: {str(e)}")

def generate_music(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate music from text prompt
    
    Expected input:
    {
        "prompt": "upbeat electronic music",
        "duration": 30.0,
        "seed": 42,
        "guidance_scale": 3.0,
        "num_inference_steps": 50
    }
    """
    
    start_time = time.time()
    
    try:
        # Parse input parameters
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}
        
        duration = job_input.get("duration", 30.0)
        seed = job_input.get("seed", None)
        guidance_scale = job_input.get("guidance_scale", 3.0)
        num_inference_steps = job_input.get("num_inference_steps", 50)
        
        logger.info(f"Generating music: '{prompt}' ({duration}s)")
        
        # Load model (will use cache if available)
        pipeline = load_model()
        
        # Generate music
        audio_array = pipeline.generate(
            prompt=prompt,
            duration=duration,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        
        # Convert to base64 WAV
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # ACE-Step typically outputs at 16kHz
            sf.write(tmp_file.name, audio_array, 16000)
            
            # Read and encode
            with open(tmp_file.name, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Cleanup
            os.unlink(tmp_file.name)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return {
            "audio_base64": audio_base64,
            "generation_time": generation_time,
            "prompt": prompt,
            "duration": duration,
            "seed": seed,
            "audio_length_seconds": len(audio_array) / 16000,
            "success": True
        }
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"Generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "error": error_msg,
            "generation_time": generation_time,
            "success": False
        }

def handler(job):
    """
    RunPod serverless handler
    This is the main entry point for serverless requests
    """
    
    job_input = job.get("input", {})
    
    logger.info(f"Received job: {job_input}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        }
        logger.info(f"GPU Info: {gpu_info}")
    else:
        logger.warning("No GPU available - will be very slow!")
    
    # Generate music
    result = generate_music(job_input)
    
    return result

# Initialize RunPod serverless
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless ACE-Step Handler")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
