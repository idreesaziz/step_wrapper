"""
Simple ACE-Step Music Generation API
A minimal FastAPI wrapper around ACE-Step for easy music generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import torch
import torchaudio
import random
from datetime import datetime
import uvicorn
from loguru import logger

# Import ACE-Step components
try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from acestep.apg_guidance import apg_forward, MomentumBuffer
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
except ImportError as e:
    logger.error(f"Failed to import ACE-Step components: {e}")
    logger.error("Please install ACE-Step first: pip install git+https://github.com/ace-step/ACE-Step.git")
    exit(1)

app = FastAPI(
    title="ACE-Step Music Generator API",
    description="Simple API for generating music from text prompts using ACE-Step",
    version="1.0.0"
)

# Request/Response Models
class MusicRequest(BaseModel):
    prompt: str
    duration: int = 60  # seconds, default 1 minute
    seed: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "upbeat electronic dance music, 128 bpm, energetic, synthesizers, drums",
                "duration": 60,
                "seed": 42
            }
        }

class MusicResponse(BaseModel):
    status: str
    audio_path: str
    prompt: str
    duration: int
    seed: int
    sample_rate: int
    generation_time: float

# Global model instance
model = None

class SimpleMusicGenerator:
    def __init__(self, checkpoint_dir: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initializing ACE-Step on device: {self.device}")
        
        # Initialize ACE-Step pipeline
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            dtype="bfloat16" if self.device.type == "cuda" else "float32",
            torch_compile=False,  # Set to True for faster inference if you have triton
            cpu_offload=self.device.type == "cpu"
        )
        
        logger.info("ACE-Step model loaded successfully")
    
    def generate(self, prompt: str, duration: int = 60, seed: Optional[int] = None) -> tuple:
        """Generate music from text prompt"""
        import time
        start_time = time.time()
        
        # Set random seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Create output directory
        output_dir = "generated_music"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/music_{timestamp}_{seed}.wav"
        
        # Generate music using ACE-Step pipeline
        try:
            result = self.pipeline(
                prompt=prompt,
                lyrics="",  # Empty lyrics for instrumental
                audio_duration=duration,
                infer_step=27,  # Good balance of speed/quality
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg", 
                omega_scale=10.0,
                manual_seeds=[seed],
                save_path=output_path,
                format="wav"
            )
            
            generation_time = time.time() - start_time
            
            # Get audio info
            if os.path.exists(output_path):
                info = torchaudio.info(output_path)
                sample_rate = info.sample_rate
            else:
                sample_rate = 48000  # Default
                
            return output_path, sample_rate, generation_time
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    global model
    try:
        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
        model = SimpleMusicGenerator(checkpoint_dir=checkpoint_dir)
        logger.info("API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "ACE-Step Music Generator API",
        "status": "ready",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy", "model": "ACE-Step"}

@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate duration
    if not (10 <= request.duration <= 240):
        raise HTTPException(status_code=400, detail="Duration must be between 10 and 240 seconds")
    
    try:
        logger.info(f"Generating music: '{request.prompt}' ({request.duration}s)")
        
        audio_path, sample_rate, generation_time = model.generate(
            prompt=request.prompt,
            duration=request.duration,
            seed=request.seed
        )
        
        return MusicResponse(
            status="success",
            audio_path=audio_path,
            prompt=request.prompt,
            duration=request.duration,
            seed=request.seed or 0,
            sample_rate=sample_rate,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Error generating music: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run with: python simple_ace_api.py
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
