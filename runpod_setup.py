#!/usr/bin/env python3
"""
RunPod wrapper for ACE-Step Music Generator API
Handles RunPod-specific setup and endpoints
"""

import os
import sys
import json
import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RunPod Input/Output Models
class RunPodInput(BaseModel):
    input: Dict[str, Any]

class RunPodOutput(BaseModel):
    output: Dict[str, Any]
    status: str = "success"

class MusicGenerationInput(BaseModel):
    prompt: str
    duration: Optional[float] = 30.0
    seed: Optional[int] = None
    guidance_scale: Optional[float] = 3.0
    num_inference_steps: Optional[int] = 50

class MusicGenerationOutput(BaseModel):
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    generation_time: float
    error: Optional[str] = None

# Global variables for model
model = None
model_loaded = False

def load_model():
    """Load ACE-Step model"""
    global model, model_loaded
    
    if model_loaded:
        return model
        
    try:
        logger.info("Loading ACE-Step model...")
        
        # Import ACE-Step components
        sys.path.append('/app/ACE-Step')
        from ace_step.inference import InferencePipeline
        
        # Initialize model
        model = InferencePipeline()
        model_loaded = True
        
        logger.info("ACE-Step model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def generate_music_sync(input_data: MusicGenerationInput) -> MusicGenerationOutput:
    """Synchronous music generation"""
    import time
    import base64
    import tempfile
    import soundfile as sf
    
    start_time = time.time()
    
    try:
        # Load model if not already loaded
        pipeline = load_model()
        
        # Generate music
        logger.info(f"Generating music for prompt: '{input_data.prompt}'")
        
        # Call the generation method
        audio_array = pipeline.generate(
            prompt=input_data.prompt,
            duration=input_data.duration,
            seed=input_data.seed,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps
        )
        
        # Save to temporary file and encode as base64
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, 16000)  # ACE-Step uses 16kHz
            
            # Read and encode as base64
            with open(tmp_file.name, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        generation_time = time.time() - start_time
        logger.info(f"Music generation completed in {generation_time:.2f}s")
        
        return MusicGenerationOutput(
            audio_base64=audio_base64,
            generation_time=generation_time
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"Generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return MusicGenerationOutput(
            generation_time=generation_time,
            error=error_msg
        )

# FastAPI app
app = FastAPI(
    title="ACE-Step Music Generator - RunPod",
    description="Music generation API using ACE-Step model on RunPod",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ACE-Step Music Generator API",
        "status": "ready",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }

@app.post("/runsync", response_model=RunPodOutput)
async def run_sync(request: RunPodInput):
    """RunPod synchronous endpoint"""
    try:
        # Parse input
        input_data = MusicGenerationInput(**request.input)
        
        # Generate music
        result = generate_music_sync(input_data)
        
        if result.error:
            return RunPodOutput(
                output={"error": result.error},
                status="error"
            )
        
        return RunPodOutput(
            output={
                "audio_base64": result.audio_base64,
                "generation_time": result.generation_time,
                "prompt": input_data.prompt,
                "duration": input_data.duration
            }
        )
        
    except Exception as e:
        logger.error(f"RunSync error: {str(e)}")
        logger.error(traceback.format_exc())
        return RunPodOutput(
            output={"error": str(e)},
            status="error"
        )

@app.post("/generate", response_model=MusicGenerationOutput)
async def generate_music(request: MusicGenerationInput):
    """Direct music generation endpoint"""
    return generate_music_sync(request)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting ACE-Step API...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available. GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    else:
        logger.warning("CUDA not available. Running on CPU (will be slow!)")
    
    # Pre-load model in background
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to pre-load model: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
