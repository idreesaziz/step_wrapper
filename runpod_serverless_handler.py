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
        
        # Add ACE-Step to path in multiple ways
        ace_step_path = '/opt/ACE-Step'
        if ace_step_path not in sys.path:
            sys.path.insert(0, ace_step_path)
        
        # Change to ACE-Step directory for relative imports
        import os
        original_dir = os.getcwd()
        os.chdir(ace_step_path)
        
        # Import required modules
        from acestep.pipeline_ace_step import ACEStepPipeline
        from acestep.apg_guidance import apg_forward, MomentumBuffer
        from transformers import AutoTokenizer
        import random
        from diffusers.utils.torch_utils import randn_tensor
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
        from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        
        # Create InferencePipeline wrapper class (copied from trainer-api.py)
        class InferencePipeline:
            def __init__(self, checkpoint_dir: str, device: str = "cuda"):
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
                logger.info(f"Initializing model on device: {self.device}")

                # Load the ACEStepPipeline with consistent dtype
                self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)
                
                # Get the pipeline's dtype for consistency
                self.dtype = self.acestep_pipeline.dtype
                logger.info(f"Using model dtype: {self.dtype}")

                # Initialize components with consistent dtype (don't force float conversion)
                self.transformers = self.acestep_pipeline.ace_step_transformer.to(self.device).eval()
                self.dcae = self.acestep_pipeline.music_dcae.to(self.device).eval() 
                self.text_encoder_model = self.acestep_pipeline.text_encoder_model.to(self.device).eval()
                self.text_tokenizer = self.acestep_pipeline.text_tokenizer

                # Initialize scheduler
                self.scheduler = FlowMatchEulerDiscreteScheduler()

                # Ensure no gradients are computed
                self.transformers.requires_grad_(False)
                self.dcae.requires_grad_(False)
                self.text_encoder_model.requires_grad_(False)

                # Initialize scheduler exactly like trainer-api.py
                self.scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=3.0,
                )
                
            def get_text_embeddings(self, texts, device, text_max_length=256):
                """Get text embeddings from prompts - exactly like trainer-api.py"""
                inputs = self.text_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=text_max_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = self.text_encoder_model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                return last_hidden_states, attention_mask

            def diffusion_process(self, duration, encoder_text_hidden_states, text_attention_mask, 
                                speaker_embds, lyric_token_ids, lyric_mask, random_generator=None,
                                infer_steps=60, guidance_scale=15.0, omega_scale=10.0):
                """Simplified diffusion process wrapper"""
                # Use ACEStepPipeline's text2music_diffusion_process
                pred_latents = self.acestep_pipeline.text2music_diffusion_process(
                    duration=duration,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    random_generators=[random_generator] if random_generator else None,
                    infer_steps=infer_steps,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                )
                return pred_latents
                
            def generate_audio(self, prompt: str, duration: int, infer_steps: int, 
                             guidance_scale: float, omega_scale: float, seed=None):
                """Generate audio from text prompt (copied from trainer-api.py)"""
                # Set random seed
                if seed is not None:
                    random.seed(seed)
                    torch.manual_seed(seed)
                else:
                    seed = random.randint(0, 2**32 - 1)
                    random.seed(seed)
                    torch.manual_seed(seed)

                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Get text embeddings
                encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
                    prompt, self.device
                )

                # Dummy speaker embeddings and lyrics (since not provided in API request)
                bsz = 1
                speaker_embds = torch.zeros(bsz, 512, device=self.device, dtype=encoder_text_hidden_states.dtype)
                lyric_token_ids = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)
                lyric_mask = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)

                # Run diffusion process
                pred_latents = self.diffusion_process(
                    duration=duration,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    random_generator=generator,
                    infer_steps=infer_steps,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                )

                # Decode latents to audio  
                audio_lengths = torch.tensor([int(duration * 44100)], device=self.device, dtype=torch.long)
                sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)

                # Save audio
                import tempfile
                import torchaudio
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = tempfile.mktemp(suffix=f"_{timestamp}_{seed}.wav")
                torchaudio.save(temp_path, pred_wavs.float().cpu(), sr)

                return temp_path, sr, seed
        
        # Initialize the wrapper
        checkpoint_dir = "/opt/ACE-Step"
        model = InferencePipeline(checkpoint_dir)
        logger.info("Model loaded successfully!")
        
        # Restore original directory
        os.chdir(original_dir)
        
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Also log the current Python path for debugging
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"ACE-Step directory contents: {os.listdir('/opt/ACE-Step') if os.path.exists('/opt/ACE-Step') else 'Not found'}")
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
        
        # Generate music using ACEStepPipeline
        audio_path, sample_rate, used_seed = pipeline.generate_audio(
            prompt=prompt,
            duration=int(duration),  # Duration should be int
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            omega_scale=1.0,  # Default omega scale
            seed=seed,
        )
        
        # Convert audio file to base64
        with open(audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Cleanup the temporary audio file
        os.unlink(audio_path)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return {
            "audio_base64": audio_base64,
            "generation_time": generation_time,
            "prompt": prompt,
            "duration": duration,
            "seed": used_seed,
            "sample_rate": sample_rate,
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
