"""
RunPod Serverless Handler for ACE-Step Music Generator
Based on official trainer-api.py implementation
"""

import runpod
import torch
import torchaudio
import logging
import os
import sys
import base64
import tempfile
import time
import traceback
import random
from typing import Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def load_model():
    """Load ACE-Step model - EXACTLY like trainer-api.py"""
    global model
    
    if model is not None:
        logger.info("Using cached model")
        return model
    
    try:
        logger.info("Loading ACE-Step model for serverless...")
        
        # Add ACE-Step to path
        ace_step_path = '/opt/ACE-Step'
        if ace_step_path not in sys.path:
            sys.path.insert(0, ace_step_path)
        
        # Change to ACE-Step directory for imports
        original_dir = os.getcwd()
        os.chdir(ace_step_path)
        
        # Import EXACTLY like trainer-api.py
        from acestep.pipeline_ace_step import ACEStepPipeline
        from acestep.apg_guidance import apg_forward, MomentumBuffer
        from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
        from transformers import AutoTokenizer
        from diffusers.utils.torch_utils import randn_tensor
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
        from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        
        # InferencePipeline EXACTLY from trainer-api.py (lines 35-211)
        class InferencePipeline:
            def __init__(self, checkpoint_dir: str, device: str = "cuda"):
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
                logger.info(f"Initializing model on device: {self.device}")

                # Load the ACEStepPipeline
                self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)

                # Initialize components EXACTLY like trainer-api.py
                self.transformers = self.acestep_pipeline.ace_step_transformer.float().to(self.device).eval()
                self.dcae = self.acestep_pipeline.music_dcae.float().to(self.device).eval()
                self.text_encoder_model = self.acestep_pipeline.text_encoder_model.float().to(self.device).eval()
                self.text_tokenizer = self.acestep_pipeline.text_tokenizer
                self.lyric_tokenizer = VoiceBpeTokenizer()  # NEW: Add lyric tokenizer for lyrics support

                # Ensure no gradients are computed
                self.transformers.requires_grad_(False)
                self.dcae.requires_grad_(False)
                self.text_encoder_model.requires_grad_(False)

                # Initialize scheduler EXACTLY like trainer-api.py
                self.scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=3.0,
                )
                
            def get_text_embeddings(self, texts, device, text_max_length=256):
                """EXACTLY from trainer-api.py lines 38-87"""
                text_inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=text_max_length,
                    truncation=True,
                    padding=True,
                )
                input_ids = text_inputs.input_ids.to(device)
                attention_mask = text_inputs.attention_mask.to(device)
                
                with torch.no_grad():
                    text_embeddings = self.text_encoder(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state
                
                return text_embeddings, attention_mask

            def tokenize_lyrics(self, lyrics: str):
                """Tokenize lyrics EXACTLY like official implementation"""
                import re
                
                # Structure pattern for [verse], [chorus], etc.
                structure_pattern = re.compile(r"\[.*?\]")
                
                lines = lyrics.split("\n")
                lyric_token_idx = [261]  # Start token like official code
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        lyric_token_idx += [2]  # Line break token
                        continue

                    # Detect language (simplified - default to English for serverless)
                    lang = "en"

                    try:
                        # Handle structure markers like [Verse], [Chorus] 
                        if structure_pattern.match(line):
                            token_idx = self.lyric_tokenizer.encode(line, "en")
                        else:
                            token_idx = self.lyric_tokenizer.encode(line, lang)
                        
                        lyric_token_idx = lyric_token_idx + token_idx + [2]  # Add line break
                    except Exception as e:
                        logger.warning(f"Lyric tokenization error for line '{line}': {e}")
                        # Skip problematic lines
                        continue

                return lyric_token_idx

            def diffusion_process(self, duration, encoder_text_hidden_states, text_attention_mask,
                                speaker_embds, lyric_token_ids, lyric_mask, random_generator=None,
                                infer_steps=60, guidance_scale=15.0, omega_scale=10.0):
                """EXACTLY from trainer-api.py lines 88-153"""
                do_classifier_free_guidance = guidance_scale > 1.0
                device = encoder_text_hidden_states.device
                dtype = encoder_text_hidden_states.dtype
                bsz = encoder_text_hidden_states.shape[0]

                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler, num_inference_steps=infer_steps, device=device
                )

                frame_length = int(duration * 44100 / 512 / 8)
                target_latents = randn_tensor(
                    shape=(bsz, 8, 16, frame_length),
                    generator=random_generator,
                    device=device,
                    dtype=dtype,
                )
                attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)

                if do_classifier_free_guidance:
                    attention_mask = torch.cat([attention_mask] * 2, dim=0)
                    encoder_text_hidden_states = torch.cat(
                        [encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)],
                        0,
                    )
                    text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)
                    speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)], 0)
                    lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0)
                    lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

                momentum_buffer = MomentumBuffer()

                for t in timesteps:
                    latent_model_input = (
                        torch.cat([target_latents] * 2) if do_classifier_free_guidance else target_latents
                    )
                    timestep = t.expand(latent_model_input.shape[0])
                    with torch.no_grad():
                        noise_pred = self.transformers(
                            hidden_states=latent_model_input,
                            attention_mask=attention_mask,
                            encoder_text_hidden_states=encoder_text_hidden_states,
                            text_attention_mask=text_attention_mask,
                            speaker_embeds=speaker_embds,
                            lyric_token_idx=lyric_token_ids,
                            lyric_mask=lyric_mask,
                            timestep=timestep,
                        ).sample

                    if do_classifier_free_guidance:
                        noise_pred_with_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = apg_forward(
                            pred_cond=noise_pred_with_cond,
                            pred_uncond=noise_pred_uncond,
                            guidance_scale=guidance_scale,
                            momentum_buffer=momentum_buffer,
                        )

                    target_latents = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=target_latents,
                        omega=omega_scale,
                    )[0]

                return target_latents
                
            def generate_audio(self, prompt: str, duration: int, infer_steps: int,
                             guidance_scale: float, omega_scale: float, seed: Optional[int],
                             lyrics: str = "", guidance_scale_lyric: float = 0.0, 
                             guidance_scale_text: float = 0.0, use_erg_lyric: bool = True):
                """EXACTLY from trainer-api.py lines 167-217 WITH lyrics support"""
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
                    [prompt], self.device
                )

                # Prepare speaker embeddings and lyrics 
                bsz = 1
                speaker_embds = torch.zeros(bsz, 512, device=self.device, dtype=encoder_text_hidden_states.dtype)
                
                # Process lyrics if provided - EXACTLY like official implementation
                if lyrics and lyrics.strip():
                    lyric_token_idx = self.tokenize_lyrics(lyrics)
                    lyric_mask = [1] * len(lyric_token_idx)
                    
                    # Pad or truncate to fixed length (4096 max like in official code)
                    max_lyric_length = 4096
                    if len(lyric_token_idx) > max_lyric_length:
                        lyric_token_idx = lyric_token_idx[:max_lyric_length]
                        lyric_mask = lyric_mask[:max_lyric_length]
                    
                    # Convert to tensors and pad to batch
                    lyric_token_ids = torch.tensor(lyric_token_idx, device=self.device, dtype=torch.long).unsqueeze(0)
                    lyric_mask = torch.tensor(lyric_mask, device=self.device, dtype=torch.long).unsqueeze(0)
                else:
                    # Empty lyrics (instrumental)
                    lyric_token_ids = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)
                    lyric_mask = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)                # Run diffusion process
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

                # Decode latents to audio - EXACTLY trainer-api.py line 199
                audio_lengths = torch.tensor([int(duration * 44100)], device=self.device)
                sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)

                # Save audio - EXACTLY trainer-api.py lines 204-207
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "generated_audio"
                os.makedirs(output_dir, exist_ok=True)
                audio_path = f"{output_dir}/generated_{timestamp}_{seed}.wav"
                torchaudio.save(audio_path, pred_wavs.float().cpu(), sr)

                return audio_path, sr, seed
        
        # Initialize exactly like trainer-api.py startup
        checkpoint_dir = "/opt/ACE-Step"
        model = InferencePipeline(checkpoint_dir=checkpoint_dir)
        logger.info("Model loaded successfully!")
        
        # Restore directory
        os.chdir(original_dir)
        
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load model: {str(e)}")

def generate_music(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate music from text prompt and optional lyrics
    
    Expected input:
    {
        "prompt": "upbeat electronic music",        // REQUIRED: Music description
        "lyrics": "song lyrics with structure tags", // OPTIONAL: Song lyrics
        "duration": 30.0                            // OPTIONAL: Audio length in seconds (1-240)
    }
    """
    
    start_time = time.time()
    
    try:
        # Parse input parameters - keeping it simple with just 3 parameters
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}
        
        lyrics = job_input.get("lyrics", "")  # Optional lyrics support
        duration = job_input.get("duration", 30.0)  # Default 30 seconds
        
        # Use sensible defaults for internal parameters
        seed = None  # Always random generation
        guidance_scale = 3.0  # Good default for quality
        guidance_scale_lyric = 0.0 if not lyrics else 2.0  # Auto-enable if lyrics provided
        guidance_scale_text = 0.0  # Keep simple
        use_erg_lyric = True  # Always use ERG for better quality
        num_inference_steps = 50  # Good quality/speed balance
        
        logger.info(f"Generating music: '{prompt}' ({duration}s) with lyrics: {bool(lyrics)}")
        
        # Load model (will use cache if available)
        pipeline = load_model()
        
        # Generate music using ACEStepPipeline with lyrics support
        audio_path, sample_rate, used_seed = pipeline.generate_audio(
            prompt=prompt,
            duration=int(duration),  # Duration should be int
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            omega_scale=1.0,  # Default omega scale
            seed=seed,
            lyrics=lyrics,                          # NEW: Pass lyrics
            guidance_scale_lyric=guidance_scale_lyric,  # NEW: Lyric guidance
            guidance_scale_text=guidance_scale_text,    # NEW: Text guidance  
            use_erg_lyric=use_erg_lyric,               # NEW: ERG lyric setting
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
            "lyrics": lyrics,                       # NEW: Include lyrics in response
            "duration": duration,
            "seed": used_seed,
            "sample_rate": sample_rate,
            "guidance_scale_lyric": guidance_scale_lyric,  # NEW: Include lyric guidance used
            "guidance_scale_text": guidance_scale_text,    # NEW: Include text guidance used
            "use_erg_lyric": use_erg_lyric,               # NEW: Include ERG lyric setting used
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
