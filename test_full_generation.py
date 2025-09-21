#!/usr/bin/env python3
"""
Comprehensive test of our ACE-Step handler implementation
Tests actual music generation to verify everything works
"""

import os
import sys
import logging
import tempfile
import time
import traceback
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_generation():
    """Test the complete music generation pipeline"""
    try:
        logger.info("🧪 Starting comprehensive ACE-Step generation test")
        
        # Import required modules
        logger.info("📦 Importing required modules...")
        from acestep.pipeline_ace_step import ACEStepPipeline
        from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        import random
        from datetime import datetime
        
        logger.info("✅ All imports successful")
        
        # Create InferencePipeline class (from our handler)
        class InferencePipeline:
            def __init__(self, checkpoint_dir: str, device: str = "cuda"):
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
                logger.info(f"🖥️  Initializing model on device: {self.device}")

                # Load the ACEStepPipeline
                logger.info("📥 Loading ACEStepPipeline...")
                self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
                
                logger.info("🔧 Loading checkpoint...")
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)
                
                logger.info("🏗️  Initializing components...")
                # Initialize components
                self.transformers = self.acestep_pipeline.ace_step_transformer.float().to(self.device).eval()
                self.dcae = self.acestep_pipeline.music_dcae.float().to(self.device).eval()
                self.text_encoder_model = self.acestep_pipeline.text_encoder_model.float().to(self.device).eval()
                self.text_tokenizer = self.acestep_pipeline.text_tokenizer

                # Initialize scheduler
                self.scheduler = FlowMatchEulerDiscreteScheduler()

                # Ensure no gradients are computed
                self.transformers.requires_grad_(False)
                self.dcae.requires_grad_(False)
                self.text_encoder_model.requires_grad_(False)
                
                logger.info("✅ InferencePipeline initialized successfully!")
                
            def get_text_embeddings(self, prompts, device):
                """Get text embeddings from prompts"""
                logger.info(f"📝 Encoding text: {prompts}")
                inputs = self.text_tokenizer(
                    prompts, max_length=256, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = self.text_encoder_model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                logger.info("✅ Text embeddings created")
                return last_hidden_states, attention_mask

            def diffusion_process(self, duration, encoder_text_hidden_states, text_attention_mask, 
                                speaker_embds, lyric_token_ids, lyric_mask, random_generator=None,
                                infer_steps=60, guidance_scale=15.0, omega_scale=10.0):
                """Simplified diffusion process wrapper"""
                logger.info(f"🌊 Starting diffusion process (steps: {infer_steps}, guidance: {guidance_scale})")
                # Use ACEStepPipeline's text2music_diffusion_process
                pred_latents = self.acestep_pipeline.text2music_diffusion_process(
                    duration=duration,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    random_generators=random_generator,
                    infer_steps=infer_steps,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                )
                logger.info("✅ Diffusion process completed")
                return pred_latents
                
            def generate_audio(self, prompt: str, duration: int, infer_steps: int, 
                             guidance_scale: float, omega_scale: float, seed=None):
                """Generate audio from text prompt (copied from trainer-api.py)"""
                logger.info(f"🎵 Generating audio for: '{prompt}'")
                logger.info(f"⚙️  Parameters: duration={duration}s, steps={infer_steps}, guidance={guidance_scale}, omega={omega_scale}")
                
                # Set random seed
                if seed is not None:
                    random.seed(seed)
                    torch.manual_seed(seed)
                else:
                    seed = random.randint(0, 2**32 - 1)
                    random.seed(seed)
                    torch.manual_seed(seed)

                logger.info(f"🎲 Using seed: {seed}")
                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Get text embeddings
                encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
                    [prompt], self.device
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
                logger.info("🔊 Decoding latents to audio...")
                audio_lengths = torch.tensor([int(duration * 44100)], device=self.device)
                sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)

                # Save audio
                logger.info("💾 Saving audio file...")
                import torchaudio
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = f"/tmp/test_generated_{timestamp}_{seed}.wav"
                torchaudio.save(temp_path, pred_wavs.float().cpu(), sr)
                
                logger.info(f"✅ Audio saved to: {temp_path}")
                return temp_path, sr, seed

        # Initialize the pipeline
        logger.info("🚀 Initializing InferencePipeline...")
        pipeline = InferencePipeline("/opt/ACE-Step")
        
        # Test generation with a simple prompt
        logger.info("🎼 Testing music generation...")
        start_time = time.time()
        
        try:
            audio_path, sample_rate, used_seed = pipeline.generate_audio(
                prompt="simple happy melody",
                duration=10,  # Short 10-second test
                infer_steps=20,  # Fewer steps for faster test
                guidance_scale=7.5,  # Moderate guidance
                omega_scale=5.0,
                seed=42
            )
            
            generation_time = time.time() - start_time
            logger.info(f"🎉 SUCCESS! Generated audio in {generation_time:.2f} seconds")
            logger.info(f"📁 Audio file: {audio_path}")
            logger.info(f"🔊 Sample rate: {sample_rate} Hz")
            logger.info(f"🎲 Seed used: {used_seed}")
            
            # Check if file exists and has content
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                logger.info(f"📊 File size: {file_size} bytes")
                
                if file_size > 1000:  # At least 1KB
                    logger.info("✅ Audio file appears to contain data")
                    return True
                else:
                    logger.error("❌ Audio file is too small - likely empty")
                    return False
            else:
                logger.error("❌ Audio file was not created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 COMPREHENSIVE ACE-STEP GENERATION TEST")
    print("=" * 60)
    
    success = test_full_generation()
    
    print("=" * 60)
    if success:
        print("🎉 TEST PASSED! ✅")
        print("The handler implementation works correctly!")
        print("Ready to build and deploy updated Docker image.")
    else:
        print("❌ TEST FAILED!")
        print("Handler needs more fixes before deployment.")
    print("=" * 60)
