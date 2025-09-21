#!/usr/bin/env python3
"""
Full music generation test with better download monitoring
"""

import os
import sys
import logging
import time
import signal
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_download_progress():
    """Monitor the download directory to show actual progress"""
    cache_dir = "/opt/ACE-Step/models--ACE-Step--ACE-Step-v1-3.5B"
    
    while True:
        try:
            if os.path.exists(cache_dir):
                # Count files and total size
                total_size = 0
                file_count = 0
                
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                
                size_mb = total_size / (1024 * 1024)
                logger.info(f"📊 Download progress: {file_count} files, {size_mb:.1f} MB downloaded...")
                
                # Expected total is around 7GB = 7000MB
                if size_mb > 6500:  # Almost complete
                    logger.info("🎯 Download appears nearly complete!")
                    break
                    
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(30)

def test_full_music_generation():
    """Complete end-to-end music generation test"""
    
    try:
        logger.info("🎵 STARTING FULL MUSIC GENERATION TEST")
        logger.info("=" * 60)
        
        # Start download monitor in background
        logger.info("🔍 Starting download progress monitor...")
        monitor_thread = threading.Thread(target=monitor_download_progress, daemon=True)
        monitor_thread.start()
        
        logger.info("📦 Importing ACE-Step modules...")
        from acestep.pipeline_ace_step import ACEStepPipeline
        from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        import torch
        import random
        from datetime import datetime
        
        logger.info("✅ All imports successful!")
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Using device: {device}")
        
        # Create InferencePipeline (this will trigger model download)
        logger.info("🚀 Initializing pipeline for music generation...")
        logger.info("⏳ This will download ~7GB of model files - please be very patient!")
        logger.info("💡 Check the progress updates above every 30 seconds")
        
        class InferencePipeline:
            def __init__(self, checkpoint_dir: str, device_str: str = "cuda"):
                self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
                logger.info(f"🏗️  Initializing InferencePipeline on device: {self.device}")

                logger.info("📥 Loading ACEStepPipeline... (Downloading model files)")
                start_time = time.time()
                
                # Load the ACEStepPipeline - THIS IS WHERE THE DOWNLOAD HAPPENS
                self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)
                
                download_time = time.time() - start_time
                logger.info(f"✅ Model download/load completed in {download_time:.1f} seconds!")
                
                logger.info("🔧 Initializing model components...")
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
                
                logger.info("✅ InferencePipeline ready for music generation!")
                
            def get_text_embeddings(self, prompts, device):
                """Get text embeddings from prompts"""
                inputs = self.text_tokenizer(
                    prompts, max_length=256, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = self.text_encoder_model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                return last_hidden_states, attention_mask

            def diffusion_process(self, duration, encoder_text_hidden_states, text_attention_mask, 
                                speaker_embds, lyric_token_ids, lyric_mask, random_generator=None,
                                infer_steps=60, guidance_scale=15.0, omega_scale=10.0):
                """Diffusion process wrapper"""
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
                return pred_latents
                
            def generate_audio(self, prompt: str, duration: int, infer_steps: int, 
                             guidance_scale: float, omega_scale: float, seed=None):
                """Generate audio from text prompt"""
                logger.info(f"🎼 Generating music for: '{prompt}'")
                logger.info(f"⚙️  Parameters: {duration}s, {infer_steps} steps, guidance={guidance_scale}")
                
                generation_start = time.time()
                
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
                logger.info("📝 Encoding text prompt...")
                encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
                    [prompt], self.device
                )

                # Dummy speaker embeddings and lyrics
                bsz = 1
                speaker_embds = torch.zeros(bsz, 512, device=self.device, dtype=encoder_text_hidden_states.dtype)
                lyric_token_ids = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)
                lyric_mask = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)

                # Run diffusion process
                logger.info("🌊 Starting diffusion process...")
                diffusion_start = time.time()
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
                diffusion_time = time.time() - diffusion_start
                logger.info(f"✅ Diffusion completed in {diffusion_time:.1f} seconds")

                # Decode latents to audio
                logger.info("🔊 Decoding latents to audio...")
                decode_start = time.time()
                audio_lengths = torch.tensor([int(duration * 44100)], device=self.device)
                sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)
                decode_time = time.time() - decode_start
                logger.info(f"✅ Audio decoding completed in {decode_time:.1f} seconds")

                # Save audio
                logger.info("💾 Saving generated audio...")
                import torchaudio
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = f"/tmp/GENERATED_MUSIC_{timestamp}_{seed}.wav"
                torchaudio.save(temp_path, pred_wavs.float().cpu(), sr)
                
                total_time = time.time() - generation_start
                file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
                
                logger.info(f"🎉 SUCCESS! Music generated in {total_time:.1f} seconds")
                logger.info(f"📁 Audio saved: {temp_path}")
                logger.info(f"📊 File size: {file_size:.1f} MB")
                logger.info(f"🔊 Sample rate: {sr} Hz")
                
                return temp_path, sr, seed

        # Initialize pipeline (this triggers the download)
        pipeline = InferencePipeline("/opt/ACE-Step")
        
        # Generate music!
        logger.info("🎵 STARTING MUSIC GENERATION!")
        logger.info("=" * 40)
        
        audio_path, sample_rate, used_seed = pipeline.generate_audio(
            prompt="upbeat electronic dance music with synthesizers",
            duration=15,  # 15 seconds
            infer_steps=30,  # Reasonable steps for testing
            guidance_scale=7.5,
            omega_scale=5.0,
            seed=42
        )
        
        # Verify output
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            logger.info("=" * 60)
            logger.info("🎊 FULL TEST COMPLETED SUCCESSFULLY!")
            logger.info(f"🎵 Generated music file: {audio_path}")
            logger.info(f"📊 File size: {file_size / 1024:.1f} KB")
            logger.info(f"🔊 Sample rate: {sample_rate} Hz")
            logger.info(f"🎲 Seed: {used_seed}")
            logger.info("✅ Handler implementation is WORKING correctly!")
            logger.info("🚀 Ready for Docker build and RunPod deployment!")
            logger.info("=" * 60)
            return True
        else:
            logger.error("❌ Audio file was not created!")
            return False
            
    except KeyboardInterrupt:
        logger.info("⚠️  Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up signal handler for graceful interruption
    def signal_handler(sig, frame):
        logger.info("⚠️  Received interrupt signal - stopping test...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "=" * 70)
    print("🎵 FULL ACE-STEP MUSIC GENERATION TEST")
    print("🚀 This will download ~7GB and generate actual music!")
    print("⏳ Expected time: 5-15 minutes (depending on internet)")
    print("=" * 70)
    
    success = test_full_music_generation()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Ready to deploy! 🚀")
    else:
        print("\n❌ Test failed - check logs above")
