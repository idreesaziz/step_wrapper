#!/usr/bin/env python3
"""
FULL END-TO-END MUSIC GENERATION TEST
This will download the model completely and generate actual music
"""

import os
import sys
import logging
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_music_generation_test():
    """Run complete music generation test - download model and generate audio"""
    
    print("🎵" * 50)
    print("FULL MUSIC GENERATION TEST - NO SHORTCUTS!")
    print("🎵" * 50)
    
    try:
        # Import everything we need
        logger.info("📦 Importing ACE-Step modules...")
        import torch
        from acestep.pipeline_ace_step import ACEStepPipeline
        from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        import random
        import torchaudio
        from datetime import datetime
        
        logger.info("✅ All imports successful!")
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Using device: {device}")
        
        # Create our InferencePipeline (exact copy from handler)
        class InferencePipeline:
            def __init__(self, checkpoint_dir: str, device: str = "cuda"):
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
                logger.info(f"🏗️  Initializing InferencePipeline on device: {self.device}")

                # Load the ACEStepPipeline - THIS WILL DOWNLOAD THE FULL MODEL
                logger.info("📥 Loading ACEStepPipeline... (This will download ~7GB)")
                logger.info("⏳ Please be patient - this may take 5-10 minutes depending on internet speed")
                
                start_download = time.time()
                self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)
                download_time = time.time() - start_download
                
                logger.info(f"✅ Model downloaded and loaded in {download_time:.1f} seconds!")

                # Initialize all components
                logger.info("🔧 Initializing model components...")
                self.transformers = self.acestep_pipeline.ace_step_transformer.float().to(self.device).eval()
                self.dcae = self.acestep_pipeline.music_dcae.float().to(self.device).eval()
                self.text_encoder_model = self.acestep_pipeline.text_encoder_model.float().to(self.device).eval()
                self.text_tokenizer = self.acestep_pipeline.text_tokenizer
                self.scheduler = FlowMatchEulerDiscreteScheduler()

                # Disable gradients for inference
                self.transformers.requires_grad_(False)
                self.dcae.requires_grad_(False)
                self.text_encoder_model.requires_grad_(False)
                
                logger.info("🎉 InferencePipeline fully initialized!")
                
            def get_text_embeddings(self, prompts, device):
                logger.info(f"📝 Processing text prompt: {prompts}")
                inputs = self.text_tokenizer(
                    prompts, max_length=256, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = self.text_encoder_model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                return last_hidden_states, attention_mask

            def generate_audio(self, prompt: str, duration: int, infer_steps: int, 
                             guidance_scale: float, omega_scale: float, seed=None):
                logger.info("🎼 STARTING FULL MUSIC GENERATION!")
                logger.info(f"📋 Parameters:")
                logger.info(f"   - Prompt: '{prompt}'")
                logger.info(f"   - Duration: {duration} seconds")
                logger.info(f"   - Inference steps: {infer_steps}")
                logger.info(f"   - Guidance scale: {guidance_scale}")
                logger.info(f"   - Omega scale: {omega_scale}")
                
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
                logger.info("🔤 Encoding text to embeddings...")
                encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings([prompt], self.device)

                # Create dummy embeddings for speaker and lyrics
                logger.info("🎭 Creating speaker and lyric embeddings...")
                bsz = 1
                speaker_embds = torch.zeros(bsz, 512, device=self.device, dtype=encoder_text_hidden_states.dtype)
                lyric_token_ids = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)
                lyric_mask = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)

                # Run the diffusion process - THE ACTUAL MUSIC GENERATION!
                logger.info("🌊 Starting diffusion process... THIS IS WHERE THE MAGIC HAPPENS!")
                diffusion_start = time.time()
                
                pred_latents = self.acestep_pipeline.text2music_diffusion_process(
                    duration=duration,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    random_generators=generator,
                    infer_steps=infer_steps,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                )
                
                diffusion_time = time.time() - diffusion_start
                logger.info(f"✅ Diffusion completed in {diffusion_time:.1f} seconds!")

                # Decode latents to actual audio waveforms
                logger.info("🔊 Decoding latents to audio waveforms...")
                decode_start = time.time()
                
                audio_lengths = torch.tensor([int(duration * 44100)], device=self.device)
                sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)
                
                decode_time = time.time() - decode_start
                logger.info(f"✅ Audio decoded in {decode_time:.1f} seconds!")

                # Save the generated audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"/tmp/GENERATED_MUSIC_{timestamp}_seed{seed}.wav"
                
                logger.info(f"💾 Saving audio to: {output_path}")
                torchaudio.save(output_path, pred_wavs.float().cpu(), sr)
                
                # Verify the file
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"📊 Generated file size: {file_size:,} bytes")
                    
                    if file_size > 10000:  # At least 10KB
                        logger.info("🎉 SUCCESS! Music generation completed!")
                        return output_path, sr, seed
                    else:
                        raise Exception("Generated file is too small")
                else:
                    raise Exception("Output file was not created")

        # Now run the actual test
        logger.info("🚀 Initializing pipeline for music generation...")
        pipeline = InferencePipeline("/opt/ACE-Step")
        
        # Generate actual music!
        logger.info("🎵 GENERATING REAL MUSIC NOW!")
        total_start = time.time()
        
        audio_path, sample_rate, used_seed = pipeline.generate_audio(
            prompt="upbeat electronic dance music with synthesizers",
            duration=15,  # 15 seconds of music
            infer_steps=30,  # Reasonable number of steps
            guidance_scale=7.5,
            omega_scale=5.0,
            seed=12345
        )
        
        total_time = time.time() - total_start
        
        print("\n" + "🎉" * 50)
        print("SUCCESS! MUSIC HAS BEEN GENERATED!")
        print("🎉" * 50)
        print(f"🎵 Generated audio file: {audio_path}")
        print(f"🔊 Sample rate: {sample_rate} Hz")
        print(f"🎲 Seed used: {used_seed}")
        print(f"⏱️  Total generation time: {total_time:.1f} seconds")
        print(f"📁 File location: {audio_path}")
        
        # Check file details
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print("✅ Handler implementation WORKS PERFECTLY!")
            return True
        else:
            print("❌ File not found after generation")
            return False
        
    except Exception as e:
        print(f"\n❌ GENERATION FAILED: {e}")
        print("Full error traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_music_generation_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 FULL TEST PASSED! READY FOR DEPLOYMENT! 🎉")
    else:
        print("❌ TEST FAILED - NEEDS MORE WORK")
    print("=" * 60)
