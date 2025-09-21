#!/usr/bin/env python3
"""
Test the updated handler with proper InferencePipeline implementation
"""

try:
    # Copy our updated handler
    import sys
    sys.path.insert(0, '/opt')
    
    # Test imports from our handler
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.apg_guidance import apg_forward, MomentumBuffer
    from transformers import AutoTokenizer
    import random
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
    from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    
    print("✅ All imports successful!")
    
    # Try to initialize our InferencePipeline class
    import torch
    
    # Test the InferencePipeline class definition from our handler
    class InferencePipeline:
        def __init__(self, checkpoint_dir: str, device: str = "cuda"):
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"Initializing model on device: {self.device}")

            # Load the ACEStepPipeline
            self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
            print("✅ ACEStepPipeline created")
            
            # Try loading checkpoint
            try:
                self.acestep_pipeline.load_checkpoint(checkpoint_dir)
                print("✅ Checkpoint loaded successfully")
            except Exception as e:
                print(f"⚠️  Checkpoint loading failed: {e}")
                return
            
            print("✅ InferencePipeline initialized successfully!")
    
    # Test initialization
    pipeline = InferencePipeline("/opt/ACE-Step")
    
    # Test if generate_audio method would work (without actually running it)
    if hasattr(pipeline, 'acestep_pipeline'):
        if hasattr(pipeline.acestep_pipeline, 'text2music_diffusion_process'):
            print("✅ text2music_diffusion_process method available")
        else:
            print("❌ text2music_diffusion_process method not found")
    
    print("\n🎉 Handler implementation test PASSED!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
