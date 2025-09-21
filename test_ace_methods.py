#!/usr/bin/env python3
"""
Test the available methods on ACEStepPipeline
"""

try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    
    # Initialize pipeline
    pipeline = ACEStepPipeline("/opt/ACE-Step")
    
    # List all public methods
    methods = [method for method in dir(pipeline) if not method.startswith('_')]
    print("Available methods on ACEStepPipeline:")
    for method in sorted(methods):
        print(f"  - {method}")
        
    # Check if text2music_diffusion_process exists and get its signature
    if hasattr(pipeline, 'text2music_diffusion_process'):
        import inspect
        sig = inspect.signature(pipeline.text2music_diffusion_process)
        print(f"\ntext2music_diffusion_process signature: {sig}")
    else:
        print("\ntext2music_diffusion_process method not found")
        
    # Look for other potential generation methods
    generation_methods = [m for m in methods if 'generate' in m.lower() or 'diffusion' in m.lower() or 'infer' in m.lower()]
    print(f"\nPotential generation methods: {generation_methods}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
