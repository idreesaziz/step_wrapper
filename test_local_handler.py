#!/usr/bin/env python3
"""
Local test script for the corrected ACE-Step handler
Tests the import and basic functionality before Docker deployment
"""

import sys
import os
import tempfile
import traceback

def test_imports():
    """Test if the corrected imports work"""
    print("Testing imports...")
    try:
        # Test the corrected import path
        from acestep.pipeline_ace_step import ACEStepPipeline
        print("✅ ACEStepPipeline import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test if we can initialize the pipeline"""
    print("Testing pipeline initialization...")
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        
        # Try to initialize (this might fail if no checkpoint, but we want to see the error)
        pipeline = ACEStepPipeline("/opt/ACE-Step")
        print("✅ Pipeline initialization successful")
        return True, pipeline
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False, None

def test_generate_audio_method():
    """Test if the generate_audio method exists and has correct signature"""
    print("Testing generate_audio method...")
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        
        # Check if method exists
        if hasattr(ACEStepPipeline, 'generate_audio'):
            print("✅ generate_audio method exists")
            
            # Try to get method signature
            import inspect
            sig = inspect.signature(ACEStepPipeline.generate_audio)
            print(f"Method signature: {sig}")
            return True
        else:
            print("❌ generate_audio method not found")
            # List available methods
            methods = [method for method in dir(ACEStepPipeline) if not method.startswith('_')]
            print(f"Available methods: {methods}")
            return False
    except Exception as e:
        print(f"❌ Method check failed: {e}")
        return False

def main():
    print("🧪 Testing ACE-Step Handler Locally")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed - ACE-Step not properly installed")
        return
    
    print()
    
    # Test 2: Pipeline initialization
    init_success, pipeline = test_pipeline_initialization()
    if not init_success:
        print("\n⚠️  Pipeline initialization failed, but imports work")
        print("This might be expected if checkpoints aren't downloaded yet")
    
    print()
    
    # Test 3: Method signature
    if not test_generate_audio_method():
        print("\n❌ generate_audio method test failed")
        return
    
    print("\n" + "=" * 50)
    if init_success:
        print("✅ All tests passed! Handler should work correctly")
    else:
        print("⚠️  Imports and methods work, but pipeline needs checkpoints")
        print("This is expected - the Docker container will download them")

if __name__ == "__main__":
    main()
