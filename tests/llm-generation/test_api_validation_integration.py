#!/usr/bin/env python3
"""
Test API Validation Integration in Sequential Pipeline

Quick test to verify API validation is working in the pipeline.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'api-mcp'))

from sequential_pipeline import SequentialPipeline

# Test code with known API hallucination (from Query 5)
test_code_with_hallucination = """
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage

stencil = vtkImageStencilToImage()
stencil.SetOutputWholeExtent([0, 10, 0, 10, 0, 10])  # ❌ Method doesn't exist!
stencil.Update()
"""

# Test code that is valid
test_code_valid = """
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

mapper = vtkPolyDataMapper()
actor = vtkActor()
actor.SetMapper(mapper)
"""

def test_api_validation():
    """Test that API validation is integrated"""
    print("="*80)
    print("Testing API Validation Integration")
    print("="*80)
    
    # Initialize pipeline with API validation enabled
    print("\n1. Initializing pipeline with API validation...")
    pipeline = SequentialPipeline(
        use_llm_decomposition=False,  # Don't need LLM for this test
        enable_validation=False,  # Disable LLM validation
        enable_api_validation=True  # Enable API validation
    )
    
    if pipeline.api_validator:
        print("   ✅ API validator loaded successfully")
    else:
        print("   ❌ API validator not loaded")
        return False
    
    # Test with hallucinated method
    print("\n2. Testing code with hallucinated method...")
    print("   Code: stencil.SetOutputWholeExtent([...])  # Should fail")
    
    code, is_valid, errors = pipeline._validate_api_with_mcp(test_code_with_hallucination)
    
    if not is_valid and len(errors) > 0:
        print(f"   ✅ Correctly detected {len(errors)} error(s):")
        for error in errors[:2]:  # Show first 2 errors
            print(f"      - {error}")
    else:
        print("   ❌ Should have detected hallucinated method")
        return False
    
    # Test with valid code
    print("\n3. Testing valid code...")
    print("   Code: actor.SetMapper(mapper)  # Should pass")
    
    code, is_valid, errors = pipeline._validate_api_with_mcp(test_code_valid)
    
    if is_valid and len(errors) == 0:
        print("   ✅ Correctly validated valid code")
    else:
        print(f"   ❌ Should have passed validation, got {len(errors)} errors")
        return False
    
    print("\n" + "="*80)
    print("✅ API Validation Integration Test PASSED!")
    print("="*80)
    print("\nAPI validation is successfully integrated into the pipeline.")
    print("\n📋 Validation Flow:")
    print("  1. API validator detects hallucinations")
    print("  2. Errors are logged with suggestions")
    print("  3. Errors passed to LLM validator context")
    print("  4. LLM validator fixes API errors + other issues")
    print("  5. API validator re-validates after fixes")
    print("  6. Final code is hallucination-free ✅")
    print("\nNext steps:")
    print("  - Run full evaluation on queries 1-10")
    print("  - Verify Query 5 hallucination gets fixed by LLM")
    print("  - Measure improvement in success rate")
    
    return True


if __name__ == '__main__':
    try:
        success = test_api_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
