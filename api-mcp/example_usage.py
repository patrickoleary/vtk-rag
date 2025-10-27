#!/usr/bin/env python3
"""
VTK API Validation - Example Usage

Demonstrates how to use the VTK code validator to catch API hallucinations.
"""

from pathlib import Path
from vtk_api_server import VTKAPIIndex
from vtk_validator import VTKCodeValidator


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def validate_code_example(validator: VTKCodeValidator, code: str, description: str):
    """Validate a code example and print results"""
    print(f"📝 {description}")
    print("-" * 80)
    print("Code:")
    print(code)
    print("-" * 80)
    
    result = validator.validate_code(code)
    
    if result.is_valid:
        print("✅ VALID - No errors found!\n")
    else:
        print("❌ INVALID - Errors detected:")
        print(result.format_errors())
        print()


def main():
    print_section("VTK API Validation - Example Usage")
    
    # Initialize the validator
    print("Loading VTK API index...")
    api_docs_path = Path("../data/raw/vtk-python-docs.jsonl")
    
    if not api_docs_path.exists():
        print(f"❌ Error: API docs not found at {api_docs_path}")
        print("   Please ensure vtk-python-docs.jsonl exists in data/raw/")
        return
    
    api_index = VTKAPIIndex(api_docs_path)
    validator = VTKCodeValidator(api_index)
    print(f"✅ Loaded {len(api_index.classes)} VTK classes\n")
    
    # Example 1: Valid Code
    print_section("Example 1: Valid VTK Code")
    valid_code = """
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper

reader = vtkSTLReader()
reader.SetFileName('model.stl')
reader.Update()

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
"""
    validate_code_example(validator, valid_code, "Reading STL and creating mapper")
    
    # Example 2: Method Hallucination (Query 5 Error)
    print_section("Example 2: Method Hallucination (CAUGHT!)")
    hallucinated_method = """
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage

stencil_to_image = vtkImageStencilToImage()
stencil_to_image.SetOutputWholeExtent([0, 10, 0, 10, 0, 10])  # ❌ Method doesn't exist!
stencil_to_image.Update()
"""
    validate_code_example(
        validator,
        hallucinated_method,
        "LLM hallucinated SetOutputWholeExtent() method"
    )
    
    # Example 3: Wrong Import Module
    print_section("Example 3: Wrong Import Module (CAUGHT!)")
    wrong_import = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper  # ❌ Wrong module!

mapper = vtkPolyDataMapper()
mapper.SetInputData(data)
"""
    validate_code_example(
        validator,
        wrong_import,
        "vtkPolyDataMapper imported from wrong module"
    )
    
    # Example 4: Non-existent Class
    print_section("Example 4: Non-existent Class (CAUGHT!)")
    fake_class = """
from vtkmodules.vtkFiltersCore import vtkImageDataToPolyDataConverter  # ❌ Class doesn't exist!

converter = vtkImageDataToPolyDataConverter()
converter.SetInputData(image_data)
"""
    validate_code_example(
        validator,
        fake_class,
        "LLM invented a class that doesn't exist in VTK"
    )
    
    # Example 5: Multiple Errors
    print_section("Example 5: Multiple Errors (CAUGHT!)")
    multiple_errors = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper  # ❌ Wrong module
from vtkmodules.vtkIOGeometry import vtkFakeReader  # ❌ Class doesn't exist

reader = vtkFakeReader()  # ❌ Non-existent class
reader.SetFileName('test.stl')
reader.Update()

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.EnableCaching()  # ❌ Method doesn't exist
"""
    validate_code_example(
        validator,
        multiple_errors,
        "Code with multiple validation errors"
    )
    
    # Example 6: Correct Import Validation
    print_section("Example 6: Import Validation Examples")
    
    print("Testing import validation...")
    test_imports = [
        ("from vtkmodules.vtkRenderingCore import vtkPolyDataMapper", True),
        ("from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper", False),
        ("from vtkmodules.vtkIOGeometry import vtkSTLReader", True),
        ("from vtkmodules.vtkFiltersCore import vtkSTLReader", False),
    ]
    
    for import_stmt, should_be_valid in test_imports:
        result = validator.validate_code(import_stmt)
        status = "✅" if result.is_valid == should_be_valid else "❌"
        print(f"{status} {import_stmt}")
        if not result.is_valid:
            print(f"   Error: {result.errors[0].message}")
    
    # Summary
    print_section("Summary")
    print("✅ VTK API Validation can detect:")
    print("   1. Method hallucinations (e.g., SetOutputWholeExtent)")
    print("   2. Non-existent classes")
    print("   3. Wrong import modules")
    print("   4. Multiple errors in one code block")
    print()
    print("💡 Integration with pipeline will:")
    print("   - Catch these errors before code execution")
    print("   - Provide helpful error messages with suggestions")
    print()
    print("📖 See README.md for more details and integration plans.")
    print()


if __name__ == '__main__':
    main()
