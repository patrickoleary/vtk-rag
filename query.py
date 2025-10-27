#!/usr/bin/env python3
"""
VTK RAG - Unified Query System

Complete pipeline for VTK documentation queries:
- Code generation with explanations
- API documentation lookup
- Concept explanations
- Data file finding
- Optional visual validation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Add pipeline modules to path
sys.path.append(str(Path(__file__).parent / 'retrieval-pipeline'))
sys.path.append(str(Path(__file__).parent / 'grounding-prompting'))
sys.path.append(str(Path(__file__).parent / 'llm-generation'))
sys.path.append(str(Path(__file__).parent / 'post-processing'))
sys.path.append(str(Path(__file__).parent / 'evaluation'))

from sequential_pipeline import SequentialPipeline
from json_response_processor import JSONResponseProcessor

# Load environment
load_dotenv()


def check_prerequisites(require_visual: bool = False) -> list:
    """Check if prerequisites are met"""
    issues = []
    
    # Check .env
    if not (Path(__file__).parent / '.env').exists():
        issues.append(".env file not found (need LLM API key)")
    
    # Check Qdrant
    try:
        import requests
        response = requests.get('http://localhost:6333/healthz', timeout=2)
        if response.status_code != 200:
            issues.append("Qdrant not responding")
    except:
        issues.append("Qdrant not running (start: docker run -p 6333:6333 qdrant/qdrant)")
    
    # Check collection
    try:
        import requests
        response = requests.get('http://localhost:6333/collections/vtk_docs')
        if response.status_code != 200:
            issues.append("vtk_docs collection missing (run: python build-indexes/build_qdrant_index.py)")
    except:
        pass
    
    # Check Docker if visual testing requested
    if require_visual:
        try:
            import subprocess
            result = subprocess.run(['docker', 'ps'], capture_output=True, timeout=5)
            if result.returncode != 0:
                issues.append("Docker not running (required for --visual-test)")
        except:
            issues.append("Docker not available (required for --visual-test)")
    
    return issues


def query_vtk(
    query: str,
    visual_test: bool = False,
    enrich: bool = False,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete VTK RAG query
    
    Args:
        query: User's question
        visual_test: Execute code in Docker sandbox
        enrich: Use LLM to improve explanations
        output_file: Save JSON result to file
        verbose: Print progress
    
    Returns:
        Dict with response and metadata
    """
    
    if verbose:
        print("=" * 80)
        print("VTK RAG - UNIFIED QUERY SYSTEM")
        print("=" * 80)
        print(f"\n📝 Query: {query}\n")
    
    # Initialize pipeline with API validation
    pipeline = SequentialPipeline(
        use_llm_decomposition=True,
        enable_validation=False,
        enable_api_validation=True  # Catch VTK API hallucinations
    )
    
    if verbose:
        print("🔍 Step 1: Query Processing")
        print("-" * 80)
    
    # Process query (handles ALL query types)
    response = pipeline.process_query(query)
    
    # Detect query type
    response_type = response.get('response_type', 'unknown')
    content_type = response.get('content_type', 'unknown')
    
    if verbose:
        print(f"  ✓ Response Type: {response_type}")
        print(f"  ✓ Content Type: {content_type}")
        
        if content_type == 'code' and 'steps' in response:
            print(f"  → Multi-step code generation ({len(response.get('steps', []))} steps)")
        elif content_type == 'api':
            print(f"  → API documentation lookup")
        elif content_type == 'explanation':
            print(f"  → Concept explanation")
        
        citations = response.get('citations', [])
        if citations:
            print(f"  ✓ Citations: {len(citations)} sources")
    
    # Optional: LLM enrichment
    if enrich and content_type == 'code':
        if verbose:
            print("\n✨ Step 2: LLM Enrichment")
            print("-" * 80)
        
        processor = JSONResponseProcessor()
        chunks = getattr(pipeline, 'last_retrieved_chunks', [])
        
        try:
            response = processor.enrich_with_llm(
                response=response,
                documentation_chunks=chunks,
                llm_client=pipeline.llm_client
            )
            
            if verbose and response.get('_enrichment', {}).get('was_enriched'):
                print(f"  ✓ Explanation enhanced")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Enrichment failed: {e}")
    
    # Optional: Visual testing
    if visual_test and content_type == 'code' and response.get('code'):
        if verbose:
            print(f"\n🖼️  Step 3: Visual Validation")
            print("-" * 80)
        
        try:
            from visual_evaluator import VisualEvaluator
            
            evaluator = VisualEvaluator(
                enable_execution=True,
                data_dir=Path('evaluation/data'),
                create_baselines=False,
                timeout=30
            )
            
            # Execute code
            exec_result = evaluator.execute_code(
                response['code'],
                query_id=0,
                create_baseline=False
            )
            
            response['_visual_validation'] = {
                'execution_success': exec_result.success,
                'execution_time': exec_result.execution_time,
                'has_visual_output': exec_result.has_visual_output,
                'error': exec_result.error if not exec_result.success else None
            }
            
            if verbose:
                if exec_result.success:
                    print(f"  ✅ Code executed successfully ({exec_result.execution_time:.2f}s)")
                    if exec_result.has_visual_output:
                        print(f"  ✅ Visual output generated")
                else:
                    print(f"  ❌ Execution failed: {exec_result.error}")
        
        except ImportError:
            if verbose:
                print(f"  ⚠ Visual testing unavailable (missing dependencies)")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Visual testing failed: {e}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("✅ COMPLETE")
        print("=" * 80)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(response, f, indent=2)
        if verbose:
            print(f"\n💾 Saved to: {output_file}")
    
    return response


def format_response(response: Dict) -> str:
    """Format response for display"""
    lines = []
    
    response_type = response.get('response_type', 'unknown')
    content_type = response.get('content_type', 'unknown')
    
    lines.append("\n" + "=" * 80)
    lines.append(f"RESPONSE: {response_type.upper()} ({content_type.upper()})")
    lines.append("=" * 80)
    lines.append("")
    
    # CODE
    if 'code' in response and response['code']:
        lines.append("📝 CODE:")
        lines.append("-" * 80)
        lines.append("```python")
        lines.append(response['code'])
        lines.append("```")
        lines.append("")
    
    # EXPLANATION
    if 'explanation' in response and response['explanation']:
        lines.append("📖 EXPLANATION:")
        lines.append("-" * 80)
        lines.append(response['explanation'])
        lines.append("")
    
    # DATA FILES
    if 'data_files' in response and response['data_files']:
        lines.append("📁 DATA FILES:")
        lines.append("-" * 80)
        for df in response['data_files']:
            if isinstance(df, dict):
                lines.append(f"  • {df.get('filename', 'unknown')}")
                if df.get('url'):
                    lines.append(f"    URL: {df['url']}")
            else:
                lines.append(f"  • {df}")
        lines.append("")
    
    # IMAGE URL
    if 'image_url' in response and response['image_url']:
        lines.append("🖼️  IMAGE:")
        lines.append("-" * 80)
        lines.append(f"  {response['image_url']}")
        lines.append("")
    
    # API DOCS
    if content_type == 'api':
        if 'class_name' in response:
            lines.append(f"📚 CLASS: {response['class_name']}")
            lines.append("-" * 80)
        if 'methods' in response:
            lines.append(f"\nMethods: {len(response['methods'])}")
            for method in response.get('methods', [])[:5]:
                lines.append(f"  • {method}")
            if len(response.get('methods', [])) > 5:
                lines.append(f"  ... and {len(response['methods']) - 5} more")
        lines.append("")
    
    # CITATIONS
    citations = response.get('citations', [])
    if citations:
        lines.append("🔗 CITATIONS:")
        lines.append("-" * 80)
        for cite in citations[:10]:
            lines.append(f"  • {cite}")
        if len(citations) > 10:
            lines.append(f"  ... and {len(citations) - 10} more")
        lines.append("")
    
    # VISUAL VALIDATION
    if '_visual_validation' in response:
        vv = response['_visual_validation']
        lines.append("🖼️  VISUAL VALIDATION:")
        lines.append("-" * 80)
        lines.append(f"  Execution: {'✅ SUCCESS' if vv['execution_success'] else '❌ FAILED'}")
        if vv['execution_success']:
            lines.append(f"  Time: {vv['execution_time']:.2f}s")
            lines.append(f"  Visual Output: {'✅ Yes' if vv['has_visual_output'] else '❌ No'}")
        else:
            lines.append(f"  Error: {vv['error']}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="VTK RAG - Unified query system for VTK documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple code query
  python query.py "How do I create a cylinder in VTK?"
  
  # With visual testing
  python query.py "Create a red sphere" --visual-test
  
  # API documentation
  python query.py "What methods does vtkPolyDataMapper have?"
  
  # Save to file
  python query.py "Show me a cone example" --output result.json
  
  # Enrich explanations with LLM
  python query.py "Create a visualization pipeline" --enrich
        """
    )
    
    parser.add_argument('query', help='Your VTK question')
    parser.add_argument('--visual-test', action='store_true',
                       help='Execute code in Docker sandbox (requires Docker)')
    parser.add_argument('--enrich', action='store_true',
                       help='Use LLM to enhance explanations')
    parser.add_argument('--output', '-o', metavar='FILE',
                       help='Save JSON response to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (only show result)')
    
    args = parser.parse_args()
    
    # Check prerequisites
    issues = check_prerequisites(require_visual=args.visual_test)
    if issues:
        print("⚠️  Prerequisites not met:", file=sys.stderr)
        for issue in issues:
            print(f"  • {issue}", file=sys.stderr)
        return 1
    
    # Run query
    try:
        response = query_vtk(
            query=args.query,
            visual_test=args.visual_test,
            enrich=args.enrich,
            output_file=args.output,
            verbose=not args.quiet
        )
        
        # Display result
        print(format_response(response))
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
