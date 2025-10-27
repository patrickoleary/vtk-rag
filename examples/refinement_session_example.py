#!/usr/bin/env python3
"""
Refinement Session Example - Undo/Rollback Support

Demonstrates session-based code refinement with undo/redo capabilities.
Shows GUI-friendly API usage patterns.

This example shows how a GUI application would integrate the refinement session.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm-generation'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'retrieval-pipeline'))

from sequential_pipeline import SequentialPipeline
from refinement_session import RefinementSession


def example_basic_session():
    """Example 1: Basic session usage"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Session Usage")
    print("=" * 80)
    
    # Original code
    original_code = """from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

cylinder = vtkCylinderSource()
cylinder.SetResolution(8)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)
"""
    
    print("\nOriginal Code:")
    print("-" * 80)
    print(original_code)
    
    # Create session
    session = RefinementSession(original_code)
    print(f"\n✅ Session created: {session.session_id}")
    print(f"   Current version: {session.get_current_version()}")
    print(f"   Can undo: {session.can_undo()}")
    print(f"   Can redo: {session.can_redo()}")
    
    print("\n" + "=" * 80 + "\n")


def example_session_with_pipeline():
    """Example 2: Session with pipeline integration"""
    print("=" * 80)
    print("EXAMPLE 2: Session with Pipeline")
    print("=" * 80)
    
    # Initialize pipeline (without LLM for demo)
    pipeline = SequentialPipeline(use_llm_decomposition=False)
    
    # Original code
    original_code = """from vtkmodules.vtkFiltersSources import vtkCylinderSource
cylinder = vtkCylinderSource()
cylinder.SetResolution(8)
"""
    
    # Create session through pipeline (GUI would do this)
    session = pipeline.create_refinement_session(original_code)
    
    print(f"Session created: {session.session_id}")
    print(f"Initial code length: {len(original_code)} characters")
    
    # Simulate refinements (GUI would call these)
    print("\n📝 Refinement 1: 'Increase resolution to 50'")
    # In real usage with LLM:
    # result = pipeline.refine_in_session(session, "Increase resolution to 50")
    
    # For demo, manually add a refinement
    mock_result = {
        'code': original_code.replace('SetResolution(8)', 'SetResolution(50)'),
        'modifications': [{'step_number': 1, 'modification': 'Changed resolution'}],
        'explanation': 'Increased resolution from 8 to 50',
        'diff': '- cylinder.SetResolution(8)\n+ cylinder.SetResolution(50)'
    }
    version = session.add_refinement("Increase resolution to 50", mock_result)
    print(f"   ✅ Version {version} created")
    
    # Check state
    print(f"\n   Session state:")
    print(f"   - Current version: {session.get_current_version()}")
    print(f"   - Total versions: {session.get_version_count()}")
    print(f"   - Can undo: {session.can_undo()}")
    
    print("\n" + "=" * 80 + "\n")


def example_undo_redo():
    """Example 3: Undo/Redo operations"""
    print("=" * 80)
    print("EXAMPLE 3: Undo/Redo Operations")
    print("=" * 80)
    
    # Setup
    original_code = "cylinder.SetResolution(8)"
    session = RefinementSession(original_code)
    
    # Add some refinements
    refinements = [
        ("Increase to 50", {'code': "cylinder.SetResolution(50)", 'modifications': [], 'explanation': "Changed to 50"}),
        ("Make it 100", {'code': "cylinder.SetResolution(100)", 'modifications': [], 'explanation': "Changed to 100"}),
        ("Set to 75", {'code': "cylinder.SetResolution(75)", 'modifications': [], 'explanation': "Changed to 75"}),
    ]
    
    for query, result in refinements:
        version = session.add_refinement(query, result)
        print(f"✅ Version {version}: {query}")
    
    print(f"\nCurrent version: {session.get_current_version()} (resolution=75)")
    print(f"Current code: {session.get_current_code()}")
    
    # Undo operations
    print("\n" + "-" * 80)
    print("UNDO OPERATIONS:")
    print("-" * 80)
    
    print("\n⏪ Undo 1 step")
    code = session.undo()
    print(f"   Version: {session.get_current_version()} (resolution=100)")
    print(f"   Code: {code}")
    print(f"   Can undo: {session.can_undo()}")
    print(f"   Can redo: {session.can_redo()}")
    
    print("\n⏪ Undo 2 more steps")
    code = session.undo(2)
    print(f"   Version: {session.get_current_version()} (back to original)")
    print(f"   Code: {code}")
    print(f"   Can undo: {session.can_undo()}")
    
    # Try to undo too far
    print("\n⏪ Try to undo again (should fail)")
    try:
        session.undo()
    except ValueError as e:
        print(f"   ❌ Error (expected): {e}")
    
    # Redo operations
    print("\n" + "-" * 80)
    print("REDO OPERATIONS:")
    print("-" * 80)
    
    print("\n⏩ Redo 1 step")
    code = session.redo()
    print(f"   Version: {session.get_current_version()} (resolution=50)")
    print(f"   Code: {code}")
    
    print("\n⏩ Redo 2 more steps")
    code = session.redo(2)
    print(f"   Version: {session.get_current_version()} (resolution=75)")
    print(f"   Code: {code}")
    print(f"   Can redo: {session.can_redo()}")
    
    print("\n" + "=" * 80 + "\n")


def example_version_history():
    """Example 4: Version history display"""
    print("=" * 80)
    print("EXAMPLE 4: Version History (GUI Display)")
    print("=" * 80)
    
    # Setup
    original_code = "# Original code"
    session = RefinementSession(original_code, session_id="demo-123")
    
    # Add refinements
    changes = [
        "Add color property",
        "Increase resolution",
        "Add rotation",
        "Change opacity"
    ]
    
    for i, change in enumerate(changes, 1):
        result = {
            'code': f"# Code after: {change}",
            'modifications': [{'step_number': 1}],
            'explanation': f"Applied {change}",
            'diff': f"Applied {change}"
        }
        session.add_refinement(change, result)
    
    # Display version list (as GUI would)
    print("\n📋 VERSION HISTORY:")
    print("-" * 80)
    
    versions = session.get_version_list()
    for v in versions:
        current_marker = " ◀ CURRENT" if v['is_current'] else ""
        print(f"v{v['version']}: {v['query']}{current_marker}")
        print(f"   Time: {v['timestamp']}")
        print(f"   Modifications: {v['num_modifications']}")
        print()
    
    # Show session info
    print("-" * 80)
    print("SESSION INFO:")
    print("-" * 80)
    info = session.get_session_info()
    print(f"Session ID: {info['session_id']}")
    print(f"Created: {info['created_at']}")
    print(f"Total Versions: {info['total_versions']}")
    print(f"Current Version: {info['current_version']}")
    print(f"Can Undo: {info['can_undo']}")
    print(f"Can Redo: {info['can_redo']}")
    
    # Show modification summary
    print("\n" + "-" * 80)
    print("MODIFICATION SUMMARY:")
    print("-" * 80)
    summary = session.get_modification_summary()
    print(f"Total Refinements: {summary['total_refinements']}")
    print(f"Total Modifications: {summary['total_modifications']}")
    print(f"Queries: {', '.join(summary['queries'])}")
    
    print("\n" + "=" * 80 + "\n")


def example_gui_integration_pattern():
    """Example 5: Pattern for GUI integration"""
    print("=" * 80)
    print("EXAMPLE 5: GUI Integration Pattern")
    print("=" * 80)
    
    print("\nPseudo-code for GUI application:\n")
    
    gui_code = '''
class CodeRefinementGUI:
    """Example GUI application with undo/redo support"""
    
    def __init__(self):
        # Initialize pipeline
        self.pipeline = SequentialPipeline()
        
        # Session (None until user starts refinement)
        self.session = None
        
        # UI components
        self.code_editor = CodeEditor()
        self.undo_button = Button("Undo", callback=self.on_undo)
        self.redo_button = Button("Redo", callback=self.on_redo)
        self.version_list = ListWidget()
        
        # Initially disabled
        self.update_undo_redo_buttons()
    
    def start_refinement_session(self, initial_code):
        """User clicks 'Start Refinement' button"""
        # Create session
        self.session = self.pipeline.create_refinement_session(initial_code)
        
        # Update UI
        self.code_editor.setText(initial_code)
        self.update_version_list()
        self.update_undo_redo_buttons()
        
        print(f"Session started: {self.session.session_id}")
    
    def on_refine_clicked(self, user_query):
        """User enters refinement query and clicks 'Refine'"""
        if not self.session:
            show_error("No active session")
            return
        
        try:
            # Perform refinement
            result = self.pipeline.refine_in_session(
                self.session,
                user_query
            )
            
            # Update code editor
            self.code_editor.setText(result['code'])
            
            # Update UI state
            self.update_version_list()
            self.update_undo_redo_buttons()
            
            # Show status
            version = result['session_version']
            show_status(f"Refinement applied (v{version})")
            
        except Exception as e:
            show_error(f"Refinement failed: {e}")
    
    def on_undo(self):
        """User clicks 'Undo' button"""
        if not self.session or not self.session.can_undo():
            return
        
        try:
            # Undo
            previous_code = self.pipeline.undo_refinement(self.session)
            
            # Update UI
            self.code_editor.setText(previous_code)
            self.update_version_list()
            self.update_undo_redo_buttons()
            
            version = self.session.get_current_version()
            show_status(f"Undone to v{version}")
            
        except ValueError as e:
            show_error(str(e))
    
    def on_redo(self):
        """User clicks 'Redo' button"""
        if not self.session or not self.session.can_redo():
            return
        
        try:
            # Redo
            next_code = self.pipeline.redo_refinement(self.session)
            
            # Update UI
            self.code_editor.setText(next_code)
            self.update_version_list()
            self.update_undo_redo_buttons()
            
            version = self.session.get_current_version()
            show_status(f"Redone to v{version}")
            
        except ValueError as e:
            show_error(str(e))
    
    def update_undo_redo_buttons(self):
        """Enable/disable undo/redo based on session state"""
        if self.session:
            self.undo_button.setEnabled(self.session.can_undo())
            self.redo_button.setEnabled(self.session.can_redo())
        else:
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
    
    def update_version_list(self):
        """Refresh version history display"""
        if not self.session:
            self.version_list.clear()
            return
        
        # Get version list
        versions = self.pipeline.get_session_versions(self.session)
        
        # Populate list
        self.version_list.clear()
        for v in versions:
            is_current = " (current)" if v['is_current'] else ""
            item_text = f"v{v['version']}: {v['query']}{is_current}"
            self.version_list.addItem(item_text)
    
    def on_version_clicked(self, version_index):
        """User clicks on a version in the history list"""
        try:
            # Jump to that version
            code = self.session.go_to_version(version_index)
            
            # Update UI
            self.code_editor.setText(code)
            self.update_version_list()
            self.update_undo_redo_buttons()
            
            show_status(f"Jumped to v{version_index}")
            
        except ValueError as e:
            show_error(str(e))
'''
    
    print(gui_code)
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "Refinement Session - Undo/Redo Example" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run examples
    example_basic_session()
    example_session_with_pipeline()
    example_undo_redo()
    example_version_history()
    example_gui_integration_pattern()
    
    print("\n✅ All examples completed!")
    print("\nKey takeaways for GUI integration:")
    print("  1. Create session with: pipeline.create_refinement_session(code)")
    print("  2. Refine with: pipeline.refine_in_session(session, query)")
    print("  3. Undo with: pipeline.undo_refinement(session)")
    print("  4. Redo with: pipeline.redo_refinement(session)")
    print("  5. Check state with: session.can_undo() / session.can_redo()")
    print("  6. Display versions with: pipeline.get_session_versions(session)")
    print()
