#!/usr/bin/env python3
"""
Refinement Session - Undo/Rollback Support for Code Refinement

Provides session-based tracking of code refinements with undo/rollback capabilities.
Designed with GUI integration in mind - simple, clear API.

Usage:
    # Create session
    session = RefinementSession(initial_code)
    
    # Track refinements
    session.add_refinement(query, result)
    
    # Undo
    previous_code = session.undo()
    
    # List versions
    versions = session.get_version_list()
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


@dataclass
class RefinementVersion:
    """A single version in refinement history"""
    version: int
    timestamp: datetime
    code: str
    query: Optional[str]
    modifications: List[Dict]
    explanation: str
    diff: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'code': self.code,
            'query': self.query,
            'modifications': self.modifications,
            'explanation': self.explanation,
            'diff': self.diff,
            'code_preview': self.code[:100] + '...' if len(self.code) > 100 else self.code
        }


class RefinementSession:
    """
    Manages refinement history with undo/rollback support.
    
    GUI-friendly design:
    - Clear method names
    - Simple return values
    - No complex state management
    - Easy to serialize
    
    Example:
        session = RefinementSession(initial_code)
        session.add_refinement("Make it blue", result)
        session.add_refinement("Increase resolution", result)
        
        # Undo last change
        previous = session.undo()
        
        # Or jump to specific version
        original = session.go_to_version(0)
    """
    
    def __init__(self, initial_code: str, session_id: Optional[str] = None):
        """
        Create a new refinement session.
        
        Args:
            initial_code: Starting code before any refinements
            session_id: Optional session identifier (auto-generated if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.created_at = datetime.now()
        
        # History stores all versions
        self._history: List[RefinementVersion] = [
            RefinementVersion(
                version=0,
                timestamp=self.created_at,
                code=initial_code,
                query=None,
                modifications=[],
                explanation='Initial version',
                diff=None
            )
        ]
        
        # Current position in history (for undo/redo)
        self._current_index = 0
    
    # ========== CORE OPERATIONS ==========
    
    def add_refinement(self, query: str, result: Dict) -> int:
        """
        Add a refinement to the session.
        
        GUI Usage:
            After getting refinement result from pipeline:
            version = session.add_refinement(user_query, pipeline_result)
            
        Args:
            query: User's modification request
            result: Complete result from pipeline.process_query()
        
        Returns:
            int: New version number
        """
        # If we're not at the latest version, truncate future history
        if self._current_index < len(self._history) - 1:
            self._history = self._history[:self._current_index + 1]
        
        # Create new version
        new_version = len(self._history)
        version = RefinementVersion(
            version=new_version,
            timestamp=datetime.now(),
            code=result['code'],
            query=query,
            modifications=result.get('modifications', []),
            explanation=result.get('explanation', ''),
            diff=result.get('diff', None)
        )
        
        self._history.append(version)
        self._current_index = new_version
        
        return new_version
    
    def undo(self, steps: int = 1) -> str:
        """
        Undo N refinement steps.
        
        GUI Usage:
            code = session.undo()  # Go back 1 step
            code = session.undo(3)  # Go back 3 steps
        
        Args:
            steps: Number of steps to go back (default: 1)
        
        Returns:
            str: Code at previous version
        
        Raises:
            ValueError: If already at initial version
        """
        if self._current_index == 0:
            raise ValueError("Already at initial version, cannot undo")
        
        # Calculate target index
        target_index = max(0, self._current_index - steps)
        self._current_index = target_index
        
        return self._history[target_index].code
    
    def redo(self, steps: int = 1) -> str:
        """
        Redo N refinement steps (after undo).
        
        GUI Usage:
            code = session.redo()  # Go forward 1 step
        
        Args:
            steps: Number of steps to go forward (default: 1)
        
        Returns:
            str: Code at next version
        
        Raises:
            ValueError: If already at latest version
        """
        max_index = len(self._history) - 1
        
        if self._current_index >= max_index:
            raise ValueError("Already at latest version, cannot redo")
        
        # Calculate target index
        target_index = min(max_index, self._current_index + steps)
        self._current_index = target_index
        
        return self._history[target_index].code
    
    def go_to_version(self, version: int) -> str:
        """
        Jump to a specific version.
        
        GUI Usage:
            code = session.go_to_version(2)  # Jump to version 2
        
        Args:
            version: Version number to jump to
        
        Returns:
            str: Code at specified version
        
        Raises:
            ValueError: If version doesn't exist
        """
        if version < 0 or version >= len(self._history):
            raise ValueError(
                f"Version {version} not found. "
                f"Valid range: 0 to {len(self._history) - 1}"
            )
        
        self._current_index = version
        return self._history[version].code
    
    # ========== QUERY OPERATIONS ==========
    
    def get_current_code(self) -> str:
        """
        Get code at current version.
        
        GUI Usage:
            current_code = session.get_current_code()
        
        Returns:
            str: Current code
        """
        return self._history[self._current_index].code
    
    def get_current_version(self) -> int:
        """
        Get current version number.
        
        GUI Usage:
            version = session.get_current_version()
            status_label.setText(f"Version {version}")
        
        Returns:
            int: Current version number
        """
        return self._current_index
    
    def get_version_count(self) -> int:
        """
        Get total number of versions.
        
        GUI Usage:
            total = session.get_version_count()
            progress_bar.setMaximum(total - 1)
        
        Returns:
            int: Total versions in history
        """
        return len(self._history)
    
    def can_undo(self) -> bool:
        """
        Check if undo is possible.
        
        GUI Usage:
            undo_button.setEnabled(session.can_undo())
        
        Returns:
            bool: True if can undo
        """
        return self._current_index > 0
    
    def can_redo(self) -> bool:
        """
        Check if redo is possible.
        
        GUI Usage:
            redo_button.setEnabled(session.can_redo())
        
        Returns:
            bool: True if can redo
        """
        return self._current_index < len(self._history) - 1
    
    def get_version_info(self, version: Optional[int] = None) -> Dict:
        """
        Get detailed info about a version.
        
        GUI Usage:
            # Get current version info
            info = session.get_version_info()
            
            # Get specific version info
            info = session.get_version_info(2)
        
        Args:
            version: Version number (None = current)
        
        Returns:
            dict: Version information
        """
        idx = version if version is not None else self._current_index
        
        if idx < 0 or idx >= len(self._history):
            raise ValueError(f"Version {idx} not found")
        
        return self._history[idx].to_dict()
    
    def get_version_list(self) -> List[Dict]:
        """
        Get list of all versions for display.
        
        GUI Usage:
            versions = session.get_version_list()
            for v in versions:
                list_widget.addItem(f"v{v['version']}: {v['query']}")
        
        Returns:
            List[dict]: List of version summaries
        """
        return [
            {
                'version': v.version,
                'timestamp': v.timestamp.isoformat(),
                'query': v.query or 'Initial version',
                'num_modifications': len(v.modifications),
                'is_current': (v.version == self._current_index)
            }
            for v in self._history
        ]
    
    def get_diff_between_versions(self, from_version: int, to_version: int) -> str:
        """
        Get diff between two versions.
        
        GUI Usage:
            diff = session.get_diff_between_versions(0, 2)
            diff_viewer.setText(diff)
        
        Args:
            from_version: Start version
            to_version: End version
        
        Returns:
            str: Unified diff
        """
        import difflib
        
        if from_version < 0 or from_version >= len(self._history):
            raise ValueError(f"Version {from_version} not found")
        if to_version < 0 or to_version >= len(self._history):
            raise ValueError(f"Version {to_version} not found")
        
        from_code = self._history[from_version].code
        to_code = self._history[to_version].code
        
        diff = difflib.unified_diff(
            from_code.splitlines(keepends=True),
            to_code.splitlines(keepends=True),
            fromfile=f'version_{from_version}',
            tofile=f'version_{to_version}',
            lineterm=''
        )
        
        return ''.join(diff)
    
    # ========== UTILITY OPERATIONS ==========
    
    def get_session_info(self) -> Dict:
        """
        Get session metadata.
        
        GUI Usage:
            info = session.get_session_info()
            status_bar.showMessage(f"Session: {info['session_id']}")
        
        Returns:
            dict: Session information
        """
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'total_versions': len(self._history),
            'current_version': self._current_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }
    
    def get_modification_summary(self) -> Dict:
        """
        Get summary of all modifications made.
        
        GUI Usage:
            summary = session.get_modification_summary()
            label.setText(f"{summary['total_refinements']} refinements")
        
        Returns:
            dict: Modification summary
        """
        total_mods = sum(
            len(v.modifications) 
            for v in self._history[1:]  # Skip initial version
        )
        
        return {
            'total_refinements': len(self._history) - 1,
            'total_modifications': total_mods,
            'queries': [v.query for v in self._history[1:]]
        }
    
    def clear_future_history(self):
        """
        Clear all versions after current (called when adding new refinement).
        
        This happens automatically in add_refinement(), but exposed for manual use.
        """
        if self._current_index < len(self._history) - 1:
            self._history = self._history[:self._current_index + 1]
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"RefinementSession(id={self.session_id}, "
            f"versions={len(self._history)}, "
            f"current={self._current_index})"
        )
