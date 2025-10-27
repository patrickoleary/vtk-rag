#!/usr/bin/env python3
"""
JSON Logger for Pipeline Debugging

Logs all JSON exchanges between application and LLM for debugging and analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class PipelineLogger:
    """Logger for JSON-based pipeline exchanges"""
    
    def __init__(self, output_dir: Path = Path("evaluation/logs")):
        """
        Initialize logger
        
        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_log = {
            "timestamp": datetime.now().isoformat(),
            "query": None,
            "decomposition": None,
            "steps": [],
            "validation": None,
            "final_result": None
        }
    
    def log_query(self, query: str):
        """Log the original query"""
        self.current_log["query"] = query
        self.current_log["timestamp"] = datetime.now().isoformat()
    
    def log_decomposition(self, input_data: Dict, output_data: Dict):
        """
        Log decomposition input and output
        
        Args:
            input_data: JSON input sent to LLM
            output_data: JSON output received from LLM
        """
        self.current_log["decomposition"] = {
            "input": input_data,
            "output": output_data
        }
    
    def log_step(self, step_number: int, input_data: Dict, output_data: Dict):
        """
        Log a generation step
        
        Args:
            step_number: Current step number
            input_data: JSON input sent to LLM
            output_data: JSON output received from LLM
        """
        self.current_log["steps"].append({
            "step_number": step_number,
            "input": input_data,
            "output": output_data
        })
    
    def log_validation(self, input_data: Dict, output_data: Optional[Dict] = None):
        """
        Log validation attempt
        
        Args:
            input_data: JSON input sent to LLM for validation
            output_data: JSON output received from LLM (if validation was needed)
        """
        if self.current_log["validation"] is None:
            self.current_log["validation"] = []
        
        self.current_log["validation"].append({
            "input": input_data,
            "output": output_data
        })
    
    def log_final_result(self, result: Dict):
        """
        Log final assembled result
        
        Args:
            result: Final result dictionary
        """
        self.current_log["final_result"] = result
    
    def save(self, filename: Optional[str] = None):
        """
        Save log to file
        
        Args:
            filename: Optional filename. If None, generates timestamp-based name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = self._slugify(self.current_log.get("query", "unknown")[:30])
            filename = f"{timestamp}_{query_slug}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(self.current_log, f, indent=2)
        
        print(f"✓ Saved pipeline log to {output_file}")
        return output_file
    
    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug"""
        import re
        # Remove non-alphanumeric characters
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        # Replace spaces with underscores
        slug = re.sub(r'\s+', '_', slug)
        # Convert to lowercase
        return slug.lower()
    
    def clear(self):
        """Clear current log (start fresh)"""
        self.current_log = {
            "timestamp": datetime.now().isoformat(),
            "query": None,
            "decomposition": None,
            "steps": [],
            "validation": None,
            "final_result": None
        }
    
    def get_log(self) -> Dict:
        """Get current log as dictionary"""
        return self.current_log.copy()
    
    @staticmethod
    def load_log(filepath: Path) -> Dict:
        """
        Load a log file
        
        Args:
            filepath: Path to log file
            
        Returns:
            Log data as dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def format_log_summary(log: Dict) -> str:
        """
        Format log as human-readable summary
        
        Args:
            log: Log dictionary
            
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PIPELINE LOG SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {log.get('timestamp', 'N/A')}")
        lines.append(f"Query: {log.get('query', 'N/A')}")
        lines.append("")
        
        # Decomposition
        decomp = log.get('decomposition')
        if decomp:
            output = decomp.get('output', {})
            lines.append(f"Decomposition: {len(output.get('steps', []))} steps")
            lines.append(f"  Understanding: {output.get('understanding', 'N/A')[:80]}...")
        
        # Steps
        steps = log.get('steps', [])
        lines.append(f"\nGeneration Steps: {len(steps)}")
        for step in steps:
            output = step.get('output', {})
            lines.append(f"  Step {step['step_number']}: {len(output.get('code', ''))} chars of code")
        
        # Validation
        validation = log.get('validation')
        if validation:
            lines.append(f"\nValidation: {len(validation)} attempt(s)")
        
        # Final result
        final = log.get('final_result')
        if final:
            code = final.get('final_code', {})
            lines.append(f"\nFinal Code: {len(code.get('complete', ''))} chars")
        
        lines.append("=" * 80)
        return "\n".join(lines)
