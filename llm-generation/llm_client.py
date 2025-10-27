#!/usr/bin/env python3
"""
Multi-Provider LLM Client

Unified interface for multiple LLM providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Local models (OpenAI-compatible APIs)

Configured via .env file.
"""

import os
import logging
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM generation"""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 1.0
    api_base: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables"""
        provider_str = os.getenv('LLM_PROVIDER', 'openai').lower()
        provider = LLMProvider(provider_str)
        
        if provider == LLMProvider.OPENAI:
            return cls(
                provider=provider,
                model=os.getenv('OPENAI_MODEL', 'gpt-4'),
                api_key=os.getenv('OPENAI_API_KEY', ''),
                temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
                top_p=float(os.getenv('OPENAI_TOP_P', '1.0'))
            )
        elif provider == LLMProvider.ANTHROPIC:
            return cls(
                provider=provider,
                model=os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                api_key=os.getenv('ANTHROPIC_API_KEY', ''),
                temperature=float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '2000')),
                top_p=float(os.getenv('ANTHROPIC_TOP_P', '1.0'))
            )
        elif provider == LLMProvider.GOOGLE:
            return cls(
                provider=provider,
                model=os.getenv('GOOGLE_MODEL', 'gemini-pro'),
                api_key=os.getenv('GOOGLE_API_KEY', ''),
                temperature=float(os.getenv('GOOGLE_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('GOOGLE_MAX_TOKENS', '2000')),
                top_p=float(os.getenv('GOOGLE_TOP_P', '1.0'))
            )
        elif provider == LLMProvider.LOCAL:
            return cls(
                provider=provider,
                model=os.getenv('LOCAL_MODEL', 'local-model'),
                api_key=os.getenv('LOCAL_API_KEY', 'not-needed'),
                api_base=os.getenv('LOCAL_API_BASE', 'http://localhost:8000/v1'),
                temperature=float(os.getenv('LOCAL_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('LOCAL_MAX_TOKENS', '2000'))
            )
        else:
            raise ValueError(f"Unknown provider: {provider_str}")


class LLMClient:
    """
    Unified client for multiple LLM providers
    
    Provides consistent interface across OpenAI, Anthropic, and local models.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client
        
        Args:
            config: LLM configuration. If None, loads from environment.
        """
        self.config = config or LLMConfig.from_env()
        self.client = self._init_client()
        
        # LLM client initialized
    
    def _init_client(self):
        """Initialize provider-specific client"""
        if self.config.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=self.config.api_key)
        
        elif self.config.provider == LLMProvider.ANTHROPIC:
            from anthropic import Anthropic
            return Anthropic(api_key=self.config.api_key)
        
        elif self.config.provider == LLMProvider.GOOGLE:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            return genai
        
        elif self.config.provider == LLMProvider.LOCAL:
            from openai import OpenAI
            return OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_message: Optional system message (extracted from prompt if not provided)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        # Parse system message if not provided
        if system_message is None:
            system_message, prompt = self._split_system_and_user(prompt)
        
        # Log request if enabled
        if os.getenv('LOG_REQUESTS', 'false').lower() == 'true':
            logger.debug(f"System: {system_message[:200]}...")
            logger.debug(f"Prompt: {prompt[:200]}...")
        
        # Generate based on provider
        if self.config.provider == LLMProvider.OPENAI or self.config.provider == LLMProvider.LOCAL:
            response = self._generate_openai(system_message, prompt, **kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            response = self._generate_anthropic(system_message, prompt, **kwargs)
        elif self.config.provider == LLMProvider.GOOGLE:
            response = self._generate_google(system_message, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        
        # Log response if enabled
        if os.getenv('LOG_RESPONSES', 'false').lower() == 'true':
            logger.debug(f"Response: {response[:200]}...")
        
        return response
    
    def generate_json(
        self,
        prompt_data: Dict,
        schema_name: str,
        temperature: float = 0.1,
        max_retries: int = 2
    ) -> Dict:
        """
        Generate structured JSON response from LLM
        
        Args:
            prompt_data: Structured input data as dictionary
            schema_name: Expected output schema name (for validation)
            temperature: Generation temperature
            max_retries: Maximum retry attempts for invalid JSON
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            ValueError: If LLM fails to return valid JSON after retries
        """
        # Convert input data to JSON string
        json_prompt = json.dumps(prompt_data, indent=2)
        
        # Create system message requesting JSON output with explicit schema
        schema_details = self._get_schema_details(schema_name)
        system_message = (
            f"You are a helpful assistant that responds ONLY with valid JSON.\n\n"
            f"REQUIRED OUTPUT FORMAT ({schema_name}):\n"
            f"{schema_details}\n\n"
            f"CRITICAL RULES:\n"
            f"- Return ONLY a JSON object matching this EXACT structure\n"
            f"- Use the EXACT field names shown above (case-sensitive)\n"
            f"- Do not add extra fields\n"
            f"- Do not include any text before or after the JSON\n"
            f"- Do not wrap the JSON in markdown code blocks"
        )
        
        # Try to get valid JSON response with retries
        for attempt in range(max_retries + 1):
            try:
                # Generate response
                response = self.generate(
                    prompt=json_prompt,
                    system_message=system_message,
                    temperature=temperature
                )
                
                # Extract and parse JSON
                parsed_json = self._extract_json_from_response(response)
                
                # Validate schema if validation function exists
                if not self._validate_json_schema(parsed_json, schema_name):
                    raise ValueError(f"JSON does not match {schema_name} schema")
                
                pass  # JSON generated successfully
                return parsed_json
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries:
                    logger.error(f"Failed to generate valid JSON after {max_retries + 1} attempts")
                    logger.error(f"Last response: {response[:500]}")
                    raise ValueError(f"LLM failed to return valid JSON for {schema_name}: {e}")
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                # Add error feedback to next attempt
                system_message += f"\n\nPrevious attempt failed with error: {e}. Please return valid JSON only."
        
        raise ValueError(f"Unexpected error in generate_json for {schema_name}")
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """
        Extract JSON from LLM response, handling markdown code blocks
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try to parse response directly first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # No valid JSON found
        raise json.JSONDecodeError(
            "No valid JSON found in response",
            response,
            0
        )
    
    def _get_schema_details(self, schema_name: str) -> str:
        """
        Get explicit schema structure details for system message
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            String describing the required JSON structure
        """
        if schema_name == "DecompositionOutput":
            return """
{
  "understanding": "string - Brief summary of what user wants",
  "requires_visualization": boolean,
  "libraries_needed": ["string array - e.g. vtk, pandas, numpy"],
  "data_files": ["string array - e.g. file.csv"],
  "steps": [
    {
      "step_number": integer (1, 2, 3...),
      "description": "string - Detailed step description",
      "search_query": "string - Search query for documentation",
      "focus": "string - One of: data_io, data_processing, geometry, filtering, transformation, visualization, rendering, utility"
    }
  ]
}"""
        elif schema_name == "GenerationOutput":
            return """
{
  "step_number": integer,
  "understanding": "string - What this step does",
  "imports": ["string array - Python import statements"],
  "code": "string - Python code for this step",
  "citations": [integer array - Documentation chunk indices used]
}"""
        elif schema_name == "ValidationOutput":
            return """
{
  "fixed_code": "string - Complete corrected code",
  "changes_made": [
    {
      "error_type": "string - Type of error fixed",
      "fix": "string - Description of fix",
      "line": integer
    }
  ]
}"""
        elif schema_name == "APILookupOutput":
            return """
{
  "response_type": "answer",
  "content_type": "api",
  "explanation": "string - Detailed explanation of the API/method",
  "usage_example": "string - Code example showing usage",
  "parameters": [
    {
      "name": "string",
      "type": "string",
      "description": "string"
    }
  ],
  "return_value": "string - Description of return value",
  "related_methods": ["string array - Related methods/classes"],
  "citations": [{"number": integer, "reason": "string"}],
  "confidence": "high|medium|low"
}"""
        elif schema_name == "ExplanationOutput":
            return """
{
  "response_type": "answer",
  "content_type": "explanation",
  "explanation": "string - Detailed concept explanation",
  "key_concepts": [
    {
      "concept": "string",
      "description": "string"
    }
  ],
  "examples": ["string array - Example descriptions"],
  "related_concepts": ["string array"],
  "citations": [{"number": integer, "reason": "string"}],
  "confidence": "high|medium|low"
}"""
        elif schema_name == "DataToCodeOutput":
            return """
{
  "response_type": "answer",
  "content_type": "code",
  "data_analysis": "string - Description of data type and suitability",
  "suggested_techniques": ["string array - List of technique names"],
  "code": "string - Working code for most common technique",
  "explanation": "string - How the code works",
  "alternative_approaches": [
    {
      "technique": "string",
      "description": "string",
      "vtk_classes": ["string array"],
      "complexity": "simple|moderate|advanced"
    }
  ],
  "vtk_classes_used": ["string array"],
  "data_files_used": ["string array"],
  "citations": [{"number": integer, "reason": "string"}],
  "confidence": "high|medium|low"
}"""
        elif schema_name == "CodeToDataOutput":
            return """
{
  "response_type": "answer",
  "content_type": "data",
  "explanation": "string - What data the code needs",
  "code_requirements": "string - Description of data format expected",
  "data_files": [
    {
      "filename": "string",
      "description": "string",
      "source_example": "string",
      "download_url": "string",
      "file_type": "string",
      "file_size": "string"
    }
  ],
  "vtk_classes_used": ["string array"],
  "citations": [{"number": integer, "reason": "string"}],
  "confidence": "high|medium|low"
}"""
        elif schema_name == "ExplanationEnrichmentOutput":
            return """
{
  "improved_explanation": "string - Enhanced or generated explanation",
  "key_points": ["string array - Main takeaways"],
  "vtk_classes_explained": [
    {
      "name": "string - VTK class name",
      "purpose": "string - What this class does in the code"
    }
  ],
  "citations": [{"number": integer, "reason": "string"}],
  "confidence": "high|medium|low"
}"""
        elif schema_name == "ModificationDecompositionOutput":
            return """
{
  "understanding": "string - What the user wants to change",
  "modification_steps": [
    {
      "step_number": integer (1, 2, 3...),
      "description": "string - What to modify",
      "requires_retrieval": boolean
    }
  ],
  "preserved_elements": ["string array - What should NOT be changed"]
}"""
        elif schema_name == "CodeModificationOutput":
            return """
{
  "modifications": [
    {
      "step_number": integer,
      "modification": "string - What was modified (short)",
      "explanation": "string - Why this change was made (detailed)",
      "code_changed": "string - Line that was changed (or empty)",
      "code_added": "string - Code that was added (or empty)",
      "variable_affected": "string - Variable name modified"
    }
  ],
  "updated_code": "string - Complete updated code",
  "new_imports": ["string array - New import statements added"],
  "preserved_structure": boolean,
  "diff_summary": "string - Human-readable summary of changes"
}"""
        else:
            return f"{{JSON object matching {schema_name} schema}}"
    
    def _validate_json_schema(self, data: Dict, schema_name: str) -> bool:
        """
        Validate JSON data against expected schema
        
        Args:
            data: Parsed JSON data
            schema_name: Schema name to validate against
            
        Returns:
            True if valid, False otherwise
        """
        # Import validation functions from schemas module
        try:
            from schemas import (
                validate_decomposition_output,
                validate_generation_output,
                validate_validation_output,
                validate_api_lookup_output,
                validate_explanation_output,
                validate_data_to_code_output,
                validate_code_to_data_output,
                validate_explanation_enrichment_output,
                validate_modification_decomposition_output,
                validate_code_modification_output
            )
            
            validators = {
                "DecompositionOutput": validate_decomposition_output,
                "GenerationOutput": validate_generation_output,
                "ValidationOutput": validate_validation_output,
                "APILookupOutput": validate_api_lookup_output,
                "ExplanationOutput": validate_explanation_output,
                "DataToCodeOutput": validate_data_to_code_output,
                "CodeToDataOutput": validate_code_to_data_output,
                "ExplanationEnrichmentOutput": validate_explanation_enrichment_output,
                "ModificationDecompositionOutput": validate_modification_decomposition_output,
                "CodeModificationOutput": validate_code_modification_output,
            }
            
            validator = validators.get(schema_name)
            if validator:
                return validator(data)
            else:
                logger.warning(f"No validator found for schema: {schema_name}")
                return True  # Assume valid if no validator
                
        except ImportError:
            logger.warning("schemas module not found, skipping validation")
            return True
    
    def _generate_openai(
        self,
        system_message: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate using OpenAI API"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            top_p=kwargs.get('top_p', self.config.top_p)
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        system_message: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate using Anthropic API"""
        # Claude Sonnet 4.5+ doesn't allow both temperature and top_p
        # Only include top_p if explicitly provided and different from default
        api_kwargs = {
            'model': self.config.model,
            'system': system_message,
            'messages': [
                {"role": "user", "content": user_prompt}
            ],
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens)
        }
        
        # Only add top_p if explicitly requested and not default
        top_p_value = kwargs.get('top_p', self.config.top_p)
        if 'top_p' in kwargs and top_p_value != 1.0:
            api_kwargs['top_p'] = top_p_value
        
        response = self.client.messages.create(**api_kwargs)
        
        return response.content[0].text
    
    def _generate_google(
        self,
        system_message: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate using Google Gemini API"""
        # Combine system message and user prompt for Gemini
        # Gemini doesn't have separate system/user roles like others
        full_prompt = f"{system_message}\n\n{user_prompt}"
        
        model = self.client.GenerativeModel(
            model_name=self.config.model,
            generation_config={
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_output_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
            }
        )
        
        response = model.generate_content(full_prompt)
        
        return response.text
    
    def _split_system_and_user(self, prompt: str) -> tuple[str, str]:
        """
        Split combined prompt into system and user parts
        
        Assumes format from prompt_templates.py
        """
        if "USER QUESTION:" in prompt:
            parts = prompt.split("USER QUESTION:")
            system = parts[0].strip()
            user = "USER QUESTION:" + parts[1] if len(parts) > 1 else ""
            return system, user.strip()
        else:
            # No split found - treat entire prompt as user message
            return "", prompt
