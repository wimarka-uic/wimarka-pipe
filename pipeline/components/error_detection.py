"""
Error Detection Component using Gemma 3
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config import ModelConfig


class ErrorDetector:
    """Error detection component using Gemma 3"""
    
    def __init__(self, config: ModelConfig):
        """Initialize error detector with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                cache_dir="./cache",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
            
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            self.logger.info(f"Error detection model loaded: {config.model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load error detection model: {str(e)}")
            raise
    
    def detect_errors(self, text: str, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Detect errors in the input text using Gemma 3
        
        Args:
            text: Input text to analyze
            tokens: Tokenized text
            
        Returns:
            List of detected errors with details
        """
        try:
            # Create prompt for error detection
            prompt = self._create_error_detection_prompt(text)
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Parse errors from response
            errors = self._parse_errors(response, text, tokens)
            
            self.logger.debug(f"Detected {len(errors)} errors in text")
            return errors
            
        except Exception as e:
            self.logger.error(f"Error detection failed: {str(e)}")
            raise
    
    def _create_error_detection_prompt(self, text: str) -> str:
        """Create prompt for error detection"""
        return f"""Analyze the following text for grammatical, spelling, and semantic errors. 
        Return a JSON list of errors with the following format:
        [
            {{
                "type": "grammar|spelling|semantic",
                "position": [start, end],
                "error_text": "incorrect text",
                "suggestion": "corrected text",
                "confidence": 0.95
            }}
        ]
        
        Text to analyze: "{text}"
        
        Errors:"""
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from Gemma 3 model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            raise
    
    def _parse_errors(self, response: str, original_text: str, tokens: List[str]) -> List[Dict[str, Any]]:
        """Parse errors from model response"""
        errors = []
        
        try:
            import json
            
            # Try to parse JSON response
            if response.strip().startswith('['):
                parsed_errors = json.loads(response)
                
                for error in parsed_errors:
                    if isinstance(error, dict):
                        errors.append({
                            "type": error.get("type", "unknown"),
                            "position": error.get("position", [0, 0]),
                            "error_text": error.get("error_text", ""),
                            "suggestion": error.get("suggestion", ""),
                            "confidence": error.get("confidence", 0.5)
                        })
            
        except json.JSONDecodeError:
            # Fallback: simple error detection
            self.logger.warning("Failed to parse JSON response, using fallback error detection")
            errors = self._fallback_error_detection(original_text, tokens)
        
        return errors
    
    def _fallback_error_detection(self, text: str, tokens: List[str]) -> List[Dict[str, Any]]:
        """Fallback error detection when JSON parsing fails"""
        errors = []
        
        # Simple rule-based error detection
        common_errors = {
            "teh": "the",
            "recieve": "receive",
            "seperate": "separate",
            "definately": "definitely",
            "occured": "occurred"
        }
        
        for i, token in enumerate(tokens):
            if token.lower() in common_errors:
                errors.append({
                    "type": "spelling",
                    "position": [i, i + 1],
                    "error_text": token,
                    "suggestion": common_errors[token.lower()],
                    "confidence": 0.8
                })
        
        return errors
    
    def get_info(self) -> Dict[str, Any]:
        """Get error detection model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "model_type": "causal_lm"
        }
