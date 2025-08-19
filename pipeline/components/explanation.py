"""
Explanation Generation Component using Tiny T5
"""

import logging
from typing import List, Dict, Any
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from ..config import ModelConfig


class ExplanationGenerator:
    """Explanation generation component using Tiny T5"""
    
    def __init__(self, config: ModelConfig):
        """Initialize explanation generator with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            self.model = T5ForConditionalGeneration.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
            
            self.model.to(self.device)
            
            self.logger.info(f"Explanation model loaded: {config.model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load explanation model: {str(e)}")
            raise
    
    def generate_explanations(self, text: str, error_detections: List[Dict[str, Any]], scores: Dict[str, float]) -> List[str]:
        """
        Generate explanations for errors and scores using Tiny T5
        
        Args:
            text: Input text
            error_detections: List of detected errors
            scores: Quality scores
            
        Returns:
            List of explanations
        """
        try:
            explanations = []
            
            # Generate explanations for errors
            if error_detections:
                error_explanations = self._generate_error_explanations(text, error_detections)
                explanations.extend(error_explanations)
            
            # Generate explanations for scores
            score_explanations = self._generate_score_explanations(text, scores)
            explanations.extend(score_explanations)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(text, error_detections, scores)
            explanations.append(overall_explanation)
            
            self.logger.debug(f"Generated {len(explanations)} explanations")
            return explanations
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {str(e)}")
            raise
    
    def _generate_error_explanations(self, text: str, errors: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for detected errors"""
        explanations = []
        
        for i, error in enumerate(errors):
            try:
                error_type = error.get("type", "unknown")
                error_text = error.get("error_text", "")
                suggestion = error.get("suggestion", "")
                
                prompt = f"Explain the {error_type} error in this text: '{error_text}'. Suggestion: '{suggestion}'. Provide a clear explanation:"
                
                explanation = self._generate_single_explanation(prompt)
                explanations.append(f"Error {i+1} ({error_type}): {explanation}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate explanation for error {i}: {str(e)}")
                explanations.append(f"Error {i+1}: Unable to generate explanation")
        
        return explanations
    
    def _generate_score_explanations(self, text: str, scores: Dict[str, float]) -> List[str]:
        """Generate explanations for quality scores"""
        explanations = []
        
        # Find areas that need improvement
        low_scores = {k: v for k, v in scores.items() if v < 0.7}
        
        if low_scores:
            for category, score in low_scores.items():
                try:
                    prompt = f"Explain why the {category.replace('_', ' ')} score is {score:.2f} for this text: '{text[:100]}...'. Provide improvement suggestions:"
                    
                    explanation = self._generate_single_explanation(prompt)
                    explanations.append(f"{category.replace('_', ' ').title()}: {explanation}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate score explanation for {category}: {str(e)}")
        
        return explanations
    
    def _generate_overall_explanation(self, text: str, errors: List[Dict[str, Any]], scores: Dict[str, float]) -> str:
        """Generate overall explanation for the text"""
        try:
            num_errors = len(errors)
            avg_score = sum(scores.values()) / len(scores)
            
            prompt = f"""Provide an overall assessment of this text: '{text[:200]}...'
            Number of errors: {num_errors}
            Average quality score: {avg_score:.2f}
            Give a concise summary and recommendations:"""
            
            explanation = self._generate_single_explanation(prompt)
            return f"Overall Assessment: {explanation}"
            
        except Exception as e:
            self.logger.warning(f"Failed to generate overall explanation: {str(e)}")
            return "Overall Assessment: Unable to generate explanation"
    
    def _generate_single_explanation(self, prompt: str) -> str:
        """Generate a single explanation using T5"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.device)
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=3
                )
            
            # Decode explanation
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up explanation
            explanation = explanation.strip()
            if explanation.startswith("explain:"):
                explanation = explanation[8:].strip()
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Single explanation generation failed: {str(e)}")
            return "Unable to generate explanation"
    
    def get_info(self) -> Dict[str, Any]:
        """Get explanation model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "model_type": "text2text"
        }
