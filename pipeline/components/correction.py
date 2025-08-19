"""
Correction Generation Component using DistilBERT Attention Head
"""

import logging
from typing import List, Dict, Any
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

from ..config import ModelConfig


class CorrectionGenerator:
    """Correction generation component using DistilBERT attention head"""
    
    def __init__(self, config: ModelConfig):
        """Initialize correction generator with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load tokenizer and model
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            self.model = DistilBertModel.from_pretrained(
                config.model_name,
                cache_dir="./cache",
                output_attentions=True
            )
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
            
            self.model.to(self.device)
            
            # Initialize correction head
            self.correction_head = self._create_correction_head()
            
            self.logger.info(f"Correction model loaded: {config.model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load correction model: {str(e)}")
            raise
    
    def _create_correction_head(self) -> nn.Module:
        """Create the correction head using attention mechanism"""
        hidden_size = self.model.config.hidden_size
        
        correction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.tokenizer.vocab_size)
        )
        
        correction_head.to(self.device)
        return correction_head
    
    def generate_corrections(self, text: str, error_detections: List[Dict[str, Any]], explanations: List[str]) -> List[str]:
        """
        Generate corrections for the input text using DistilBERT attention head
        
        Args:
            text: Input text to correct
            error_detections: List of detected errors
            explanations: List of explanations
            
        Returns:
            List of corrected versions
        """
        try:
            corrections = []
            
            # Generate corrections for each error
            if error_detections:
                for error in error_detections:
                    correction = self._generate_single_correction(text, error, explanations)
                    if correction:
                        corrections.append(correction)
            
            # Generate overall improvement correction
            overall_correction = self._generate_overall_correction(text, error_detections, explanations)
            if overall_correction:
                corrections.append(overall_correction)
            
            self.logger.debug(f"Generated {len(corrections)} corrections")
            return corrections
            
        except Exception as e:
            self.logger.error(f"Correction generation failed: {str(e)}")
            raise
    
    def _generate_single_correction(self, text: str, error: Dict[str, Any], explanations: List[str]) -> str:
        """Generate correction for a single error"""
        try:
            error_type = error.get("type", "unknown")
            error_text = error.get("error_text", "")
            suggestion = error.get("suggestion", "")
            
            # Create context from explanations
            context = " ".join(explanations[:2])  # Use first 2 explanations as context
            
            # Prepare input for correction
            correction_input = f"Error: {error_text} | Type: {error_type} | Suggestion: {suggestion} | Context: {context} | Text: {text}"
            
            # Generate correction using attention mechanism
            corrected_text = self._apply_attention_correction(text, correction_input, error)
            
            return corrected_text
            
        except Exception as e:
            self.logger.warning(f"Failed to generate single correction: {str(e)}")
            return ""
    
    def _generate_overall_correction(self, text: str, errors: List[Dict[str, Any]], explanations: List[str]) -> str:
        """Generate overall correction for the entire text"""
        try:
            # Create comprehensive context
            error_summary = f"Found {len(errors)} errors"
            context = " ".join(explanations[-2:])  # Use last 2 explanations
            
            correction_input = f"Overall improvement needed | {error_summary} | Context: {context} | Text: {text}"
            
            # Apply attention-based correction
            corrected_text = self._apply_attention_correction(text, correction_input, None)
            
            return corrected_text
            
        except Exception as e:
            self.logger.warning(f"Failed to generate overall correction: {str(e)}")
            return ""
    
    def _apply_attention_correction(self, text: str, correction_input: str, error: Dict[str, Any] = None) -> str:
        """Apply attention-based correction to text"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                correction_input,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                
                # Get attention weights from the last layer
                attention_weights = outputs.attentions[-1]  # Last layer attention
                
                # Get hidden states
                hidden_states = outputs.last_hidden_state
                
                # Apply correction head with attention
                correction_logits = self.correction_head(hidden_states)
                
                # Use attention weights to focus on relevant tokens
                if error and "position" in error:
                    start_pos, end_pos = error["position"]
                    # Focus attention on error region
                    attention_focus = attention_weights[:, :, start_pos:end_pos, :]
                    attention_focus = attention_focus.mean(dim=(1, 2, 3), keepdim=True)
                else:
                    # Use average attention across all heads
                    attention_focus = attention_weights.mean(dim=(1, 2, 3), keepdim=True)
                
                # Apply attention-weighted correction
                weighted_logits = correction_logits * attention_focus
                
                # Get predicted tokens
                predicted_tokens = torch.argmax(weighted_logits, dim=-1)
            
            # Decode corrected text
            corrected_text = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
            
            # Clean up the corrected text
            corrected_text = self._clean_corrected_text(corrected_text, text)
            
            return corrected_text
            
        except Exception as e:
            self.logger.error(f"Attention correction failed: {str(e)}")
            return text  # Return original text if correction fails
    
    def _clean_corrected_text(self, corrected_text: str, original_text: str) -> str:
        """Clean and validate corrected text"""
        # Remove extra whitespace
        corrected_text = " ".join(corrected_text.split())
        
        # Ensure the corrected text is not too different from original
        if len(corrected_text) < len(original_text) * 0.5:
            return original_text
        
        # If correction is too long, truncate
        if len(corrected_text) > len(original_text) * 2:
            corrected_text = corrected_text[:len(original_text) * 2]
        
        return corrected_text
    
    def get_info(self) -> Dict[str, Any]:
        """Get correction model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "model_type": "attention_based",
            "vocab_size": self.tokenizer.vocab_size
        }
