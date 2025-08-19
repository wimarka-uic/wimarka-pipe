"""
Multi-Label Regression Scoring Component
"""

import logging
from typing import List, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

from ..config import ModelConfig


class MultiLabelScorer:
    """Multi-label regression scoring component"""
    
    def __init__(self, config: ModelConfig):
        """Initialize scorer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define scoring categories
        self.scoring_categories = [
            "grammar_quality",
            "spelling_accuracy", 
            "semantic_coherence",
            "fluency",
            "clarity",
            "overall_quality"
        ]
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            self.model = AutoModel.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
            
            self.model.to(self.device)
            
            # Initialize scoring head
            self.scoring_head = self._create_scoring_head()
            
            self.logger.info(f"Scoring model loaded: {config.model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load scoring model: {str(e)}")
            raise
    
    def _create_scoring_head(self) -> nn.Module:
        """Create the scoring head for multi-label regression"""
        hidden_size = self.model.config.hidden_size
        num_categories = len(self.scoring_categories)
        
        scoring_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_categories),
            nn.Sigmoid()  # Output scores between 0 and 1
        )
        
        scoring_head.to(self.device)
        return scoring_head
    
    def score(self, text: str, embeddings: List[float], error_detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Score the text across multiple quality dimensions
        
        Args:
            text: Input text to score
            embeddings: Text embeddings
            error_detections: List of detected errors
            
        Returns:
            Dictionary of scores for each category
        """
        try:
            # Get base scores from model
            base_scores = self._get_base_scores(text)
            
            # Adjust scores based on error detections
            adjusted_scores = self._adjust_scores_for_errors(base_scores, error_detections)
            
            # Normalize scores
            final_scores = self._normalize_scores(adjusted_scores)
            
            self.logger.debug(f"Generated scores for {len(self.scoring_categories)} categories")
            return final_scores
            
        except Exception as e:
            self.logger.error(f"Scoring failed: {str(e)}")
            raise
    
    def _get_base_scores(self, text: str) -> Dict[str, float]:
        """Get base scores from the model"""
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
                
                # Get scores from scoring head
                scores = self.scoring_head(pooled_output)
                scores = scores.cpu().numpy()[0]
            
            # Create score dictionary
            score_dict = {}
            for i, category in enumerate(self.scoring_categories):
                score_dict[category] = float(scores[i])
            
            return score_dict
            
        except Exception as e:
            self.logger.error(f"Base scoring failed: {str(e)}")
            # Return default scores
            return {category: 0.5 for category in self.scoring_categories}
    
    def _adjust_scores_for_errors(self, base_scores: Dict[str, float], errors: List[Dict[str, Any]]) -> Dict[str, float]:
        """Adjust scores based on detected errors"""
        adjusted_scores = base_scores.copy()
        
        if not errors:
            return adjusted_scores
        
        # Calculate error penalty
        total_errors = len(errors)
        error_penalty = min(0.3, total_errors * 0.05)  # Max 30% penalty
        
        # Adjust scores based on error types
        for error in errors:
            error_type = error.get("type", "unknown")
            confidence = error.get("confidence", 0.5)
            
            if error_type == "grammar":
                adjusted_scores["grammar_quality"] *= (1 - confidence * 0.2)
            elif error_type == "spelling":
                adjusted_scores["spelling_accuracy"] *= (1 - confidence * 0.3)
            elif error_type == "semantic":
                adjusted_scores["semantic_coherence"] *= (1 - confidence * 0.25)
        
        # Apply overall penalty
        for category in adjusted_scores:
            adjusted_scores[category] = max(0.0, adjusted_scores[category] - error_penalty)
        
        return adjusted_scores
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to ensure they're in valid range"""
        normalized_scores = {}
        
        for category, score in scores.items():
            # Ensure score is between 0 and 1
            normalized_score = max(0.0, min(1.0, score))
            normalized_scores[category] = round(normalized_score, 3)
        
        return normalized_scores
    
    def get_score_summary(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Get a summary of the scores"""
        return {
            "average_score": round(sum(scores.values()) / len(scores), 3),
            "min_score": min(scores.values()),
            "max_score": max(scores.values()),
            "overall_quality": scores.get("overall_quality", 0.0),
            "needs_improvement": [k for k, v in scores.items() if v < 0.6]
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get scoring model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "scoring_categories": self.scoring_categories,
            "num_categories": len(self.scoring_categories)
        }
