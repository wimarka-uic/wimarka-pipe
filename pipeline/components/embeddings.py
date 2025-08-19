"""
Embeddings Component
"""

import logging
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer

from ..config import ModelConfig


class EmbeddingGenerator:
    """Embedding generator component"""
    
    def __init__(self, config: ModelConfig):
        """Initialize embedding generator with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = SentenceTransformer(
                config.model_name,
                cache_folder="./cache"
            )
            
            # Set device
            if config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.device
            
            self.model.to(self.device)
            self.logger.info(f"Embedding model loaded: {config.model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate(self, text: str) -> List[float]:
        """
        Generate embeddings for input text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Convert to list
            embedding_list = embeddings.cpu().numpy().tolist()
            
            self.logger.debug(f"Generated embeddings of dimension: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding lists
        """
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )
            
            # Convert to list of lists
            embedding_lists = embeddings.cpu().numpy().tolist()
            
            self.logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embedding_lists
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "device": self.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size
        }
