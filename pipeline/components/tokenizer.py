"""
Tokenizer Component
"""

import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer

from ..config import ModelConfig


class Tokenizer:
    """Tokenizer component for text tokenization"""
    
    def __init__(self, config: ModelConfig):
        """Initialize tokenizer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir="./cache"
            )
            self.logger.info(f"Tokenizer loaded: {config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        try:
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            
            # Truncate if necessary
            if len(tokens) > self.config.max_length:
                tokens = tokens[:self.config.max_length]
                self.logger.warning(f"Text truncated to {self.config.max_length} tokens")
            
            self.logger.debug(f"Tokenized text into {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            self.logger.error(f"Tokenization failed: {str(e)}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        try:
            encoding = self.tokenizer.encode(
                text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            return encoding[0].tolist()
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {str(e)}")
            raise
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Decoding failed: {str(e)}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get tokenizer information"""
        return {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "vocab_size": self.tokenizer.vocab_size,
            "special_tokens": list(self.tokenizer.special_tokens_map.keys())
        }
