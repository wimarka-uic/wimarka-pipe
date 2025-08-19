"""
Pipeline Configuration
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_name: str
    max_length: int = 512
    batch_size: int = 8
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    
    # Model configurations
    tokenizer_config: ModelConfig = ModelConfig(
        model_name="bert-base-uncased",
        max_length=512
    )
    
    embedding_config: ModelConfig = ModelConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=512
    )
    
    error_detection_config: ModelConfig = ModelConfig(
        model_name="google/gemma-3-2b",
        max_length=512
    )
    
    scoring_config: ModelConfig = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=512
    )
    
    explanation_config: ModelConfig = ModelConfig(
        model_name="google/t5-small",
        max_length=256
    )
    
    correction_config: ModelConfig = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=512
    )
    
    # Pipeline settings
    cache_dir: str = "./cache"
    output_dir: str = "./output"
    log_level: str = "INFO"
    
    # Processing settings
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        Path(self.cache_dir).mkdir(exist_ok=True)
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "tokenizer_config": self.tokenizer_config.__dict__,
            "embedding_config": self.embedding_config.__dict__,
            "error_detection_config": self.error_detection_config.__dict__,
            "scoring_config": self.scoring_config.__dict__,
            "explanation_config": self.explanation_config.__dict__,
            "correction_config": self.correction_config.__dict__,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir,
            "log_level": self.log_level,
            "enable_caching": self.enable_caching,
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers
        }
