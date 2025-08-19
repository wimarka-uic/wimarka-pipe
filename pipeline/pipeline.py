"""
Main MT Pipeline Orchestrator
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .config import PipelineConfig
from .components.tokenizer import Tokenizer
from .components.embeddings import EmbeddingGenerator
from .components.error_detection import ErrorDetector
from .components.scoring import MultiLabelScorer
from .components.explanation import ExplanationGenerator
from .components.correction import CorrectionGenerator
from .utils.cache import CacheManager


@dataclass
class PipelineResult:
    """Result from the pipeline processing"""
    input_text: str
    tokens: List[str]
    embeddings: List[float]
    error_detections: List[Dict[str, Any]]
    scores: Dict[str, float]
    explanations: List[str]
    corrections: List[str]
    final_output: str
    metadata: Dict[str, Any]


class MTPipeline:
    """Main MT NLP Pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(config.cache_dir, config.enable_caching)
        
        # Initialize components
        self.tokenizer = Tokenizer(config.tokenizer_config)
        self.embedding_generator = EmbeddingGenerator(config.embedding_config)
        self.error_detector = ErrorDetector(config.error_detection_config)
        self.scorer = MultiLabelScorer(config.scoring_config)
        self.explanation_generator = ExplanationGenerator(config.explanation_config)
        self.correction_generator = CorrectionGenerator(config.correction_config)
        
        self.logger.info("MT Pipeline initialized successfully")
    
    def process(self, input_text: str) -> PipelineResult:
        """
        Process text through the complete pipeline
        
        Args:
            input_text: Input text to process
            
        Returns:
            PipelineResult: Complete pipeline result
        """
        self.logger.info(f"Processing input text: {input_text[:50]}...")
        
        # Check cache first
        cache_key = self._generate_cache_key(input_text)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self.logger.info("Using cached result")
            return cached_result
        
        try:
            # Step 1: Tokenization
            self.logger.debug("Step 1: Tokenization")
            tokens = self.tokenizer.tokenize(input_text)
            
            # Step 2: Embeddings
            self.logger.debug("Step 2: Generating embeddings")
            embeddings = self.embedding_generator.generate(input_text)
            
            # Step 3: Error Detection using Gemma 3
            self.logger.debug("Step 3: Error detection")
            error_detections = self.error_detector.detect_errors(input_text, tokens)
            
            # Step 4: Multi-Label Regression Scoring
            self.logger.debug("Step 4: Multi-label scoring")
            scores = self.scorer.score(input_text, embeddings, error_detections)
            
            # Step 5: Tiny T5 Explanation Generation
            self.logger.debug("Step 5: Generating explanations")
            explanations = self.explanation_generator.generate_explanations(
                input_text, error_detections, scores
            )
            
            # Step 6: DistilBERT Attention Head Correction
            self.logger.debug("Step 6: Generating corrections")
            corrections = self.correction_generator.generate_corrections(
                input_text, error_detections, explanations
            )
            
            # Step 7: Generate final output
            self.logger.debug("Step 7: Generating final output")
            final_output = self._generate_final_output(
                input_text, corrections, explanations
            )
            
            # Create result
            result = PipelineResult(
                input_text=input_text,
                tokens=tokens,
                embeddings=embeddings,
                error_detections=error_detections,
                scores=scores,
                explanations=explanations,
                corrections=corrections,
                final_output=final_output,
                metadata={
                    "pipeline_version": "1.0.0",
                    "processing_steps": 7,
                    "cache_key": cache_key
                }
            )
            
            # Cache the result
            self.cache_manager.set(cache_key, result)
            
            self.logger.info("Pipeline processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    def _generate_cache_key(self, input_text: str) -> str:
        """Generate cache key for input text"""
        import hashlib
        return hashlib.md5(input_text.encode()).hexdigest()
    
    def _generate_final_output(self, input_text: str, corrections: List[str], explanations: List[str]) -> str:
        """Generate final output combining corrections and explanations"""
        if not corrections:
            return input_text
        
        # Simple approach: apply first correction if available
        # In a real implementation, you might want more sophisticated merging
        return corrections[0] if corrections else input_text
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline components"""
        return {
            "pipeline_name": "MT NLP Pipeline",
            "version": "1.0.0",
            "components": {
                "tokenizer": self.tokenizer.get_info(),
                "embedding_generator": self.embedding_generator.get_info(),
                "error_detector": self.error_detector.get_info(),
                "scorer": self.scorer.get_info(),
                "explanation_generator": self.explanation_generator.get_info(),
                "correction_generator": self.correction_generator.get_info()
            },
            "config": self.config.to_dict()
        }
