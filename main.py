#!/usr/bin/env python3
"""
MT (Machine Translation) NLP Pipeline
Main entry point for the complete pipeline
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.pipeline import MTPipeline
from pipeline.utils.logger import setup_logging

def main():
    """Main entry point for the MT NLP Pipeline"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config = PipelineConfig()
        
        # Initialize pipeline
        pipeline = MTPipeline(config)
        
        # Example usage
        input_text = "This is a sample text for translation and error detection."
        
        logger.info("Starting MT NLP Pipeline...")
        result = pipeline.process(input_text)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Input: {input_text}")
        logger.info(f"Output: {result}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
