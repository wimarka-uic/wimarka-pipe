# MT (Machine Translation) NLP Pipeline

A comprehensive NLP pipeline for machine translation quality assessment, error detection, and text correction using state-of-the-art transformer models.

## 🏗️ Architecture Overview

The pipeline follows a modular architecture with the following flow:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   INPUT     │───▶│  TOKENIZER   │───▶│ EMBEDDINGS   │───▶│TRANSFORMER   │
│   TEXT      │    │ (BERT-based) │    │(Sentence     │    │(Gemma 3)     │
└─────────────┘    └──────────────┘    │Transformers) │    │Error Detection│
                                       └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   OUTPUT    │◀───│  CORRECTION  │◀───│ EXPLANATION  │◀───│MULTI-LABEL   │
│ CORRECTED   │    │(DistilBERT   │    │(Tiny T5)     │    │REGRESSION    │
│   TEXT      │    │Attention)    │    │Explanation   │    │(Scoring)     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## 🔄 Pipeline Flow

### 1. **Input Processing**
- Raw text input validation and preprocessing
- Text normalization and cleaning

### 2. **Tokenization** 
- BERT-based tokenizer for text segmentation
- Handles special tokens and subword tokenization
- Configurable max length and truncation

### 3. **Embedding Generation**
- Sentence Transformers for semantic embeddings
- Generates high-dimensional vector representations
- Supports batch processing for efficiency

### 4. **Error Detection (Gemma 3)**
- Advanced error detection using Google's Gemma 3 model
- Identifies grammatical, spelling, and semantic errors
- Provides confidence scores and suggestions
- JSON-structured error output with positions

### 5. **Multi-Label Regression Scoring**
- DistilBERT-based scoring across multiple dimensions:
  - Grammar Quality
  - Spelling Accuracy
  - Semantic Coherence
  - Fluency
  - Clarity
  - Overall Quality
- Error-aware score adjustment

### 6. **Explanation Generation (Tiny T5)**
- Text-to-text explanations using Tiny T5
- Generates human-readable explanations for errors
- Provides improvement suggestions
- Context-aware explanations

### 7. **Correction Generation (DistilBERT Attention)**
- Attention-based correction using DistilBERT
- Focuses on error regions using attention weights
- Generates corrected text versions
- Maintains semantic coherence

## 📁 Project Structure

```
wimarka-pipe/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── README.md                       # This file
├── pipeline/                       # Main pipeline package
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── pipeline.py                 # Main pipeline orchestrator
│   ├── components/                 # Pipeline components
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # Tokenization component
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── error_detection.py     # Error detection (Gemma 3)
│   │   ├── scoring.py             # Multi-label scoring
│   │   ├── explanation.py         # Explanation generation (T5)
│   │   └── correction.py          # Correction generation (DistilBERT)
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── cache.py               # Caching system
│       └── logger.py              # Logging utilities
├── cache/                         # Model cache directory
└── output/                        # Output directory
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd wimarka-pipe
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the pipeline:**
```bash
python main.py
```

### Basic Usage

```python
from pipeline.config import PipelineConfig
from pipeline.pipeline import MTPipeline

# Initialize configuration
config = PipelineConfig()

# Create pipeline instance
pipeline = MTPipeline(config)

# Process text
input_text = "This is a sample text with some errors."
result = pipeline.process(input_text)

# Access results
print(f"Input: {result.input_text}")
print(f"Errors detected: {len(result.error_detections)}")
print(f"Quality scores: {result.scores}")
print(f"Explanations: {result.explanations}")
print(f"Corrections: {result.corrections}")
print(f"Final output: {result.final_output}")
```

## ⚙️ Configuration

The pipeline is highly configurable through the `PipelineConfig` class:

```python
from pipeline.config import PipelineConfig, ModelConfig

# Custom configuration
config = PipelineConfig(
    tokenizer_config=ModelConfig(
        model_name="bert-base-uncased",
        max_length=512
    ),
    error_detection_config=ModelConfig(
        model_name="google/gemma-3-2b",
        max_length=512
    ),
    # ... other configurations
)
```

### Key Configuration Options

- **Model Selection**: Choose different models for each component
- **Device Management**: Auto-detect or specify CPU/GPU usage
- **Caching**: Enable/disable result caching for performance
- **Batch Processing**: Configure batch sizes for efficiency
- **Logging**: Set log levels and output destinations

## 🔧 Components Details

### Tokenizer Component
- **Model**: BERT-based tokenizer
- **Features**: Subword tokenization, special token handling
- **Configurable**: Max length, truncation, padding

### Embedding Generator
- **Model**: Sentence Transformers
- **Features**: Semantic embeddings, batch processing
- **Output**: High-dimensional vectors

### Error Detector (Gemma 3)
- **Model**: Google Gemma 3 (2B parameters)
- **Features**: Multi-type error detection
- **Output**: Structured error information with positions

### Multi-Label Scorer
- **Model**: DistilBERT with custom scoring head
- **Features**: 6-dimensional quality scoring
- **Output**: Normalized scores (0-1) for each dimension

### Explanation Generator (Tiny T5)
- **Model**: Google T5-small
- **Features**: Text-to-text explanation generation
- **Output**: Human-readable explanations and suggestions

### Correction Generator (DistilBERT Attention)
- **Model**: DistilBERT with attention mechanism
- **Features**: Attention-weighted corrections
- **Output**: Corrected text versions


## 🙏 Acknowledgments

- **Google**: Gemma 3 and T5 models
- **Hugging Face**: Transformers library
- **Sentence Transformers**: Embedding models
- **DistilBERT**: Efficient BERT variant

---

**Note**: This pipeline is designed for research and educational purposes. For production use, additional testing, validation, and optimization may be required.
