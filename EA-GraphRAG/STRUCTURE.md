# Project Structure

This document describes the organization of EA-GraphRAG (Efficient and Adaptive GraphRAG).

## Directory Structure

```
EA-GraphRAG/
├── main.py                      # Main entry point for the framework
├── README.md                    # Main documentation
├── SETUP.md                     # Detailed setup instructions
├── STRUCTURE.md                 # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── classifier/              # Query complexity classification
│   │   ├── __init__.py
│   │   ├── mlp_classifier.py   # MLP classifier model
│   │   ├── feature_extractor.py # Feature extraction wrapper
│   │   ├── syntactic_features.py # Syntactic complexity features
│   │   └── advanced_features.py  # Advanced semantic/dependency features
│   │
│   ├── retrieval/               # Retrieval modules
│   │   ├── __init__.py
│   │   ├── dense_retriever.py   # DPR-based dense retrieval
│   │   ├── graph_retriever.py    # Graph-based retrieval
│   │   └── fusion.py            # Complexity-aware RRF fusion
│   │
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── data/                        # Data directory
│   └── .gitkeep
│
├── models/                      # Trained models
│   └── .gitkeep
│
├── output/                       # Output results
│   └── .gitkeep
│
└── scripts/                      # Utility scripts
    └── train_classifier.py       # Training script for classifier
```

## Component Descriptions

### Main Entry Point

- **main.py**: Main workflow script that integrates all components
  - Initializes classifier, retrievers, and fusion module
  - Processes queries and routes based on complexity
  - Saves results to JSON

### Classifier Module (`src/classifier/`)

- **mlp_classifier.py**: 
  - `OptimizedMLPClassifier`: MLP model with attention and residual connections
  - `load_classifier()`: Load trained model from checkpoint
  - `predict_complexity()`: Predict complexity and routing decision

- **feature_extractor.py**:
  - `QueryComplexityFeaturizer`: Main featurizer class
  - Combines syntactic and advanced features
  - Handles normalization and feature selection

- **syntactic_features.py**:
  - `SyntacticComplexityAnalyzer`: Extracts syntactic complexity measures
  - Uses Stanza and CoreNLP for parsing
  - Computes ratios (MLS, MLT, C/S, etc.)

- **advanced_features.py**:
  - `extract_advanced_features()`: Extracts comprehensive features
  - Dependency parsing features
  - Semantic and lexical features
  - Tree structure features

### Retrieval Module (`src/retrieval/`)

- **dense_retriever.py**:
  - `DenseRetriever`: Implements DPR retrieval
  - Uses BGE embeddings via Sentence Transformers
  - Caches document embeddings for efficiency

- **graph_retriever.py**:
  - `GraphRetriever`: Implements graph-based retrieval
  - Uses entity-fact graph with Personalized PageRank
  - Supports adaptive PPR damping based on complexity
  - Does not use passage nodes (entity-fact only)

- **fusion.py**:
  - `ComplexityAwareRRF`: Reciprocal Rank Fusion with complexity weighting
  - Linear, exponential, or sigmoid weighting modes
  - Combines dense and graph results

### Scripts (`scripts/`)

- **train_classifier.py**: 
  - Training script for MLP classifier
  - Feature extraction and selection
  - Model training with early stopping

## Data Flow

1. **Query Input** → `main.py`
2. **Feature Extraction** → `QueryComplexityFeaturizer`
3. **Complexity Prediction** → `OptimizedMLPClassifier`
4. **Routing Decision**:
   - Low complexity → `DenseRetriever`
   - High complexity → `GraphRetriever`
   - Medium complexity → `ComplexityAwareRRF` (fuses both)
5. **Output** → JSON results file

## Dependencies

### External Libraries
- Sentence Transformers: For dense embeddings
- python-igraph: For graph processing and PPR
- Stanford CoreNLP: For full syntactic parsing (optional)
- Stanza: For constituency parsing
- SpaCy: For dependency parsing and NER

### Python Packages
See `requirements.txt` for full list.

## File Naming Conventions

- Python modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Extension Points

To extend the framework:

1. **Add new features**: Modify `advanced_features.py` or `syntactic_features.py`
2. **Add new retrieval method**: Create new class in `src/retrieval/` and update `main.py`
3. **Modify fusion strategy**: Update `ComplexityAwareRRF` in `fusion.py`
4. **Change routing logic**: Modify routing in `main.py` `retrieve()` method

## Notes

- The framework is designed to be modular and extensible
- All paths are configurable via command-line arguments
- The code is anonymized for submission
- Original code locations are preserved in comments where appropriate

