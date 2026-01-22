# Code Organization Summary

This document summarizes the code organization for EA-GraphRAG (Efficient and Adaptive GraphRAG) submission.

## What Was Done

1. **Created Anonymous Code Package**
   - All code organized in `/home/sdong/EA-GraphRAG/`
   - Original code preserved in original locations
   - No personal information or API keys included

2. **Organized Framework Components**
   - **Classifier Module**: MLP-based complexity prediction with feature extraction
   - **Retrieval Modules**: Dense (DPR) and Graph-based retrievers
   - **Fusion Module**: Complexity-aware RRF for combining results
   - **Main Workflow**: Integrated system that routes queries based on complexity

3. **Created Documentation**
   - **README.md**: Comprehensive framework documentation
   - **SETUP.md**: Detailed setup instructions
   - **QUICKSTART.md**: Quick start guide
   - **STRUCTURE.md**: Code organization details
   - **SUMMARY.md**: This file

4. **Provided Training Scripts**
   - `scripts/train_classifier.py`: Complete training pipeline

## Framework Architecture

```
Query → Feature Extraction → Complexity Classification
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            Low Complexity    Medium Complexity  High Complexity
                    ↓               ↓               ↓
            Dense Retrieval    Fusion (RRF)   Graph Retrieval
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                            Retrieved Documents
```

## Key Components

### 1. Complexity Classifier (`src/classifier/`)
- **MLP Model**: OptimizedMLPClassifier with attention and residual connections
- **Feature Extraction**: Syntactic, semantic, dependency, and lexical features
- **Training**: Script provided for training on custom data

### 2. Retrieval Modules (`src/retrieval/`)
- **Dense Retriever**: Implements DPR using BGE embeddings
- **Graph Retriever**: Implements graph-based retrieval with entity-fact graph and PPR
- **Fusion**: Complexity-aware RRF with linear weighting

### 3. Main Workflow (`main.py`)
- Integrates all components
- Routes queries based on complexity
- Saves results in structured JSON format

## File Organization

```
EA-GraphRAG/
├── main.py                 # Main entry point
├── README.md              # Main documentation
├── SETUP.md               # Setup instructions
├── QUICKSTART.md          # Quick start guide
├── STRUCTURE.md           # Code structure
├── SUMMARY.md             # This file
├── LICENSE                # MIT License
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
│
├── src/                   # Source code
│   ├── classifier/       # Complexity classification
│   ├── retrieval/        # Retrieval modules
│   └── utils/            # Utilities
│
├── data/                 # Data directory
├── models/               # Model checkpoints
├── output/               # Output results
├── scripts/              # Training scripts
└── configs/               # Configuration files
```

## Dependencies

### External Systems
- **Sentence Transformers**: For dense embeddings
- **python-igraph**: For graph processing and PPR
- **Stanford CoreNLP**: Optional, for full syntactic features

### Python Packages
See `requirements.txt` for complete list. Key packages:
- PyTorch
- SpaCy, Stanza
- Transformers, Sentence-Transformers
- NumPy, scikit-learn

## Usage

### Basic Usage
```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --top_k 5
```

### Training
```bash
python scripts/train_classifier.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --output_dir models/
```

## Key Features

1. **Comprehensive Feature Extraction**
   - Syntactic complexity measures (MLS, MLT, C/S, etc.)
   - Dependency parsing features
   - Semantic and lexical features
   - Tree structure features

2. **Adaptive Routing**
   - Low complexity → Dense retrieval
   - High complexity → Graph retrieval
   - Medium complexity → Fusion

3. **Complexity-Aware Fusion**
   - Linear weighting based on complexity probability
   - RRF-based combination of results

4. **Modular Design**
   - Easy to extend with new features
   - Easy to add new retrieval methods
   - Configurable via command-line arguments

## Notes for Reviewers

1. **Original Code Locations**
   - Original code remains untouched in original locations
   - This package is a clean, organized version for submission

2. **Dependencies**
   - Framework is self-contained with all retrieval code included
   - Only requires standard Python packages (see requirements.txt)
   - Graph data needs to be prepared separately (see SETUP.md)

3. **Anonymization**
   - All personal information removed
   - API keys removed
   - Paths made configurable
   - Code is ready for anonymous submission

4. **Completeness**
   - All core components included
   - Training script provided
   - Documentation complete
   - Ready for reproduction

## Contact

For questions about the code organization or setup, please refer to:
- README.md for general documentation
- SETUP.md for setup issues
- QUICKSTART.md for quick start
- STRUCTURE.md for code organization

