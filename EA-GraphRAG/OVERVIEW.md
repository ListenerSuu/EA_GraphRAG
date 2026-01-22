# EA-GraphRAG: Efficient and Adaptive GraphRAG - Overview

## Submission Package

This package contains a complete implementation of EA-GraphRAG, an efficient and adaptive retrieval framework that adaptively routes queries between dense and graph-based retrieval methods based on query complexity.

## Package Contents

### Core Code (~2,700 lines)
- **Main workflow**: `main.py` - Integrated retrieval system
- **Classifier module**: MLP-based complexity prediction (~500 lines)
- **Feature extraction**: Syntactic, semantic, dependency features (~600 lines)
- **Retrieval modules**: Dense and graph retrievers (~400 lines)
- **Fusion module**: Complexity-aware RRF (~150 lines)
- **Training script**: Complete training pipeline (~300 lines)

### Documentation (~1,300 lines)
- **README.md**: Comprehensive framework documentation
- **SETUP.md**: Detailed setup instructions
- **QUICKSTART.md**: Quick start guide
- **STRUCTURE.md**: Code organization
- **SUMMARY.md**: Organization summary
- **OVERVIEW.md**: This file

### Configuration Files
- `requirements.txt`: Python dependencies
- `LICENSE`: MIT License
- `.gitignore`: Git ignore rules

## Framework Capabilities

1. **Query Complexity Analysis**
   - Extracts 80+ features from queries
   - Uses MLP classifier to predict complexity
   - Routes queries to appropriate retrieval method

2. **Adaptive Retrieval Routing**
   - Low complexity → Dense retrieval (DPR)
   - High complexity → Graph retrieval
   - Medium complexity → Fusion (complexity-aware RRF)

3. **Result Fusion**
   - Combines dense and graph results
   - Uses linear weighting based on complexity
   - Implements Reciprocal Rank Fusion (RRF)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   python -c "import stanza; stanza.download('en')"
   ```

2. **Run retrieval**:
   ```bash
   python main.py \
       --classifier_path models/mlp_router_4k_optimized_best.pt \
       --dataset merged_4 \
       --data_file data/merged_4.json \
       --top_k 5
   ```

3. **Train classifier** (if needed):
   ```bash
   python scripts/train_classifier.py \
       --train_data data/train.json \
       --test_data data/test.json \
       --output_dir models/
   ```

## Key Features

- ✅ Complete framework implementation
- ✅ Modular and extensible design
- ✅ Comprehensive documentation
- ✅ Training scripts included
- ✅ Anonymous and ready for submission
- ✅ Well-organized code structure

## Requirements

- Python 3.8+
- PyTorch
- SpaCy, Stanza
- Sentence Transformers (for embeddings)
- python-igraph (for graph processing)
- (Optional) Stanford CoreNLP

## File Structure

```
EA-GraphRAG/
├── main.py                    # Main entry point
├── README.md                  # Main documentation
├── SETUP.md                   # Setup guide
├── QUICKSTART.md              # Quick start
├── STRUCTURE.md               # Code structure
├── SUMMARY.md                 # Summary
├── OVERVIEW.md                # This file
├── requirements.txt           # Dependencies
├── LICENSE                    # License
├── .gitignore                # Git ignore
│
├── src/                       # Source code
│   ├── classifier/           # Complexity classification
│   ├── retrieval/            # Retrieval modules
│   └── utils/                # Utilities
│
├── data/                      # Data directory
├── models/                    # Model checkpoints
├── output/                    # Output results
├── scripts/                   # Training scripts
└── configs/                   # Configurations
```

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{ea_graphrag_2026,
  title={EA-GraphRAG: Efficient and Adaptive GraphRAG for Query Complexity-Aware Retrieval},
  author={Anonymous},
  booktitle={Proceedings of SIGIR 2026},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please refer to the documentation files or open an issue in the repository.

