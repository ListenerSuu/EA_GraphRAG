# EA-GraphRAG: Efficient and Adaptive GraphRAG

EA-GraphRAG is an efficient and adaptive retrieval framework that automatically routes queries between dense retrieval (DPR), graph-based retrieval, or their fusion based on the syntactic and semantic complexity of the query.

## Overview

This framework implements a complexity-aware retrieval system that:

1. **Analyzes query complexity** using an MLP classifier trained on syntactic, semantic, and dependency-based features
2. **Routes queries** to appropriate retrieval methods:
   - **Low complexity**: Dense Passage Retrieval (DPR) using BGE embeddings
   - **High complexity**: Graph-based retrieval using entity-fact graph and Personalized PageRank
   - **Medium complexity**: Complexity-aware Reciprocal Rank Fusion (RRF) of both methods
3. **Fuses results** using linear-weighted RRF when both methods are used

## Architecture

```
Query → Feature Extraction → Complexity Classification → Retrieval Routing
                                                              ↓
                    ┌─────────────────────────────────────────┼─────────────────────────┐
                    ↓                                         ↓                         ↓
            Low Complexity                            Medium Complexity          High Complexity
                    ↓                                         ↓                         ↓
            Dense Retrieval                          Fusion (RRF)              Graph Retrieval
                    ↓                                         ↓                         ↓
            Retrieved Docs                          Retrieved Docs            Retrieved Docs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for embeddings and classifier)
- Java 8+ (for CoreNLP, optional but recommended for full feature extraction)

### Setup

1. Clone or download this repository.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download SpaCy model:
```bash
python -m spacy download en_core_web_md
```

4. Download Stanza resources:
```bash
python -c "import stanza; stanza.download('en')"
```

5. (Optional) Set up Stanford CoreNLP server for full syntactic feature extraction:
   - Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/
   - Follow instructions to start the server

6. Prepare graph data (for graph retrieval, optional):
   - Extract OpenIE triples from corpus:
     ```bash
     python scripts/extract_openie.py \
         --corpus_path data/merged_4_corpus.json \
         --output_path data/openie/merged_4_openie.json
     ```
   - Build knowledge graph:
     ```bash
     python scripts/build_graph.py \
         --corpus_path data/merged_4_corpus.json \
         --openie_results_path data/openie/merged_4_openie.json \
         --output_graph_path data/graphs/merged_4_graph.pickle
     ```

## Usage

### Basic Usage

```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --top_k 5 \
    --output_path output/merged_4_results.json
```

### Advanced Options

```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --corpus_path data/merged_4_corpus.json \
    --dense_retriever BAAI/bge-base-en-v1.5 \
    --graph_llm gpt-4o-mini \
    --graph_embedding BAAI/bge-base-en-v1.5 \
    --top_k 5 \
    --damping 0.5 \
    --use_corenlp \
    --device cuda \
    --output_path output/merged_4_results.json
```

### Parameters

- `--classifier_path`: Path to trained MLP classifier checkpoint (required)
- `--dataset`: Dataset name (required)
- `--data_file`: Path to input JSON file with queries (required)
- `--corpus_path`: Path to corpus JSON file (default: `data/{dataset}_corpus.json`)
- `--dense_retriever`: Dense retriever model name (default: `BAAI/bge-base-en-v1.5`)
- `--graph_llm`: LLM name for graph construction (default: `gpt-4o-mini`)
- `--graph_embedding`: Embedding model for graph (default: `BAAI/bge-base-en-v1.5`)
- `--top_k`: Number of documents to retrieve (default: 5)
- `--damping`: Damping factor for graph retrieval PPR (default: 0.5)
- `--use_corenlp`: Use CoreNLP for feature extraction (slower but more accurate)
- `--device`: Device for classifier (`cpu` or `cuda`, default: `cpu`)
- `--output_path`: Path to save results (default: `output/{dataset}_results.json`)

## Data Format

### Input Data Format

The input JSON file should contain a list of samples, each with a `question` or `query` field:

```json
[
    {
        "id": "sample_1",
        "question": "What is the capital of France?",
        ...
    },
    {
        "id": "sample_2",
        "question": "Which actor starred in both Inception and The Dark Knight?",
        ...
    }
]
```

### Output Format

The output JSON file contains retrieval results for each query:

```json
[
    {
        "sample_id": "sample_1",
        "query": "What is the capital of France?",
        "complexity_prob": 0.23,
        "complexity_level": "low",
        "method": "dense",
        "retrieved_documents": [...],
        "scores": [...],
        "fusion_weights": null,
        "top_k": 5
    },
    {
        "sample_id": "sample_2",
        "query": "Which actor starred in both Inception and The Dark Knight?",
        "complexity_prob": 0.65,
        "complexity_level": "medium",
        "method": "fusion",
        "retrieved_documents": [...],
        "scores": [...],
        "fusion_weights": [0.35, 0.65],
        "top_k": 5
    }
]
```

## Training the Classifier

To train your own complexity classifier:

1. Prepare training data with labels indicating whether graph retrieval outperforms dense retrieval
2. Run the training script:

```bash
python scripts/train_classifier.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --output_dir models/
```

See `scripts/train_classifier.py` for details.

## Project Structure

```
EA-GraphRAG/
├── main.py                 # Main workflow script
├── src/
│   ├── classifier/        # Complexity classification module
│   │   ├── mlp_classifier.py
│   │   ├── feature_extractor.py
│   │   ├── syntactic_features.py
│   │   └── advanced_features.py
│   ├── retrieval/         # Retrieval modules
│   │   ├── dense_retriever.py
│   │   ├── graph_retriever.py
│   │   └── fusion.py
│   └── utils/             # Utility functions
├── data/                  # Data directory
├── models/                # Trained models
├── output/                # Output results
├── scripts/               # Training and utility scripts
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Complexity Classification

The classifier uses a comprehensive feature set:

### Syntactic Features
- Length measures (MLS, MLT, MLC)
- Sentence complexity (C/S)
- Subordination (C/T, CT/T, DC/C, DC/T)
- Coordination (CP/C, CP/T, T/S)
- Phrasal sophistication (CN/C, CN/T, VP/T)

### Dependency Features
- Dependency distances (max, average, long-range)
- Relation types (subject-verb, object-verb, modifiers, coordination, subordination)
- Tree imbalance metrics

### Semantic Features
- Named entity counts and types
- Entity density
- Semantic role indicators
- Question type encoding

### Lexical Features
- Lexical diversity
- Content-to-function word ratios
- Information density
- Complexity markers (coordination, subordination, negation, passive voice)

## Retrieval Methods

### Dense Retrieval (DPR)
- Uses BGE embeddings for semantic similarity
- Efficient for simple, factoid queries
- Fast retrieval with good performance on low-complexity queries

### Graph Retrieval
- Uses knowledge graph constructed from corpus (entity-fact graph)
- Performs Personalized PageRank (PPR) on the graph
- Effective for multi-hop and complex reasoning queries
- Adaptive PPR damping based on query complexity
- Does not use passage nodes in the graph (entity-fact only)

### Fusion (Complexity-Aware RRF)
- Combines dense and graph retrieval results
- Uses linear weighting based on complexity probability
- Weights: `dense_weight = 1 - complexity_prob`, `graph_weight = complexity_prob`

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{ea_graphrag_2026,
  title={EA-GraphRAG: Efficient and Adaptive GraphRAG for Query Complexity-Aware Retrieval},
  author={Anonymous},
  booktitle={Proceedings of SIGIR 2026},
  year={2026}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- Stanford CoreNLP and Stanza for syntactic parsing
- SpaCy for dependency parsing and NER
- Sentence Transformers for dense embeddings
- python-igraph for graph processing

## Contact

For questions or issues, please open an issue on the repository.

