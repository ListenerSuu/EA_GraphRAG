# Setup Instructions

This document provides detailed setup instructions for EA-GraphRAG (Efficient and Adaptive GraphRAG).

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended, for embeddings and classifier)
- At least 16GB RAM (32GB recommended)
- Java 8+ (for CoreNLP, optional)

### Python Environment

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLP Models

#### SpaCy Model
```bash
python -m spacy download en_core_web_md
```

#### Stanza Resources
```bash
python -c "import stanza; stanza.download('en')"
```

### 3. Set Up Stanford CoreNLP (Optional)

For full syntactic feature extraction, CoreNLP is recommended:

1. Download Stanford CoreNLP:
   ```bash
   wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.0.zip
   unzip stanford-corenlp-4.5.0.zip
   ```

2. Start CoreNLP server:
   ```bash
   cd stanford-corenlp-4.5.0
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
       -port 9000 -timeout 15000
   ```

The framework will automatically connect to the server if available.

### 4. Prepare Graph Data (Optional, for Graph Retrieval)

For graph-based retrieval, you need to build a knowledge graph:

1. **Extract OpenIE Triples**:
   ```bash
   python scripts/extract_openie.py \
       --corpus_path data/{dataset}_corpus.json \
       --output_path data/openie/{dataset}_openie.json
   ```
   Note: The provided script uses simple pattern matching. For better results, 
   integrate with proper OpenIE tools or LLM-based extraction.

2. **Build Knowledge Graph**:
   ```bash
   python scripts/build_graph.py \
       --corpus_path data/{dataset}_corpus.json \
       --openie_results_path data/openie/{dataset}_openie.json \
       --output_graph_path data/graphs/{dataset}_graph.pickle \
       --synonymy_threshold 0.8 \
       --synonymy_top_k 10
   ```

The graph builder will:
- Extract entities from triples
- Build entity-entity edges from triples
- Add synonymy edges between similar entities (using embeddings)
- Save the graph in igraph pickle format

### 5. Prepare Data

#### Corpus Format

The corpus should be a JSON file with the following structure:

For HotpotQA-style datasets:
```json
{
    "doc_id_1": ["sentence1", "sentence2", ...],
    "doc_id_2": ["sentence1", "sentence2", ...]
}
```

For other datasets:
```json
[
    {
        "title": "Document Title",
        "text": "Document content..."
    },
    ...
]
```

#### Query Data Format

Input queries should be in JSON format:
```json
[
    {
        "id": "query_1",
        "question": "What is the capital of France?"
    },
    ...
]
```

### 6. Download Pre-trained Classifier

Download the pre-trained MLP classifier checkpoint and place it in the `models/` directory:

```bash
mkdir -p models
# Download mlp_router_4k_optimized_best.pt to models/
```

If you don't have a pre-trained model, see the training instructions below.

## Training Your Own Classifier

### Step 1: Prepare Training Data

Your training data should include:
- Query text
- Label indicating whether graph retrieval outperforms dense retrieval (0 or 1)

Format:
```json
[
    {
        "question": "Query text",
        "label": 1  # 1 = graph better, 0 = dense better
    },
    ...
]
```

### Step 2: Run Training Script

```bash
python scripts/train_classifier.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --output_dir models/ \
    --epochs 100 \
    --batch_size 32
```

The script will:
1. Extract features from queries
2. Train the MLP classifier
3. Save the best model to `models/mlp_router_4k_optimized_best.pt`

## Verification

Test the setup with a simple query:

```bash
python -c "
from main import ComplexityAwareRetrievalSystem
system = ComplexityAwareRetrievalSystem(
    classifier_path='models/mlp_router_4k_optimized_best.pt',
    dataset='test_dataset'
)
result = system.retrieve('What is machine learning?')
print(result)
"
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Ensure all dependencies in `requirements.txt` are installed
2. Check that `sentence-transformers` and `python-igraph` are properly installed
3. Verify Python version is 3.8 or higher

### CoreNLP Connection Issues

If CoreNLP features are not working:

1. Verify the server is running: `curl http://localhost:9000`
2. Check firewall settings
3. The framework will fall back to SpaCy-only features if CoreNLP is unavailable

### GPU Issues

If you want to use GPU:

1. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
2. Set `--device cuda` when running `main.py`
3. Ensure your GPU has sufficient memory (at least 4GB recommended)

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce `--top_k` parameter
2. Process queries in smaller batches
3. Use CPU instead of GPU for classifier
4. Reduce batch size in training

## Environment Variables

Optional environment variables:

- `OPENAI_API_KEY`: For LLM-based graph construction (if using OpenAI models)
- `CUDA_VISIBLE_DEVICES`: To specify which GPUs to use
- `STANFORD_CORENLP_HOME`: Path to Stanford CoreNLP installation

## Next Steps

After setup, see the main README.md for usage instructions.

