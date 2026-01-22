# Quick Start Guide

This guide will help you get started with EA-GraphRAG (Efficient and Adaptive GraphRAG) quickly.

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8+ installed
- [ ] At least 16GB RAM
- [ ] (Optional) CUDA-capable GPU
- [ ] (Optional) Java 8+ for CoreNLP

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_md
python -c "import stanza; stanza.download('en')"
```

### Step 2: Prepare Data

Place your data files in the `data/` directory:

- **Corpus file**: `data/{dataset}_corpus.json`
- **Query file**: `data/{dataset}.json` (with `question` or `query` field)

Example query file format:
```json
[
    {"id": "1", "question": "What is machine learning?"},
    {"id": "2", "question": "Which algorithm is used for classification?"}
]
```

### Step 3: Get Pre-trained Model

Download the pre-trained classifier checkpoint:
- Place `mlp_router_4k_optimized_best.pt` in `models/` directory

If you don't have a pre-trained model, see "Training Your Own Classifier" below.

### Step 4: Run Retrieval

```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset your_dataset \
    --data_file data/your_dataset.json \
    --top_k 5
```

Results will be saved to `output/your_dataset_results.json`.

## Example Usage

### Basic Example

```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --top_k 5 \
    --output_path output/merged_4_results.json
```

### With GPU

```bash
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --device cuda \
    --top_k 5
```

### With CoreNLP (Full Features)

```bash
# Start CoreNLP server first (in another terminal)
cd stanford-corenlp-4.5.0
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

# Then run with --use_corenlp flag
python main.py \
    --classifier_path models/mlp_router_4k_optimized_best.pt \
    --dataset merged_4 \
    --data_file data/merged_4.json \
    --use_corenlp \
    --top_k 5
```

## Training Your Own Classifier

If you need to train a classifier on your data:

### Step 1: Prepare Training Data

Create training data with labels (0 = dense better, 1 = graph better):

```json
[
    {
        "question": "What is the capital of France?",
        "label": 0
    },
    {
        "question": "Which actor starred in both Inception and The Dark Knight?",
        "label": 1
    }
]
```

### Step 2: Run Training

```bash
python scripts/train_classifier.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --output_dir models/ \
    --epochs 100 \
    --batch_size 32
```

The best model will be saved to `models/mlp_router_4k_optimized_best.pt`.

## Understanding the Output

The output JSON contains:

```json
{
    "sample_id": "query_1",
    "query": "What is machine learning?",
    "complexity_prob": 0.23,
    "complexity_level": "low",
    "method": "dense",
    "retrieved_documents": ["doc1", "doc2", ...],
    "scores": [0.95, 0.87, ...],
    "fusion_weights": null,
    "top_k": 5
}
```

- **complexity_prob**: Probability that graph retrieval should be used (0-1)
- **complexity_level**: `low`, `medium`, or `high`
- **method**: `dense`, `graph`, or `fusion`
- **fusion_weights**: `[dense_weight, graph_weight]` if method is `fusion`

## Troubleshooting

### Import Errors

If you see import errors:
- Ensure all dependencies in `requirements.txt` are installed
- Check that `sentence-transformers` and `python-igraph` are installed
- Verify Python version compatibility

### Out of Memory

- Reduce `--top_k` parameter
- Use `--device cpu` instead of GPU
- Process queries in smaller batches

### CoreNLP Not Working

- Verify server is running: `curl http://localhost:9000`
- Framework will fall back to SpaCy-only features if CoreNLP unavailable

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [SETUP.md](SETUP.md) for advanced setup options
- See [STRUCTURE.md](STRUCTURE.md) for code organization

## Getting Help

- Check the troubleshooting section in README.md
- Review error messages carefully
- Ensure all dependencies are installed correctly

