# RAG Evaluation with RAGAS and Llama Stack

Complete workflow for RAG system evaluation using RAGAS metrics through Llama Stack SDK.

This project follows Red Hat OpenShift AI Self-Managed 3.0 best practices using the **inline provider mode**.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Step 1: Upload documents
python milvus-upload.py --documents-dir documents

# Step 2: Generate RAG dataset (use the vector store ID from step 1)
python rag.py --vector-store-id <your-vector-store-id>

# Step 3: Evaluate with RAGAS
python evaluate_ragas.py --dataset output/millbrook_ragas_dataset.json --batch-size 1
```

Or run the automated workflow:

```bash
./run_example.sh
```

---

## Step 1: Upload Documents (`milvus-upload.py`)

Uploads documents to a Milvus vector store using Llama Stack.

### Usage

```bash
python milvus-upload.py --documents-dir documents --embedding-model granite-embedding-125m
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | OpenShift sandbox URL | Llama Stack server URL |
| `--documents-dir` | `documents` | Directory containing documents |
| `--embedding-model` | `granite-embedding-125m` | Embedding model (**must match server config**) |
| `--verify-ssl` | False | Enable SSL verification |
| `--timeout` | 300 | Request timeout in seconds |

### Available Embedding Models

| Model | Dimension | Notes |
|-------|-----------|-------|
| `granite-embedding-125m` | 768 | **DEFAULT** - matches server config |
| `sentence-transformers/nomic-ai/nomic-embed-text-v1.5` | 768 | Alternative |

### Output

```
✓ Vector Store ID: vs_627e6e71-8a1b-45cc-bea5-d7689e71e27b
```

Save this ID for the next steps.

---

## Step 2: Generate RAG Dataset (`rag.py`)

Processes questions through the RAG system to generate answers and contexts.

### Usage

```bash
python rag.py \
  --vector-store-id vs_627e6e71-8a1b-45cc-bea5-d7689e71e27b \
  --input dataset-base/millbrook_dataset.json \
  --output output/millbrook_ragas_dataset.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | OpenShift sandbox URL | Llama Stack server URL |
| `--model-id` | `vllm-inference/llama-4-scout-17b-16e-w4a16` | LLM model |
| `--vector-store-id` | *(required)* | Vector store ID from Step 1 |
| `--input` | `dataset-base/millbrook_dataset.json` | Input questions JSON |
| `--output` | `output/millbrook_ragas_dataset.json` | Output dataset JSON |
| `--verify-ssl` | False | Enable SSL verification |
| `--timeout` | 300 | Request timeout in seconds |

### Input Format

```json
[
  {
    "id": "q001",
    "question": "What is the capital of France?",
    "ground_truth": "Paris is the capital of France."
  }
]
```

### Output Format

```json
[
  {
    "id": "q001",
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "contexts": ["Paris is the capital and most populous city..."],
    "ground_truth": "Paris is the capital of France."
  }
]
```

---

## Step 3: Evaluate with RAGAS (`evaluate_ragas.py`)

Runs RAGAS evaluation using the Llama Stack inline provider.

### Usage

```bash
# Recommended: use --batch-size 1 for reliability
python evaluate_ragas.py \
  --dataset output/millbrook_ragas_dataset.json \
  --output output/results.json \
  --batch-size 1
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | OpenShift sandbox URL | Llama Stack server URL |
| `--model-id` | `vllm-inference/llama-4-scout-17b-16e-w4a16` | LLM model for evaluation |
| `--embedding-model` | `granite-embedding-125m` | Embedding model |
| `--dataset` | *(required)* | Input RAGAS dataset JSON |
| `--output` | *(auto-generated)* | Output results JSON |
| `--metrics` | `answer_relevancy faithfulness context_precision context_recall` | Metrics to compute |
| `--batch-size` | None | Entries per batch (**use 1 for reliability**) |
| `--max-wait` | 900 | Max seconds to wait for job |
| `--poll-interval` | 5 | Seconds between status checks |

### Available Metrics

| Metric | Description |
|--------|-------------|
| `answer_relevancy` | How relevant is the answer to the question |
| `faithfulness` | Factual consistency with retrieved contexts |
| `context_precision` | Relevance of retrieved contexts |
| `context_recall` | Coverage of relevant information |
| `context_utilization` | How well the answer uses contexts |
| `semantic_similarity` | Similarity to ground truth |

### Example Output

```
======================================================================
📊 RAGAS EVALUATION RESULTS
======================================================================
🎯 Aggregated Metrics:
----------------------------------------------------------------------
  ✅ answer_relevancy         : 0.9655
  ✅ faithfulness             : 0.9556
  ⚠️ context_precision        : 0.6092
  ⚠️ context_recall           : 0.7667
----------------------------------------------------------------------
```

### Results JSON

```json
{
  "benchmark_id": "ragas_benchmark_batch_run_20251203_194538",
  "timestamp": "2025-12-03T19:56:40.953686",
  "metrics": {
    "answer_relevancy": 0.9655,
    "faithfulness": 0.9556,
    "context_precision": 0.6092,
    "context_recall": 0.7667
  },
  "individual_scores": {
    "answer_relevancy": [0.888, 1.0, 0.967],
    "faithfulness": [0.5, 1.0, 0.667]
  },
  "generations": [...],
  "failures": []
}
```

### Interpreting Scores

| Score | Status | Meaning |
|-------|--------|---------|
| **> 0.8** | ✅ Excellent | System performing well |
| **0.6 - 0.8** | ⚠️ Good | Room for improvement |
| **< 0.6** | ❌ Needs Work | Significant improvement needed |

---

## Project Structure

```
llama-stack-ragas/
├── milvus-upload.py          # Step 1: Upload documents
├── rag.py                    # Step 2: Generate RAG dataset
├── evaluate_ragas.py         # Step 3: RAGAS evaluation
├── run_example.sh            # Automated workflow
├── requirements.txt
├── dataset-base/
│   └── millbrook_dataset.json
├── documents/
│   └── *.md                  # Source documents
└── output/
    ├── millbrook_ragas_dataset.json
    └── millbrook_ragas_results.json
```

---

## Troubleshooting

### Step 1: Upload Issues

**Vector store creation fails?**
- Verify embedding model is available on server
- Default `granite-embedding-125m` must match server config

### Step 2: RAG Issues

**Empty contexts?**
- Verify documents were uploaded successfully
- Check vector store ID is correct

### Step 3: Evaluation Issues

**Job fails or times out?**
- Use `--batch-size 1` for reliability
- Increase `--max-wait` for large datasets

**Empty scores?**
- Verify `trustyai_ragas_inline` provider is configured on server
- Check LLM and embedding models are accessible

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Llama Stack Server                                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Inline RAGAS Provider (trustyai_ragas_inline)   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   LLM Model  │  │   Embedding  │  │   Milvus    │  │
│  │   (vLLM)     │  │   Model      │  │  (inline)   │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
                      ▲
                      │ Llama Stack SDK
                      │
┌─────────────────────┴───────────────────────────────────┐
│  1. milvus-upload.py  → Upload documents                │
│  2. rag.py            → Generate dataset                │
│  3. evaluate_ragas.py → Run RAGAS evaluation            │
└─────────────────────────────────────────────────────────┘
```

---

## Technical Notes

### RAGAS Inline Provider

The evaluation script passes these parameters to the server:

```python
extra_body = {
    "provider_id": "trustyai_ragas_inline",
    "judge_model": "<llm-model-id>",
    "embedding_model": "<embedding-model-id>"
}
```

### Server Requirements

- `trustyai_ragas_inline` provider enabled
- LLM model available (e.g., `vllm-inference/llama-4-scout-17b-16e-w4a16`)
- Embedding model available (e.g., `granite-embedding-125m`)
- `builtin::rag` toolgroup configured

### Known Limitations

- **Large batch evaluations fail**: The inline RAGAS provider has timeout/resource limits. With 10 entries × 4 metrics = 40+ LLM calls per job, the server often fails after ~5 minutes. Use `--batch-size 1` to process entries individually (slower but reliable).

- **`extra_body` required**: The `run_eval()` call must include `extra_body` with `provider_id`, `judge_model`, and `embedding_model`. Without these parameters, the server doesn't know which LLM to use as judge and fails silently.

- Warning `is_finished not implemented for LlamaStackInlineLLM` can be ignored

- Evaluation jobs run synchronously; long datasets take time

### Why `--batch-size 1` is Required

| Without batch-size | With --batch-size 1 |
|-------------------|---------------------|
| 10 entries in 1 job | 1 entry per job (10 jobs) |
| 40+ LLM calls at once | ~4 LLM calls per job |
| Job runs ~5+ min → **fails** | Each job ~30-60 sec → **works** |

The inline provider (`trustyai_ragas_inline`) runs RAGAS locally on the Llama Stack server. Large batches exhaust server resources or hit internal timeouts. Single-entry batches are more reliable.
