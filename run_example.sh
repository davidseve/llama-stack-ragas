#!/bin/bash
# Complete RAG Evaluation Workflow with RAGAS
# This script executes three steps:
#   1. Upload documents to Milvus (milvus-upload.py)
#   2. Generate RAG dataset with answers and contexts (rag.py)
#   3. Evaluate RAG quality with RAGAS metrics (evaluate_ragas.py)
#
# Based on Red Hat OpenShift AI Self-Managed 3.0 best practices

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Load configuration from .env if it exists
if [ -f ".env" ]; then
    echo "📝 Loading configuration from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

LLAMA_STACK_URL="${LLAMA_STACK_URL:-https://llama-stack-example-llama-stack-example.apps.ocp.sandbox5435.opentlc.com}"
MODEL_ID="${MODEL_ID:-vllm-inference/llama-4-scout-17b-16e-w4a16}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-granite-embedding-125m}"
DOCUMENTS_DIR="${DOCUMENTS_DIR:-documents}"
INPUT_DATASET="${INPUT_DATASET:-dataset-base/millbrook_dataset.json}"
OUTPUT_DATASET="${OUTPUT_DATASET:-output/millbrook_ragas_dataset.json}"
RESULTS_FILE="${RESULTS_FILE:-output/millbrook_ragas_results.json}"

# Advanced options
VERIFY_SSL="${VERIFY_SSL:-false}"
VECTOR_STORE_ID="${VECTOR_STORE_ID:-}"

# Auto-detect if we should skip upload based on VECTOR_STORE_ID
if [ -n "$VECTOR_STORE_ID" ]; then
    SKIP_UPLOAD="true"
else
    SKIP_UPLOAD="${SKIP_UPLOAD:-false}"
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "RAG EVALUATION WORKFLOW WITH RAGAS"
echo "======================================================================"
echo "Llama Stack URL:    $LLAMA_STACK_URL"
echo "Model:              $MODEL_ID"
echo "Embedding Model:    $EMBEDDING_MODEL"
echo "Documents:          $DOCUMENTS_DIR"
echo "Input Dataset:      $INPUT_DATASET"
echo "Output Dataset:     $OUTPUT_DATASET"
echo "Results File:       $RESULTS_FILE"
echo "======================================================================"
echo ""

# ============================================================================
# STEP 1: Upload Documents to Milvus
# ============================================================================
if [ "$SKIP_UPLOAD" = "true" ] && [ -n "$VECTOR_STORE_ID" ]; then
    echo "======================================================================"
    echo "STEP 1/3: Using Existing Vector Store (Skipping Upload)"
    echo "======================================================================"
    echo ""
    echo "⏭️  Skipping document upload (Vector Store ID already configured)"
    echo "📌 Using Vector Store ID: $VECTOR_STORE_ID"
    echo ""
    echo "✅ Step 1 Skipped - Using existing vector store"
    echo ""
else
    echo "======================================================================"
    echo "STEP 1/3: Uploading Documents to Milvus"
    echo "======================================================================"
    echo ""
    
    # Run milvus-upload.py and capture the vector store ID
    UPLOAD_OUTPUT=$(python milvus-upload.py \
        --url "$LLAMA_STACK_URL" \
        --documents-dir "$DOCUMENTS_DIR" \
        --embedding-model "$EMBEDDING_MODEL" \
        $([ "$VERIFY_SSL" = "true" ] && echo "--verify-ssl") \
        2>&1)
    
    echo "$UPLOAD_OUTPUT"
    
    # Extract Vector Store ID from output
    VECTOR_STORE_ID=$(echo "$UPLOAD_OUTPUT" | grep -oP "Vector Store ID: \K[^ ]+$" | tail -1)
    
    if [ -z "$VECTOR_STORE_ID" ]; then
        echo ""
        echo "❌ ERROR: Could not extract Vector Store ID from upload output"
        exit 1
    fi
    
    echo ""
    echo "✅ Step 1 Complete - Vector Store ID: $VECTOR_STORE_ID"
    echo ""
fi

# ============================================================================
# STEP 2: Generate RAG Dataset
# ============================================================================
echo "======================================================================"
echo "STEP 2/3: Generating RAG Dataset with Answers and Contexts"
echo "======================================================================"
echo ""

python rag.py \
    --url "$LLAMA_STACK_URL" \
    --model-id "$MODEL_ID" \
    --vector-store-id "$VECTOR_STORE_ID" \
    --input "$INPUT_DATASET" \
    --output "$OUTPUT_DATASET" \
    $([ "$VERIFY_SSL" = "true" ] && echo "--verify-ssl")

echo ""
echo "✅ Step 2 Complete - RAG dataset saved to: $OUTPUT_DATASET"
echo ""

# ============================================================================
# STEP 3: Evaluate with RAGAS
# ============================================================================
echo "======================================================================"
echo "STEP 3/3: Evaluating RAG Quality with RAGAS Metrics"
echo "======================================================================"
echo ""

python evaluate_ragas.py \
    --url "$LLAMA_STACK_URL" \
    --model-id "$MODEL_ID" \
    --dataset "$OUTPUT_DATASET" \
    --output "$RESULTS_FILE" \
    --batch-size 1 \
    $([ "$VERIFY_SSL" = "true" ] && echo "--verify-ssl")

echo ""
echo "✅ Step 3 Complete - Evaluation results saved to: $RESULTS_FILE"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================"
echo "🎉 WORKFLOW COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "📊 Generated Files:"
echo "   - RAG Dataset:    $OUTPUT_DATASET"
echo "   - RAGAS Results:  $RESULTS_FILE"
echo ""
echo "🔍 Vector Store ID:  $VECTOR_STORE_ID"
echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo "1. Review the RAGAS evaluation results:"
echo "   cat $RESULTS_FILE | jq '.metrics'"
echo ""
echo "2. Re-run evaluation without re-uploading documents:"
echo "   SKIP_UPLOAD=true VECTOR_STORE_ID=$VECTOR_STORE_ID ./run_example.sh"
echo ""
echo "3. Evaluate with specific metrics only:"
echo "   python evaluate_ragas.py --dataset $OUTPUT_DATASET \\"
echo "       --metrics answer_relevancy faithfulness --batch-size 1"
echo "======================================================================"
echo ""

