#!/usr/bin/env python3
"""
RAGAS Evaluation Script using Llama Stack SDK (Inline Mode)

This script evaluates RAG systems using RAGAS metrics through Llama Stack's
evaluation API. It uses the inline provider mode as recommended by Red Hat 
OpenShift AI Self-Managed 3.0 documentation.

Based on:
- Red Hat OpenShift AI Self-Managed 3.0 - Evaluating RAG systems with RAGAS
- Llama Stack Provider RAGAS - basic_demo.ipynb (inline mode)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from llama_stack_client import LlamaStackClient


# RAGAS metrics available
AVAILABLE_METRICS = [
    "answer_relevancy",      # Measures how relevant the answer is to the question
    "faithfulness",          # Measures factual consistency with retrieved contexts
    "context_precision",     # Measures relevance of retrieved contexts to the question
    "context_recall",        # Measures if all relevant information is retrieved
    "context_utilization",   # Measures how well the answer uses retrieved contexts
    "semantic_similarity",   # Measures semantic similarity between answer and reference
]

DEFAULT_METRICS = [
    "answer_relevancy",      # Measures how relevant the answer is to the question
    "faithfulness",          # Measures factual consistency with retrieved contexts
    "context_precision",     # Measures relevance of retrieved contexts to the question
    "context_recall",        # Measures if all relevant information is retrieved
]


def load_ragas_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load RAGAS dataset from JSON file.
    
    Expected format:
    [
        {
            "id": "q001",
            "question": "What is...?",
            "answer": "The answer is...",
            "contexts": ["Context 1", "Context 2"],
            "ground_truth": "Ground truth answer"
        }
    ]
    
    Args:
        dataset_path: Path to the RAGAS dataset JSON file
        
    Returns:
        List of evaluation entries
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array")
    
    # Validate required fields
    required_fields = ["question", "answer", "contexts"]
    for i, entry in enumerate(dataset):
        missing = [f for f in required_fields if f not in entry]
        if missing:
            raise ValueError(f"Entry {i} missing required fields: {missing}")
    
    return dataset


def convert_to_ragas_format(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert dataset to RAGAS evaluation format.
    
    RAGAS expects:
    - user_input: The question/query
    - retrieved_contexts: List of retrieved context strings
    - response: The generated answer
    - reference: The ground truth (optional but recommended)
    
    Args:
        dataset: Input dataset with question/answer/contexts
        
    Returns:
        Dataset in RAGAS format
    """
    ragas_data = []
    
    for entry in dataset:
        ragas_entry = {
            "user_input": entry["question"],
            "response": entry["answer"],
            "retrieved_contexts": entry["contexts"],
        }
        
        # Add ground truth if available
        if "ground_truth" in entry and entry["ground_truth"]:
            ragas_entry["reference"] = entry["ground_truth"]
        
        ragas_data.append(ragas_entry)
    
    return ragas_data


def register_dataset(
    client: LlamaStackClient,
    dataset_id: str,
    ragas_data: List[Dict[str, Any]]
) -> None:
    """
    Register dataset with Llama Stack's Datasets API.
    
    Args:
        client: LlamaStackClient instance
        dataset_id: Unique identifier for the dataset
        ragas_data: Dataset in RAGAS format
    """
    # Unregister dataset if it already exists
    try:
        client.datasets.unregister(dataset_id)
        # print(f"  ℹ️  Unregistered existing dataset: {dataset_id}")
    except Exception:
        pass  # Dataset doesn't exist, which is fine
    
    # Register new dataset
    print(f"📝 Registering dataset: {dataset_id}")
    print(f"   Entries: {len(ragas_data)}")
    
    # Use the correct API format for llama-stack-client
    # Purpose: 'eval/question-answer' for RAGAS evaluation
    # Source: rows data source with the evaluation data
    client.datasets.register(
        purpose="eval/question-answer",
        source={
            "type": "rows",
            "rows": ragas_data,
        },
        dataset_id=dataset_id,
    )
    
    print(f"✓ Dataset registered successfully")


def register_benchmark(
    client: LlamaStackClient,
    benchmark_id: str,
    dataset_id: str,
    metrics: List[str]
) -> None:
    """
    Register benchmark with Llama Stack eval API.
    
    Args:
        client: LlamaStackClient instance
        benchmark_id: Benchmark identifier
        dataset_id: Dataset identifier
        metrics: List of RAGAS metrics
    """
    print(f"\n📊 Registering benchmark: {benchmark_id}")
    print(f"   Dataset: {dataset_id}")
    print(f"   Metrics: {', '.join(metrics)}")
    
    # Use SDK's internal HTTP client to register benchmark
    url = f"{client.base_url}/v1alpha/eval/benchmarks"
    payload = {
        "benchmark_id": benchmark_id,
        "dataset_id": dataset_id,
        "scoring_functions": metrics,
        "provider_id": "trustyai_ragas_inline"
    }
    
    try:
        response = client._client.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Benchmark registered successfully")
    except Exception as e:
        # Benchmark might already exist, which is OK
        if "already exists" in str(e).lower():
            print(f"  ℹ️  Benchmark already registered")
        else:
            print(f"  ⚠️  Registration error: {e}")
            raise


def run_evaluation(
    client: LlamaStackClient,
    benchmark_id: str,
    ragas_data: List[Dict[str, Any]],
    metrics: List[str],
    model_id: str,
    embedding_model_id: Optional[str] = None,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
) -> Any:
    """
    Run RAGAS evaluation using inline provider with evaluate_rows API.
    
    Args:
        client: LlamaStackClient instance
        benchmark_id: Benchmark identifier for this evaluation
        ragas_data: Data rows in RAGAS format
        metrics: List of RAGAS metrics to compute
        model_id: LLM model to use for evaluation
        embedding_model_id: Embedding model (optional, uses default if not provided)
        max_wait_seconds: Maximum seconds to wait for the remote job
        poll_interval: Seconds between job status checks
        
    Returns:
        Evaluation results
    """
    print(f"\n🚀 Running evaluation...")
    print(f"   Benchmark ID: {benchmark_id}")
    print(f"   Metrics: {', '.join(metrics)}")
    print(f"   LLM Model: {model_id}")
    if embedding_model_id:
        print(f"   Embedding Model: {embedding_model_id}")
    print(f"   Evaluating {len(ragas_data)} entries...")
    
    # Prepare benchmark config following the exact SDK structure
    # Based on BenchmarkConfigParam, SamplingParams, and ScoringFnParamsParam types
    
    # 1. Define eval_candidate (the model to evaluate)
    # IMPORTANTE: max_tokens debe ser > 0 para evitar error 400
    eval_candidate = {
        "type": "model",
        "model": model_id,
        "sampling_params": {
            "strategy": {"type": "greedy"},
            "max_tokens": 2048  # Fix for "max_tokens must be at least 1, got 0" error
        }
    }
    
    # 2. Define scoring_params (one entry per metric)
    # For RAGAS metrics, we use "basic" type scoring with provider_id in extra_body
    scoring_params = {}
    for metric in metrics:
        scoring_params[metric] = {
            "type": "basic",
            "aggregation_functions": ["average"]
        }
    
    # 3. Create the complete benchmark_config
    benchmark_config = {
        "eval_candidate": eval_candidate,
        "scoring_params": scoring_params,
    }
    
    # 4. Build extra_body for RAGAS-specific parameters
    # Note: provider is "trustyai_ragas_inline" based on error message
    # embedding_model is required by the server
    extra_body = {
        "provider_id": "trustyai_ragas_inline",
        "judge_model": model_id,
        "embedding_model": embedding_model_id or "granite-embedding-125m",
    }
    
    # Run evaluation with run_eval (as per server error message)
    # This returns a Job object
    job = client.alpha.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config=benchmark_config,
        extra_body=extra_body,
    )
    
    print(f"✓ Evaluation job started")
    print(f"   Job ID: {job.job_id}")
    
    # Wait for the job to complete and retrieve results
    print(f"\n📥 Waiting for evaluation to complete...")
    
    import time
    waited = 0
    last_status = None
    
    while True:
        try:
            # Check job status
            job_status = client.alpha.eval.jobs.status(
                benchmark_id=benchmark_id,
                job_id=job.job_id
            )
            
            status_value = None
            if hasattr(job_status, 'status'):
                status_value = job_status.status
            elif isinstance(job_status, dict):
                status_value = job_status.get('status')
            
            if status_value:
                last_status = status_value
                print(f"   Status: {status_value}")
            else:
                print(f"   Status: <unknown> ({job_status})")
            
            if status_value in ['completed', 'success', 'succeeded']:
                print(f"✓ Evaluation completed successfully")
                break
            elif status_value in ['failed', 'error']:
                raise Exception(f"Evaluation job failed: {job_status}")
            
            time.sleep(poll_interval)
            waited += poll_interval
            
            if waited >= max_wait_seconds:
                raise TimeoutError(
                    f"Evaluation job {job.job_id} did not complete within {max_wait_seconds} seconds "
                    f"(last status: {last_status or 'unknown'}). "
                    "Increase --max-wait or try a smaller batch."
                )
        except Exception as e:
            # If status check fails, try to retrieve results anyway
            print(f"   Status check note: {e}")
            break
    
    # Retrieve results
    print(f"\n📥 Retrieving evaluation results...")
    
    # Get results using SDK - try different approaches
    try:
        # Try direct result endpoint via custom client
        import httpx
        result_url = f"{client.base_url}/v1alpha/eval/benchmarks/{benchmark_id}/jobs/{job.job_id}/result"
        response = client._client.get(result_url)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print(f"⚠️  Could not get results via SDK client: {e}")
        # Fallback to separate httpx client
        http_client = httpx.Client(verify=False, timeout=30)
        result_url = f"{client.base_url}/v1alpha/eval/benchmarks/{benchmark_id}/jobs/{job.job_id}/result"
        response = http_client.get(result_url)
        response.raise_for_status()
        results = response.json()
        http_client.close()
    
    if not results.get("scores"):
        raise RuntimeError(
            f"Evaluation job {job.job_id} returned no scores "
            f"(last status: {last_status or 'unknown'}). "
            "The job may still be running or failed server-side."
        )
    
    print(f"✓ Results retrieved successfully")
    
    return results


def format_results(results: Dict[str, Any], benchmark_id: str) -> Dict[str, Any]:
    """
    Format evaluation results into a clean dictionary.
    
    Args:
        results: Raw evaluation results from Llama Stack (dict from JSON)
        benchmark_id: Benchmark identifier
        
    Returns:
        Formatted results dictionary
    """
    formatted = {
        "benchmark_id": benchmark_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "individual_scores": {},
        "raw_results": results  # Include raw results for reference
    }
    
    # Extract scores from result (dict from JSON response)
    # The response has 'generations' and 'scores' fields
    if "scores" in results:
        formatted["metrics"] = results["scores"]
    
    if "generations" in results:
        # Store individual generation results
        formatted["generations"] = results["generations"]
    
    return formatted


def print_results_summary(formatted_results: Dict[str, Any], dataset_size: int) -> None:
    """
    Print a human-readable summary of evaluation results.
    
    Args:
        formatted_results: Formatted results dictionary
        dataset_size: Number of entries in the dataset
    """
    print("\n" + "=" * 70)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"Benchmark ID: {formatted_results.get('benchmark_id', 'N/A')}")
    if 'job_id' in formatted_results:
        print(f"Job ID:       {formatted_results['job_id']}")
    print(f"Timestamp:    {formatted_results['timestamp']}")
    print(f"Dataset Size: {dataset_size} entries")
    print("=" * 70)
    
    if formatted_results.get("failures"):
        print("\n❌ Failures:")
        print("-" * 70)
        for failure in formatted_results["failures"]:
            print(f"  Batch {failure.get('batch_index')}: {failure.get('error')}")
        print("-" * 70)

    if formatted_results["metrics"]:
        print("\n🎯 Aggregated Metrics:")
        print("-" * 70)
        for metric, score in formatted_results["metrics"].items():
            # Color code based on score (>0.8=good, 0.6-0.8=ok, <0.6=needs improvement)
            emoji = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
            print(f"  {emoji} {metric:25s}: {score:.4f}")
        print("-" * 70)
    
    if formatted_results.get("individual_scores"):
        print("\n📈 Per-Entry Scores:")
        print("-" * 70)
        for metric, scores in formatted_results["individual_scores"].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                print(f"  {metric}:")
                print(f"    Mean: {avg_score:.4f}  |  Min: {min_score:.4f}  |  Max: {max_score:.4f}")
        print("-" * 70)
    
    print("\n" + "=" * 70)


def evaluate_ragas(
    llama_stack_url: str,
    model_id: str,
    dataset_path: str,
    output_path: str,
    metrics: List[str],
    embedding_model_id: Optional[str] = None,
    verify_ssl: bool = False,
    timeout: int = 600,
    batch_size: Optional[int] = None,
    job_wait_timeout: int = 900,
    poll_interval: int = 5,
) -> Dict[str, Any]:
    """
    Complete RAGAS evaluation workflow using Llama Stack inline provider.
    
    Args:
        llama_stack_url: Base URL of Llama Stack server
        model_id: Model identifier for LLM scoring
        dataset_path: Path to RAGAS dataset JSON file
        output_path: Path to save evaluation results
        metrics: List of RAGAS metrics to compute
        embedding_model_id: Embedding model identifier (optional)
        verify_ssl: Whether to verify SSL certificates
        timeout: Timeout in seconds for requests
        batch_size: Size of batches for evaluation (None = all at once)
        job_wait_timeout: Maximum seconds to wait for remote job completion
        poll_interval: Seconds between job status checks
        
    Returns:
        Formatted evaluation results
    """
    # Initialize client
    print("🔗 Connecting to Llama Stack...")
    print(f"   URL: {llama_stack_url}")
    http_client_instance = httpx.Client(verify=verify_ssl, timeout=timeout)
    client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=http_client_instance
    )
    
    # Verify models are available
    print("\n🔍 Verifying models...")
    try:
        available_models = client.models.list()
        model_ids = [m.identifier for m in available_models if hasattr(m, 'identifier')]
        
        if model_id not in model_ids:
            print(f"⚠️  WARNING: Model '{model_id}' not found in available models")
            print(f"Available models: {', '.join(model_ids[:5])}...")
        else:
            print(f"✓ LLM model '{model_id}' is available")
        
        if embedding_model_id and embedding_model_id not in model_ids:
            print(f"⚠️  WARNING: Embedding model '{embedding_model_id}' not found")
        elif embedding_model_id:
            print(f"✓ Embedding model '{embedding_model_id}' is available")
    except Exception as e:
        print(f"⚠️  Could not verify models: {e}")
    
    # Load dataset
    print(f"\n📖 Loading dataset from: {dataset_path}")
    dataset = load_ragas_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} entries")
    
    # Convert to RAGAS format
    print("\n🔄 Converting to RAGAS format...")
    ragas_data = convert_to_ragas_format(dataset)
    print(f"✓ Converted {len(ragas_data)} entries")
    
    # Batch processing setup
    if batch_size and batch_size > 0:
        batches = [ragas_data[i:i + batch_size] for i in range(0, len(ragas_data), batch_size)]
    else:
        batches = [ragas_data]
    
    print(f"🚀 Starting evaluation in {len(batches)} batch(es)...")
    
    all_results = {
        "metrics": {},
        "individual_scores": {},
        "generations": [],
        "failures": []
    }
    
    # Generate base timestamp
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_scores = {metric: [] for metric in metrics}
    
    for i, batch in enumerate(batches):
        print(f"\n📦 Processing Batch {i+1}/{len(batches)} (Size: {len(batch)})")
        
        # Unique IDs for this batch
        batch_id_suffix = f"_{base_timestamp}_{i+1}"
        dataset_id = f"ragas_dataset{batch_id_suffix}"
        benchmark_id = f"ragas_benchmark{batch_id_suffix}"
        
        try:
            # Register dataset
            register_dataset(client, dataset_id, batch)
            
            # Register benchmark
            register_benchmark(client, benchmark_id, dataset_id, metrics)
            
            # Run evaluation
            batch_result = run_evaluation(
                client=client,
                benchmark_id=benchmark_id,
                ragas_data=batch,
                metrics=metrics,
                model_id=model_id,
                embedding_model_id=embedding_model_id,
                max_wait_seconds=job_wait_timeout,
                poll_interval=poll_interval,
            )
            
            # Process results
            if "scores" in batch_result:
                for metric, score_data in batch_result["scores"].items():
                    # Extract aggregated score for the batch
                    score_val = 0.0
                    if isinstance(score_data, dict) and "aggregated_results" in score_data:
                         score_val = score_data["aggregated_results"].get(metric, 0.0)
                    elif isinstance(score_data, (int, float)):
                         score_val = float(score_data)
                    
                    # Accumulate scores for final aggregation
                    # We store (score, weight) where weight is batch size
                    aggregated_scores[metric].append((score_val, len(batch)))
                    
                    # Extract individual scores from score_rows if available
                    if isinstance(score_data, dict) and "score_rows" in score_data:
                        if metric not in all_results["individual_scores"]:
                            all_results["individual_scores"][metric] = []
                        
                        for row in score_data["score_rows"]:
                             all_results["individual_scores"][metric].append(row.get('score', 0.0))
                    elif len(batch) == 1:
                        # Fallback for single batch if score_rows is missing
                        if metric not in all_results["individual_scores"]:
                            all_results["individual_scores"][metric] = []
                        all_results["individual_scores"][metric].append(score_val)

            if "generations" in batch_result:
                all_results["generations"].extend(batch_result.get("generations", []))
                
        except Exception as e:
            print(f"❌ Batch {i+1} failed: {e}")
            all_results["failures"].append({
                "batch_index": i + 1,
                "error": str(e),
                "data": batch
            })
            # Continue to next batch
            continue

    # Final Aggregation
    print("\n∑ Aggregating results...")
    final_metrics = {}
    for metric, values in aggregated_scores.items():
        if not values:
            continue
            
        total_score = sum(v[0] * v[1] for v in values)
        total_count = sum(v[1] for v in values)
        
        if total_count > 0:
            final_metrics[metric] = total_score / total_count
        else:
            final_metrics[metric] = 0.0

    formatted_results = {
        "benchmark_id": f"ragas_benchmark_batch_run_{base_timestamp}",
        "timestamp": datetime.now().isoformat(),
        "metrics": final_metrics,
        "individual_scores": all_results["individual_scores"],
        "generations": all_results["generations"],
        "failures": all_results["failures"],
        "batch_mode": True if len(batches) > 1 else False
    }
    
    # Save results to file
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved successfully")
    
    # Print summary
    print_results_summary(formatted_results, len(dataset))
    
    return formatted_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems with RAGAS metrics using Llama Stack (inline mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available RAGAS Metrics:
  {chr(10).join(f'  - {m}' for m in AVAILABLE_METRICS)}

Examples:
  # Basic usage with default metrics
  python evaluate_ragas.py --dataset output/millbrook_ragas_dataset.json
  
  # Use specific metrics
  python evaluate_ragas.py \\
    --dataset output/millbrook_ragas_dataset.json \\
    --metrics answer_relevancy faithfulness context_precision
  
  # Sequential evaluation (one by one) to avoid timeout/errors
  python evaluate_ragas.py \\
    --dataset output/millbrook_ragas_dataset.json \\
    --batch-size 1

  # Specify embedding model
  python evaluate_ragas.py \\
    --dataset output/millbrook_ragas_dataset.json \\
    --embedding-model sentence-transformers/nomic-ai/nomic-embed-text-v1.5
  
  # Complete workflow (after running rag.py)
  python evaluate_ragas.py \\
    --url https://llama-stack.example.com \\
    --model-id vllm-inference/llama-4-scout-17b-16e-w4a16 \\
    --dataset output/millbrook_ragas_dataset.json \\
    --output output/millbrook_ragas_results.json

Note: This script uses the INLINE provider mode, which runs RAGAS evaluation
locally within the Llama Stack server (not on a remote RAGAS service).
        """
    )
    
    parser.add_argument(
        "--url",
        default="https://llama-stack-example-llama-stack-example.apps.ocp.sandbox5435.opentlc.com",
        help="Llama Stack server URL"
    )
    parser.add_argument(
        "--model-id",
        default="vllm-inference/llama-4-scout-17b-16e-w4a16",
        help="LLM model identifier for scoring (default: llama-4-scout-17b-16e-w4a16)"
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model identifier (optional, uses server default if not specified)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to RAGAS dataset JSON file (output from rag.py)"
    )
    parser.add_argument(
        "--output",
        help="Path to save evaluation results (default: auto-generated in output/)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=DEFAULT_METRICS,
        help=f"RAGAS metrics to compute (default: {' '.join(DEFAULT_METRICS)})"
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable SSL certificate verification (disabled by default)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for requests (default: 600)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: process all at once). Set to 1 for sequential evaluation."
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=900,
        help="Maximum seconds to wait for the remote evaluation job (default: 900)."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds to wait between job status checks (default: 5)."
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not specified
    if not args.output:
        dataset_path = Path(args.dataset)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(dataset_path.parent / f"{dataset_path.stem}_results_{timestamp}.json")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RAGAS EVALUATION WITH LLAMA STACK (INLINE MODE)")
    print("=" * 70)
    print(f"Server:     {args.url}")
    print(f"Model:      {args.model_id}")
    if args.embedding_model:
        print(f"Embedding:  {args.embedding_model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {args.output}")
    print(f"Metrics:    {', '.join(args.metrics)}")
    if args.batch_size:
        print(f"Batch Size: {args.batch_size}")
    print(f"Max Wait:   {args.max_wait}s  |  Poll Interval: {args.poll_interval}s")
    if not args.verify_ssl:
        print("⚠️  SSL verification: disabled")
    print("=" * 70)
    
    try:
        results = evaluate_ragas(
            llama_stack_url=args.url,
            model_id=args.model_id,
            dataset_path=args.dataset,
            output_path=args.output,
            metrics=args.metrics,
            embedding_model_id=args.embedding_model,
            verify_ssl=args.verify_ssl,
            timeout=args.timeout,
            batch_size=args.batch_size,
            job_wait_timeout=args.max_wait,
            poll_interval=args.poll_interval,
        )
        
        print("\n" + "=" * 70)
        if results.get("failures"):
            print("⚠️ EVALUATION COMPLETED WITH FAILURES")
        else:
            print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📂 Results saved to: {args.output}")
        print("\n💡 Next Steps:")
        print(f"   - Review detailed results: cat {args.output} | jq")
        print(f"   - Compare metrics across runs")
        print(f"   - Identify low-scoring entries for improvement")
        print("=" * 70 + "\n")
        
        return 0 if not results.get("failures") else 1
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}", file=sys.stderr)
        print("\n💡 Make sure you've run rag.py first to generate the dataset.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        print("\n📋 Full traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
