#!/usr/bin/env python3
"""
RAG Dataset Generator for RAGAS Evaluation

This script processes questions from a dataset, queries a RAG system using Llama Stack,
and generates an enriched dataset with answers and contexts for RAGAS evaluation.

Uses the Responses API with file_search tool.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import httpx
from llama_stack_client import LlamaStackClient


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load questions dataset from JSON file.
    
    Args:
        dataset_path: Path to the input JSON dataset
        
    Returns:
        List of question dictionaries
    """
    with open(dataset_path, 'r') as f:
        return json.load(f)


def query_rag_with_responses_api(
    client: LlamaStackClient,
    model_id: str,
    vector_store_id: str,
    question: str
) -> Dict[str, Any]:
    """
    Query RAG system using Responses API.
    
    Red Hat documentation format:
    response = client.responses.create(
        model=model_id,
        input=query,
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
    )
    
    Args:
        client: LlamaStackClient instance
        model_id: Model identifier to use
        vector_store_id: Vector store ID containing documents
        question: Question to ask
        
    Returns:
        Dictionary with 'answer' and 'contexts' fields
    """
    response = client.responses.create(
        model=model_id,
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
        include=["file_search_call.results"],
    )
    
    # Extract answer from the last message in output
    answer = getattr(response, "output_text", str(response))
    
    # Extract contexts from response
    contexts = []
    
    # The Responses API returns output as a list
    # First item is the tool call with results, second item is the final message
    if hasattr(response, 'output') and isinstance(response.output, list):
        for item in response.output:
            # Check for FileSearchToolCall which contains the results
            if hasattr(item, 'results') and isinstance(item.results, list):
                # Extract text from each result
                for result in item.results:
                    if hasattr(result, 'text') and result.text:
                        contexts.append(result.text)
    
    return {
        "answer": answer,
        "contexts": contexts if contexts else ["No context retrieved"]
    }


def generate_ragas_dataset(
    llama_stack_url: str,
    model_id: str,
    vector_store_id: str,
    input_dataset_path: str,
    output_dataset_path: str,
    verify_ssl: bool = False,
    timeout: int = 300
) -> None:
    """
    Generate RAGAS-compatible dataset with RAG answers and contexts using Responses API.
    
    Args:
        llama_stack_url: Base URL of Llama Stack server
        model_id: Model identifier to use for inference
        vector_store_id: Vector store ID containing documents
        input_dataset_path: Path to input JSON dataset
        output_dataset_path: Path to save output dataset
        verify_ssl: Whether to verify SSL certificates
        timeout: Timeout in seconds for requests
    """
    # Initialize client
    http_client = httpx.Client(verify=verify_ssl, timeout=timeout)
    client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=http_client
    )
    
    # Load input dataset
    print(f"📖 Loading dataset from: {input_dataset_path}")
    dataset = load_dataset(input_dataset_path)
    print(f"✓ Loaded {len(dataset)} questions")
    
    # Process each question
    print(f"\n🤖 Processing questions using Responses API...")
    print(f"Model: {model_id}")
    print(f"Vector Store: {vector_store_id}\n")
    
    ragas_dataset = []
    
    for i, item in enumerate(dataset, 1):
        question_id = item.get('id', f'q_{i}')
        question = item['question']
        ground_truth = item.get('ground_truth', '')
        
        print(f"[{i}/{len(dataset)}] Processing: {question_id}")
        
        try:
            # Query RAG system using Responses API
            result = query_rag_with_responses_api(client, model_id, vector_store_id, question)
            
            # Build RAGAS entry
            ragas_entry = {
                "id": question_id,
                "question": question,
                "answer": result['answer'],
                "contexts": result['contexts'],
                "ground_truth": ground_truth,
            }
            
            # Include optional fields from original dataset
            if 'difficulty' in item:
                ragas_entry['difficulty'] = item['difficulty']
            
            ragas_dataset.append(ragas_entry)
            print(f"  ✓ Answer generated ({len(result['contexts'])} contexts retrieved)")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {question_id}: {e}")
            # Add entry with error information
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "difficulty": item.get('difficulty', 'unknown')
            })
    
    # Save output dataset
    print(f"\n💾 Saving RAGAS dataset to: {output_dataset_path}")
    with open(output_dataset_path, 'w') as f:
        json.dump(ragas_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset generation complete!")
    print(f"   Input: {len(dataset)} questions")
    print(f"   Output: {len(ragas_dataset)} entries")
    print(f"\n📊 Ready for RAGAS evaluation!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAGAS dataset with RAG answers and contexts using Llama Stack Responses API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rag.py --vector-store-id vs_627e6e71-8a1b-45cc-bea5-d7689e71e27b
  
  # Custom input/output files
  python rag.py \\
    --vector-store-id vs_627e6e71-8a1b-45cc-bea5-d7689e71e27b \\
    --input dataset-base/my_dataset.json \\
    --output output/my_ragas_dataset.json
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
        help="Model identifier to use (default: vllm-inference/llama-4-scout-17b-16e-w4a16)"
    )
    parser.add_argument(
        "--vector-store-id",
        required=True,
        help="Vector store ID containing ingested documents"
    )
    parser.add_argument(
        "--input",
        default="dataset-base/millbrook_dataset.json",
        help="Input dataset JSON file (default: dataset-base/millbrook_dataset.json)"
    )
    parser.add_argument(
        "--output",
        default="output/millbrook_ragas_dataset.json",
        help="Output dataset JSON file (default: output/millbrook_ragas_dataset.json)"
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable SSL certificate verification (disabled by default)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for requests (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RAG Dataset Generator for RAGAS Evaluation")
    print("=" * 70)
    print(f"Server: {args.url}")
    print(f"Model: {args.model_id}")
    print(f"Vector Store: {args.vector_store_id}")
    if not args.verify_ssl:
        print("⚠️  SSL verification: disabled")
    print("=" * 70)
    
    try:
        generate_ragas_dataset(
            llama_stack_url=args.url,
            model_id=args.model_id,
            vector_store_id=args.vector_store_id,
            input_dataset_path=args.input,
            output_dataset_path=args.output,
            verify_ssl=args.verify_ssl,
            timeout=args.timeout
        )
        return 0
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

