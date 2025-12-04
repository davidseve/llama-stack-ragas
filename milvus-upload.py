#!/usr/bin/env python3
"""
Simple script to upload documents to a Milvus collection using Llama Stack.
Uses the inline Milvus (local) option according to Red Hat OpenShift AI documentation.
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
import httpx


def upload_documents_to_milvus(
    llama_stack_url: str, 
    documents_dir: str = "documents", 
    embedding_model: str = "granite-embedding-125m",
    verify_ssl: bool = False, 
    timeout: int = 300
) -> str:
    """
    Upload documents from a folder to Milvus using Llama Stack.
    
    Args:
        llama_stack_url: Base URL of Llama Stack (e.g., http://localhost:5001)
        documents_dir: Directory containing documents to upload
        embedding_model: Embedding model to use (default: nomic-embed-text-v1.5)
        verify_ssl: Whether to verify SSL certificates (default: False)
        timeout: Timeout in seconds for requests (default: 300)
        
    Returns:
        ID of the collection (vector store) created in Milvus
    """
    # Initialize OpenAI-compatible client with Llama Stack
    http_client = httpx.Client(verify=verify_ssl, timeout=timeout)
    client = OpenAI(
        base_url=f"{llama_stack_url}/v1",
        api_key="fake-api-key",  # Llama Stack does not require a real API key
        http_client=http_client
    )
    
    # Validate embedding model is available
    print(f"🔍 Checking if embedding model '{embedding_model}' is available...")
    try:
        models = client.models.list()
        # Use 'identifier' field for Llama Stack models
        available_models = [m.identifier for m in models.data if hasattr(m, 'identifier')]
        embedding_models = [m for m in models.data if hasattr(m, 'model_type') and m.model_type == 'embedding']
        
        if embedding_model not in available_models:
            print(f"⚠️  WARNING: Embedding model '{embedding_model}' not found in available models")
            print(f"\nAvailable embedding models on this server:")
            for model in embedding_models:
                dim = model.metadata.get('embedding_dimension', 'unknown') if hasattr(model, 'metadata') else 'unknown'
                print(f"  - {model.identifier} (dimension: {dim})")
            
            if embedding_models:
                raise ValueError(f"Embedding model '{embedding_model}' not available. Use one of the models listed above.")
            else:
                print(f"\n⚠️  No embedding models found. The vector store creation may fail.")
        else:
            # Find the model and print its details
            matching_model = next((m for m in embedding_models if m.identifier == embedding_model), None)
            if matching_model and hasattr(matching_model, 'metadata'):
                dim = matching_model.metadata.get('embedding_dimension', 'unknown')
                print(f"✓ Embedding model '{embedding_model}' is available (dimension: {dim})")
            else:
                print(f"✓ Embedding model '{embedding_model}' is available")
    except Exception as e:
        print(f"⚠️  Could not verify embedding model availability: {e}")
        print(f"Proceeding anyway...")
    
    # Verify that the documents directory exists
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory {documents_dir} does not exist")
    
    # Get all files from the directory
    doc_files = list(docs_path.glob("*"))
    if not doc_files:
        raise ValueError(f"No documents found in {documents_dir}")
    
    print(f"📁 Found {len(doc_files)} documents in {documents_dir}")
    
    # Step 1: Upload files to Llama Stack
    print("\n📤 Uploading files...")
    file_ids = []
    for doc_file in doc_files:
        if doc_file.is_file():
            print(f"  - Uploading: {doc_file.name}")
            with open(doc_file, "rb") as f:
                file_obj = client.files.create(
                    file=f,
                    purpose="assistants"
                )
                file_ids.append(file_obj.id)
                print(f"    ✓ File ID: {file_obj.id}")
    
    if not file_ids:
        raise ValueError("Could not upload any file")
    
    print(f"\n✓ {len(file_ids)} files uploaded successfully")
    
    # Step 2: Create vector store (Milvus collection)
    print("\n🗄️  Creating Milvus collection...")
    vector_store_name = f"milvus_collection_{Path.cwd().name}"
    
    # Determine embedding dimension based on model
    # Common dimensions for available models:
    # - granite-embedding-125m: 768 (DEFAULT - matches server config)
    # - sentence-transformers/nomic-ai/nomic-embed-text-v1.5: 768
    # - all-MiniLM-L6-v2: 384 (may not be available on all servers)
    embedding_dimensions = {
        "granite-embedding-125m": 768,
        "sentence-transformers/nomic-ai/nomic-embed-text-v1.5": 768,
        "all-MiniLM-L6-v2": 384,
    }
    
    embedding_dimension = embedding_dimensions.get(embedding_model, 768)
    
    print(f"Using embedding model: {embedding_model}")
    print(f"Embedding dimension: {embedding_dimension}")
    
    vector_store = client.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "provider_id": "inline-milvus"  # Inline Milvus (local)
        }
    )
    
    vector_store_id = vector_store.id
    print(f"✓ Collection created: {vector_store_name}")
    print(f"✓ Vector Store ID: {vector_store_id}")
    print(f"✓ Embedding Model: {embedding_model}")
    
    # Step 3: Associate files to vector store
    print("\n🔗 Associating files to collection...")
    vector_store_files = client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id,
        file_ids=file_ids
    )
    
    print(f"✓ {len(file_ids)} files associated successfully")
    print(f"\n{'='*60}")
    print(f"✅ PROCESS COMPLETED")
    print(f"{'='*60}")
    print(f"Vector Store ID: {vector_store_id}")
    print(f"{'='*60}\n")
    
    return vector_store_id


def main():
    parser = argparse.ArgumentParser(
        description="Upload documents to Milvus using Llama Stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available embedding models:
  - granite-embedding-125m (768 dim, DEFAULT - matches server config)
  - sentence-transformers/nomic-ai/nomic-embed-text-v1.5 (768 dim)
  - all-MiniLM-L6-v2 (384 dim, may not be available on all servers)

⚠️ IMPORTANTE: El modelo de embedding DEBE coincidir con la configuración del servidor.
               El servidor está configurado con: granite-embedding-125m

Example:
  python milvus-upload.py --embedding-model granite-embedding-125m
        """
    )
    parser.add_argument(
        "--url",
        default="https://llama-stack-example-llama-stack-example.apps.ocp.sandbox5435.opentlc.com",
        help="Llama Stack URL (default: OpenShift sandbox)"
    )
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Directory containing documents (default: documents)"
    )
    parser.add_argument(
        "--embedding-model",
        default="granite-embedding-125m",
        help="Embedding model to use (default: granite-embedding-125m, must match server config)"
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
    
    print(f"🔗 Connecting to: {args.url}")
    if not args.verify_ssl:
        print("⚠️  SSL verification disabled (default)")
    
    try:
        vector_store_id = upload_documents_to_milvus(
            llama_stack_url=args.url,
            documents_dir=args.documents_dir,
            embedding_model=args.embedding_model,
            verify_ssl=args.verify_ssl,
            timeout=args.timeout
        )
        print(f"\n{'='*60}")
        print(f"📝 Next steps:")
        print(f"{'='*60}")
        print(f"Use this Vector Store ID in rag.py:")
        print(f"\nexport VECTOR_STORE_ID={vector_store_id}")
        print(f"python rag.py")
        print(f"{'='*60}\n")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        print("\n📋 Full traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

