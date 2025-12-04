#!/bin/bash
# Setup virtual environment with llama-stack-client 0.2.17 (matching server version)

echo "==================================="
echo "Setting up Virtual Environment"
echo "Target: llama-stack-client 0.3.3 (compatible with server 0.3.0rc3)"
echo "==================================="

# Remove old venv if exists
rm -rf .venv

# Create new venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "==================================="
echo "Installed versions:"
echo "==================================="
pip list | grep -E "(llama-stack|httpx)"

echo ""
echo "==================================="
echo "Virtual environment ready!"
echo "==================================="
echo ""
echo "Server version: Llama Stack 0.3.0rc3+rhai0"
echo "Client version: llama-stack-client 0.3.3"
echo ""
echo "To activate:"
echo "  source .venv/bin/activate"
echo ""
echo "To test:"
echo "  python rag.py --vector-store-id vs_627e6e71-8a1b-45cc-bea5-d7689e71e27b"
echo ""

