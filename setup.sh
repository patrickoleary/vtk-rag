#!/bin/bash
# VTK RAG - Setup Script
# Creates virtual environment and installs dependencies
#
# Usage:
#   ./setup.sh          # Interactive mode (asks for each component)
#   ./setup.sh --all    # Install everything without prompts

set -e  # Exit on error

# Check for --all flag
INSTALL_ALL=false
if [[ "$1" == "--all" ]]; then
    INSTALL_ALL=true
fi

echo "========================================="
echo "VTK RAG Setup"
echo "========================================="
echo ""
if [ "$INSTALL_ALL" = true ]; then
    echo "Mode: Install ALL dependencies"
else
    echo "Mode: Interactive (selective installation)"
fi
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Use it or delete it first."
    echo "To use existing: source .venv/bin/activate"
    echo "To delete: rm -rf .venv"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic requirements (Python standard library only for core functionality)
echo ""
echo "Checking requirements..."
echo "✓ Corpus preparation uses only Python standard library - no dependencies needed!"

# Optional: Install corpus preparation dependencies (visualizations)
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for visualizations? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing visualization dependencies..."
    pip install -r prepare-corpus/requirements.txt
    echo "✓ Visualization dependencies installed"
else
    echo "Skipping visualization dependencies. Analysis will work with text output only."
fi

# Optional: Install indexing dependencies
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for building hybrid index (Qdrant)? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing indexing dependencies..."
    pip install -r build-indexes/requirements.txt
    echo "✓ Indexing dependencies installed"
    echo "Note: You'll need to start Qdrant with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
else
    echo "Skipping indexing dependencies. Install later with: pip install -r build-indexes/requirements.txt"
fi

# Optional: Install RAG pipeline dependencies
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for complete RAG pipeline (retrieval, grounding, LLM, post-processing)? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing RAG pipeline dependencies..."
    pip install -r retrieval-pipeline/requirements.txt
    pip install -r grounding-prompting/requirements.txt
    pip install -r llm-generation/requirements.txt
    pip install -r post-processing/requirements.txt
    echo "✓ RAG pipeline dependencies installed"
    echo "Note: Configure LLM provider in .env file (copy from .env.example)"
else
    echo "Skipping RAG pipeline dependencies."
    echo "Install later with: pip install -r {retrieval-pipeline,grounding-prompting,llm-generation,post-processing}/requirements.txt"
fi

# Optional: Install API validation dependencies
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for API validation (hallucination detection via MCP)? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing API validation dependencies..."
    pip install -r api-mcp/requirements.txt
    echo "✓ API validation dependencies installed"
else
    echo "Skipping API validation. Install later with: pip install -r api-mcp/requirements.txt"
fi

# Optional: Install evaluation dependencies
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for evaluation (requires VTK)? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing evaluation dependencies..."
    pip install -r evaluation/requirements.txt
    echo "✓ Evaluation dependencies installed"
else
    echo "Skipping evaluation. Install later with: pip install -r evaluation/requirements.txt"
fi

# Optional: Install visual testing dependencies
echo ""
if [ "$INSTALL_ALL" = true ]; then
    REPLY="y"
else
    read -p "Install dependencies for visual testing (VTK + image processing)? (y/n) " -n 1 -r
    echo ""
fi
if [[ $REPLY =~ ^[Yy]$ ]] || [ "$INSTALL_ALL" = true ]; then
    echo "Installing visual testing dependencies..."
    pip install -r visual_testing/requirements.txt
    echo "✓ Visual testing dependencies installed"
    echo "Note: Docker is required for sandboxed execution. Use query.py --visual-test"
else
    echo "Skipping visual testing. Install later with: pip install -r visual_testing/requirements.txt"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Configure LLM: cp .env.example .env (then edit with your API key)"
echo "  2. Place raw data in data/raw/ (see data/raw/README.md)"
echo "  3. Start Qdrant: docker run -d -p 6333:6333 qdrant/qdrant"
echo "  4. Build index: python build.py"
echo "  5. Query system: python query.py 'How do I create a cylinder?'"
echo ""
echo "Quick install all dependencies (non-interactive):"
echo "  ./setup.sh --all"
echo ""
echo "Or install specific components later:"
echo "  pip install -r <component>/requirements.txt"
echo ""
echo "See README.md for full documentation."
echo ""
