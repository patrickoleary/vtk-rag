#!/bin/bash
# VTK RAG - Setup Script
# Creates virtual environment and installs dependencies
#
# Usage:
#   ./setup.sh          # Install production dependencies
#   ./setup.sh --dev    # Install production + development dependencies

set -e  # Exit on error

echo "========================================="
echo "VTK RAG Setup"
echo "========================================="
echo ""

# Check for --dev flag
INSTALL_DEV=false
if [[ "$1" == "--dev" ]]; then
    INSTALL_DEV=true
    echo "Mode: Development (includes pytest, ruff)"
else
    echo "Mode: Production"
fi
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.10+ is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
    echo "Activating existing environment..."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install package
echo ""
if [ "$INSTALL_DEV" = true ]; then
    echo "Installing vtk-rag with dev dependencies..."
    pip install -e ".[dev]"
    echo "✓ Production + dev dependencies installed"
else
    echo "Installing vtk-rag..."
    pip install -e .
    echo "✓ Production dependencies installed"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Place raw data in data/raw/"
echo "  2. Start Qdrant: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant"
echo "  3. Build: vtk-rag build"
echo "  4. Search: vtk-rag search 'create a sphere'"
echo ""
if [ "$INSTALL_DEV" = true ]; then
    echo "Development commands:"
    echo "  pytest tests/              # Run tests"
    echo "  ruff check vtk_rag/ tests/ # Lint code"
    echo ""
fi
echo "See README.md for full documentation."
echo ""
