#!/bin/bash
# VTK RAG - Setup Script
# Creates virtual environment and installs dependencies using uv
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

# Check for uv
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv found: $(uv --version)"
fi
echo ""

# Create virtual environment if it doesn't exist
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    uv venv .venv
    echo "✓ Virtual environment created"
fi
echo ""

# Install package
if [ "$INSTALL_DEV" = true ]; then
    echo "Installing vtk-rag with dev dependencies..."
    uv pip install -e ".[dev]"
    echo "✓ Production + dev dependencies installed"
else
    echo "Installing vtk-rag..."
    uv pip install -e .
    echo "✓ Production dependencies installed"
fi

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
    fi
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Usage (no activation needed with uv):"
echo "  uv run vtk-rag build              # Build the RAG index"
echo "  uv run vtk-rag search 'sphere'    # Search for code"
echo ""
echo "Or activate the virtual environment:"
echo "  source .venv/bin/activate"
echo "  vtk-rag build"
echo ""
echo "Next steps:"
echo "  1. Place raw data in data/raw/"
echo "  2. Start Qdrant: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant"
echo "  3. Build: uv run vtk-rag build"
echo "  4. Search: uv run vtk-rag search 'create a sphere'"
echo ""
if [ "$INSTALL_DEV" = true ]; then
    echo "Development commands:"
    echo "  uv run pytest tests/              # Run tests"
    echo "  uv run ruff check vtk_rag/ tests/ # Lint code"
    echo ""
fi
echo "See README.md for full documentation."
echo ""
