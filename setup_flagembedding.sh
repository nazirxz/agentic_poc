#!/bin/bash
# Setup script for FlagEmbedding with disk quota workaround

set -e

echo "=== FlagEmbedding Setup Script ==="
echo ""

# Check if FlagEmbedding is already installed
if python -c "import FlagEmbedding" 2>/dev/null; then
    echo "✓ FlagEmbedding is already installed"
    python -c "from FlagEmbedding import BGEM3FlagModel; print('✓ BGEM3FlagModel available')"
    python -c "from FlagEmbedding import FlagReranker; print('✓ FlagReranker available')"
    echo ""
    echo "No action needed!"
    exit 0
fi

echo "FlagEmbedding not found. Installing..."
echo ""

# Option 1: Try clearing pip cache first
echo "Option 1: Clearing pip cache..."
pip cache purge 2>/dev/null || echo "  (Could not purge cache, continuing...)"

# Option 2: Install with no cache
echo ""
echo "Option 2: Installing FlagEmbedding with --no-cache-dir..."
pip install --no-cache-dir FlagEmbedding || {
    echo ""
    echo "❌ Installation failed due to disk quota"
    echo ""
    echo "=== Manual Solutions ==="
    echo ""
    echo "1. Clean up disk space:"
    echo "   du -sh ~/.cache/pip"
    echo "   rm -rf ~/.cache/pip/*"
    echo ""
    echo "2. Install in a different location:"
    echo "   pip install --target /tmp/flagembedding FlagEmbedding"
    echo "   export PYTHONPATH=/tmp/flagembedding:\$PYTHONPATH"
    echo ""
    echo "3. Use system Python (if available):"
    echo "   sudo pip install FlagEmbedding"
    echo ""
    echo "4. Build from source in /tmp:"
    echo "   cd /tmp"
    echo "   git clone https://github.com/FlagOpen/FlagEmbedding.git"
    echo "   cd FlagEmbedding"
    echo "   pip install -e . --no-cache-dir"
    echo ""
    echo "5. Use Ollama fallback (already implemented):"
    echo "   Set USE_BGE_M3_HYBRID=False in settings.py"
    echo "   The system will fallback to Ollama embeddings"
    echo ""
    exit 1
}

echo ""
echo "✓ FlagEmbedding installed successfully!"
echo ""
echo "Verifying installation..."
python -c "from FlagEmbedding import BGEM3FlagModel; print('✓ BGEM3FlagModel available')"
python -c "from FlagEmbedding import FlagReranker; print('✓ FlagReranker available')"

echo ""
echo "=== Setup Complete ==="

