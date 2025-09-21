#!/bin/bash

# ACE-Step Music Generator Setup Script

echo "🎵 Setting up ACE-Step Music Generator..."

# Check if Python 3.10+ is available
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.10+ required, but found $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ace_step_env
source ace_step_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Clone ACE-Step repository
echo "📥 Cloning ACE-Step repository..."
if [ ! -d "ACE-Step" ]; then
    git clone https://github.com/ace-step/ACE-Step.git
fi

cd ACE-Step

# Install PyTorch (adjust CUDA version as needed)
echo "🔥 Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install ACE-Step
echo "🎼 Installing ACE-Step..."
pip install -e .

# Install additional API dependencies
echo "🌐 Installing API dependencies..."
pip install fastapi uvicorn loguru

# Create directories
echo "📁 Creating output directories..."
mkdir -p generated_music
mkdir -p checkpoints

# Copy our simple API
echo "📋 Setting up simple API..."
cp ../simple_ace_api.py ./

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use the API:"
echo "1. Activate environment: source ace_step_env/bin/activate"
echo "2. Run API: python simple_ace_api.py"
echo "3. Visit: http://localhost:8000"
echo ""
echo "Or use the existing trainer-api.py:"
echo "python trainer-api.py"
echo ""
echo "Note: Models will auto-download on first use (~3.5GB)"
