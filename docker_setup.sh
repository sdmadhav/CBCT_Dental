# ============================================================================
# Dockerfile
# ============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p src configs data/raw data/processed data/cache \
    saved_models logs results

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["python", "main.py", "--mode", "train", "--config", "configs/config.yaml"]


# ============================================================================
# docker-compose.yml
# ============================================================================

# version: '3.8'
# 
# services:
#   cbct-pipeline:
#     build: .
#     container_name: cbct-pipeline
#     runtime: nvidia
#     volumes:
#       - ./data:/app/data
#       - ./saved_models:/app/saved_models
#       - ./logs:/app/logs
#       - ./results:/app/results
#     environment:
#       - NVIDIA_VISIBLE_DEVICES=all
#     command: python main.py --mode train --config configs/config.yaml
# 
#   tensorboard:
#     image: tensorflow/tensorflow:latest
#     container_name: cbct-tensorboard
#     ports:
#       - "6006:6006"
#     volumes:
#       - ./logs:/logs
#     command: tensorboard --logdir /logs --host 0.0.0.0


# ============================================================================
# setup.sh - Complete Setup Script
# ============================================================================

#!/bin/bash

# CBCT Pipeline Setup Script
# This script sets up the complete project structure and environment

set -e  # Exit on error

echo "=================================="
echo "CBCT Pipeline Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python version $python_version is compatible"
else
    print_error "Python version $python_version is not compatible. Required: 3.8+"
    exit 1
fi

# Create directory structure
print_info "Creating directory structure..."
mkdir -p src
mkdir -p configs
mkdir -p data/{raw,processed,cache,splits}
mkdir -p saved_models
mkdir -p logs
mkdir -p results
mkdir -p tests

print_status "Directory structure created"

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_info "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Image processing
opencv-python>=4.5.0
scikit-image>=0.18.0
pydicom>=2.3.0
SimpleITK>=2.1.0
Pillow>=8.3.0

# Deep learning
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.10.0

# Machine learning
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Configuration
pyyaml>=5.4.0

# Progress bars
tqdm>=4.62.0
EOF
    print_status "requirements.txt created"
fi

# Install dependencies
print_info "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Create __init__.py files
print_info "Creating __init__.py files..."
touch src/__init__.py
touch tests/__init__.py
print_status "__init__.py files created"

# Create default config if it doesn't exist
if [ ! -f "configs/config.yaml" ]; then
    print_info "Creating default configuration..."
    cat > configs/config.yaml << 'EOF'
# CBCT Dental Image Analysis Pipeline Configuration

project:
  name: "CBCT_Dental_Analysis"
  version: "1.0.0"

paths:
  data_dir: "./data/raw"
  cache_dir: "./data/processed/cache"
  processed_dir: "./data/processed"
  checkpoint_dir: "./saved_models"
  log_dir: "./logs"
  results_dir: "./results"

data:
  expected_shape: [401, 401, 251]
  expected_dtype: "int16"
  target_size_2d: [256, 256]
  target_size_3d: [128, 128, 128]
  split_ratios: [0.7, 0.15, 0.15]
  use_cache: true
  num_workers: 4

preprocessing:
  normalization:
    method: "min_max"
    clip_percentile: [1, 99]
  denoising:
    method: "gaussian"
    sigma: 1.0
  contrast:
    method: "clahe"
    clip_limit: 2.0
    tile_grid_size: [8, 8]

model:
  name: "resnet18_2d"
  num_classes: 32
  pretrained: true

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  loss: "cross_entropy"
  mixed_precision: true
  early_stopping_patience: 10

hardware:
  use_gpu: true
  
seed: 42
EOF
    print_status "Default configuration created"
fi

# Create .gitignore
if [ ! -f ".gitignore" ]; then
    print_info "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Data
data/raw/*
data/processed/*
data/cache/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/cache/.gitkeep

# Models and logs
saved_models/*.pth
logs/*
results/*
!saved_models/.gitkeep
!logs/.gitkeep
!results/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/

# TensorBoard
events.out.tfevents.*
EOF
    print_status ".gitignore created"
fi

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/cache/.gitkeep
touch saved_models/.gitkeep
touch logs/.gitkeep
touch results/.gitkeep

# Create a simple test script
if [ ! -f "tests/test_installation.py" ]; then
    print_info "Creating test script..."
    cat > tests/test_installation.py << 'EOF'
"""
Test script to verify installation
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import pydicom
        print("✓ PyDICOM")
    except ImportError as e:
        print(f"✗ PyDICOM: {e}")
        return False
    
    try:
        import SimpleITK
        print("✓ SimpleITK")
    except ImportError as e:
        print(f"✗ SimpleITK: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
EOF
    print_status "Test script created"
fi

# Run tests
print_info "Running installation tests..."
python tests/test_installation.py

if [ $? -eq 0 ]; then
    print_status "Installation tests passed"
else
    print_error "Installation tests failed"
    exit 1
fi

# Create quick start script
print_info "Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash

# Quick start script for CBCT pipeline

# Activate virtual environment
source venv/bin/activate

# Check if data exists
if [ ! "$(ls -A data/raw)" ]; then
    echo "⚠️  Warning: No data found in data/raw/"
    echo "Please add your DICOM files to data/raw/ directory"
    echo "Expected structure:"
    echo "  data/raw/"
    echo "    patient_001/"
    echo "      slice_001.dcm"
    echo "      slice_002.dcm"
    exit 1
fi

# Run based on argument
case "$1" in
    train)
        echo "Starting training..."
        python main.py --mode train --config configs/config.yaml
        ;;
    evaluate)
        echo "Starting evaluation..."
        python main.py --mode evaluate --config configs/config.yaml
        ;;
    tensorboard)
        echo "Starting TensorBoard..."
        tensorboard --logdir logs/
        ;;
    *)
        echo "Usage: ./quick_start.sh {train|evaluate|tensorboard}"
        exit 1
        ;;
esac
EOF

chmod +x quick_start.sh
print_status "Quick start script created"

# Create README for data directory
cat > data/README.md << 'EOF'
# Data Directory Structure

Place your DICOM files in the following structure:

```
data/
├── raw/                    # Original DICOM files
│   ├── patient_001/
│   │   ├── slice_001.dcm
│   │   ├── slice_002.dcm
│   │   └── ...
│   ├── patient_002/
│   └── ...
│
├── processed/              # Preprocessed data
│   └── (auto-generated)
│
└── cache/                  # Cached volumes
    └── (auto-generated)
```

## Notes

- Each patient should have their own directory
- DICOM files should be in `.dcm` format
- The pipeline will automatically cache processed volumes for faster loading
- Preprocessed data will be stored in the `processed/` directory
EOF

print_status "Data README created"

# Final summary
echo ""
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Add your DICOM files to: data/raw/"
echo "2. Review configuration: configs/config.yaml"
echo "3. Run training: ./quick_start.sh train"
echo "4. Monitor with TensorBoard: ./quick_start.sh tensorboard"
echo ""
echo "For more information, see README.md"
echo ""

# Deactivate virtual environment
deactivate


# ============================================================================
# Usage Instructions
# ============================================================================

# To use this setup script:
# 
# 1. Save it as setup.sh
# 2. Make it executable: chmod +x setup.sh
# 3. Run it: ./setup.sh
# 
# The script will:
# - Check Python version
# - Create directory structure
# - Set up virtual environment
# - Install all dependencies
# - Create configuration files
# - Run installation tests
# - Create helper scripts
