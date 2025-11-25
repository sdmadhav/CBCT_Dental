```

---

# README.md

# CBCT Dental Image Analysis Pipeline

A comprehensive, production-ready deep learning pipeline for analyzing CBCT (Cone Beam Computed Tomography) dental images. This system provides end-to-end functionality from DICOM data loading to model training, evaluation, and inference.

## 🚀 Features

- **Complete Pipeline**: Data loading, preprocessing, training, evaluation, and inference
- **Multiple Architectures**: 2D CNN (ResNet), U-Net for segmentation, lightweight 3D CNN
- **Advanced Preprocessing**: Noise reduction, normalization, contrast enhancement, windowing
- **Data Augmentation**: Rotation, flip, zoom, elastic deformation
- **Production Ready**: Comprehensive error handling, logging, checkpointing
- **Easy Configuration**: YAML-based configuration system
- **GPU Optimized**: Mixed precision training, efficient memory usage
- **Extensive Metrics**: Accuracy, precision, recall, F1, IoU, confusion matrices, ROC curves

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Customization](#customization)
- [Model Architectures](#model-architectures)
- [Troubleshooting](#troubleshooting)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU is supported)
- 16GB RAM minimum (32GB recommended for 3D processing)

### Step 1: Clone or Create Project Structure

```bash
# Create project directory
mkdir cbct_pipeline
cd cbct_pipeline

# Create directory structure
mkdir -p src configs data/{raw,processed,cache} saved_models logs results
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pydicom; import SimpleITK; print('DICOM libraries OK')"
```

## 🚦 Quick Start

### 1. Prepare Your Data

Place your DICOM files in the data directory:

```
data/raw/
├── patient_001/
│   ├── slice_001.dcm
│   ├── slice_002.dcm
│   └── ...
├── patient_002/
└── ...
```

### 2. Configure the Pipeline

Edit `configs/config.yaml` to match your dataset and requirements:

```yaml
data:
  expected_shape: [401, 401, 251]  # Your volume dimensions
  target_size_2d: [256, 256]       # Target size for 2D slices

model:
  name: "resnet18_2d"               # Choose your model
  num_classes: 32                   # Number of teeth/classes

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
```

### 3. Train the Model

```bash
python main.py --mode train --config configs/config.yaml
```

### 4. Evaluate

```bash
python main.py --mode evaluate --config configs/config.yaml --checkpoint saved_models/best_model.pth
```

### 5. Run Inference

```bash
python main.py --mode inference --input data/raw/patient_new.dcm --output results/prediction.json
```

## 📁 Project Structure

```
cbct_pipeline/
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # DICOM loading & caching
│   ├── preprocessing.py         # Image preprocessing pipeline
│   ├── feature_extraction.py   # Feature extraction methods
│   ├── dataset.py               # PyTorch Dataset classes
│   ├── models.py                # Neural network architectures
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Evaluation & metrics
│   ├── inference.py             # Production inference
│   └── utils.py                 # Utility functions
│
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model_configs.yaml       # Model-specific configs (optional)
│   └── preprocessing_configs.yaml # Preprocessing presets (optional)
│
├── data/                         # Data directory
│   ├── raw/                     # Raw DICOM files
│   ├── processed/               # Preprocessed data
│   └── cache/                   # Cached volumes
│
├── saved_models/                # Trained model checkpoints
├── logs/                        # Training logs & TensorBoard
├── results/                     # Evaluation results & predictions
│
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Dockerfile                   # Docker configuration (optional)
└── .gitignore

```

## ⚙️ Configuration

The pipeline is controlled via `configs/config.yaml`. Key sections:

### Data Configuration

```yaml
data:
  expected_shape: [401, 401, 251]  # Z, Y, X dimensions
  target_size_2d: [256, 256]       # Resize target for 2D
  target_size_3d: [128, 128, 128]  # Resize target for 3D
  split_ratios: [0.7, 0.15, 0.15]  # train/val/test
  use_cache: true                   # Cache processed volumes
```

### Preprocessing Options

```yaml
preprocessing:
  normalization:
    method: "min_max"              # min_max, z_score, window
    clip_percentile: [1, 99]       # Outlier clipping
  
  denoising:
    method: "gaussian"              # gaussian, bilateral, median
    sigma: 1.0
  
  contrast:
    method: "clahe"                 # clahe, histogram_eq
    clip_limit: 2.0
```

### Model Selection

```yaml
model:
  name: "resnet18_2d"   # Options:
                        # - resnet18_2d: 2D ResNet-18 for classification
                        # - resnet50_2d: 2D ResNet-50 for classification
                        # - unet_2d: U-Net for segmentation
                        # - cnn_3d: 3D CNN for volumetric analysis
  num_classes: 32
  pretrained: true      # Use ImageNet pretrained weights
```

### Training Parameters

```yaml
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"                # adam, sgd
  loss: "cross_entropy"            # cross_entropy, dice, focal
  mixed_precision: true            # Use AMP for faster training
  early_stopping_patience: 10
```

## 💡 Usage Examples

### Example 1: Train a 2D Classifier

```bash
# 1. Update config for 2D classification
# configs/config.yaml:
#   model:
#     name: "resnet18_2d"
#     num_classes: 32

# 2. Train
python main.py --mode train --config configs/config.yaml

# 3. Monitor with TensorBoard
tensorboard --logdir logs/
```

### Example 2: Train U-Net for Segmentation

```bash
# 1. Update config for segmentation
# configs/config.yaml:
#   model:
#     name: "unet_2d"
#     num_classes: 2  # background + teeth
#   training:
#     loss: "dice"    # Dice loss for segmentation

# 2. Train
python main.py --mode train --config configs/config.yaml
```

### Example 3: Resume Training

```bash
# Resume from a checkpoint
python main.py --mode train --config configs/config.yaml --resume saved_models/latest_checkpoint.pth
```

### Example 4: Batch Inference

```python
from src.inference import InferenceEngine
from src.data_loader import DICOMLoader
from src.preprocessing import CBCTPreprocessor

# Setup
loader = DICOMLoader("./data/raw")
preprocessor = CBCTPreprocessor()

# Load model
engine = InferenceEngine(
    model=your_model,
    checkpoint_path="saved_models/best_model.pth",
    device=torch.device('cuda'),
    preprocessor=preprocessor
)

# Load images
images = []
for patient_dir in Path("./data/test").iterdir():
    result = loader.load_dicom_series(patient_dir)
    images.append(result['volume'][100])  # Get middle slice

# Batch prediction
results = engine.predict_batch(images)

# Export
engine.export_results(results, "results/batch_predictions.json")
```

## 🎨 Customization

### Adding a New Preprocessing Step

Edit `src/preprocessing.py`:

```python
def your_custom_preprocessing(self, image: np.ndarray) -> np.ndarray:
    """Your custom preprocessing step."""
    # Your code here
    processed = custom_operation(image)
    return processed

# Add to pipeline
def preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
    processed = self.normalize(image)
    processed = self.denoise(processed)
    processed = self.your_custom_preprocessing(processed)  # Add here
    return processed
```

### Adding a New Model Architecture

Edit `src/models.py`:

```python
class YourCustomModel(nn.Module):
    def __init__(self, num_classes: int = 32):
        super().__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        # Your forward pass
        return self.fc(x)

# Register in get_model function
def get_model(model_name: str, **kwargs):
    models = {
        'your_model': YourCustomModel,
        # ... existing models
    }
    return models[model_name](**kwargs)
```

### Custom Data Augmentation

Edit `src/preprocessing.py`:

```python
class DataAugmenter:
    def custom_augmentation(self, image: np.ndarray) -> np.ndarray:
        # Your augmentation
        return augmented_image
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        augmented = image.copy()
        
        # Add your custom augmentation
        if self.config.get('custom', {}).get('enabled'):
            augmented = self.custom_augmentation(augmented)
        
        return augmented
```

## 🏗️ Model Architectures

### 1. ResNet-based 2D Classifier

- **Use Case**: Tooth classification from 2D slices
- **Input**: Single-channel 2D slices (256×256)
- **Output**: Class probabilities for N classes
- **Pretrained**: ImageNet weights available
- **Parameters**: ~11M (ResNet-18), ~23M (ResNet-50)

```yaml
model:
  name: "resnet18_2d"
  num_classes: 32
  pretrained: true
```

### 2. U-Net 2D Segmentation

- **Use Case**: Pixel-wise tooth segmentation
- **Input**: Single-channel 2D slices (256×256)
- **Output**: Segmentation mask (H×W×num_classes)
- **Architecture**: Encoder-decoder with skip connections
- **Parameters**: ~31M (base_channels=64)

```yaml
model:
  name: "unet_2d"
  num_classes: 2  # background + teeth
  unet:
    base_channels: 64
```

### 3. 3D CNN

- **Use Case**: Volumetric analysis
- **Input**: 3D volumes (128×128×128)
- **Output**: Classification or volumetric segmentation
- **Architecture**: Lightweight 3D convolutions
- **Parameters**: ~5M
- **Note**: Requires more memory

```yaml
model:
  name: "cnn_3d"
  num_classes: 32
```

## 🐛 Troubleshooting

### Out of Memory Errors

```yaml
# Reduce batch size
training:
  batch_size: 4  # or 2

# Reduce input size
data:
  target_size_2d: [128, 128]
  target_size_3d: [64, 64, 64]

# Disable mixed precision
training:
  mixed_precision: false
```

### Slow Training

```yaml
# Enable mixed precision
training:
  mixed_precision: true

# Increase number of workers
data:
  num_workers: 8

# Use GPU
hardware:
  use_gpu: true
```

### DICOM Loading Errors

```python
# Check DICOM file integrity
from src.data_loader import DICOMLoader

loader = DICOMLoader("./data/raw")
try:
    result = loader.load_dicom_series("./data/raw/patient_001")
    print(f"Successfully loaded: {result['metadata']}")
except Exception as e:
    print(f"Error: {e}")
```

### Model Not Converging

```yaml
# Adjust learning rate
training:
  learning_rate: 0.0001  # Lower

# Try different optimizer
training:
  optimizer: "sgd"
  momentum: 0.9

# Increase regularization
training:
  weight_decay: 0.0001

# Check data preprocessing
preprocessing:
  normalization:
    method: "z_score"  # Try different normalization
```

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/ --port 6006

# Open browser to http://localhost:6006
```

Metrics tracked:
- Training loss (per batch and per epoch)
- Validation loss
- Learning rate
- Custom metrics (accuracy, IoU, etc.)

### Logs

```bash
# View training logs
tail -f logs/pipeline.log

# Check for errors
grep ERROR logs/pipeline.log
```

## 🐳 Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# Run
CMD ["python", "main.py", "--mode", "train", "--config", "configs/config.yaml"]
```

Build and run:

```bash
docker build -t cbct-pipeline .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/saved_models:/app/saved_models cbct-pipeline
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cbct_pipeline,
  title={CBCT Dental Image Analysis Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cbct-pipeline}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- PyDICOM and SimpleITK for DICOM processing
- scikit-image for image processing utilities

---

**Happy Training! 🚀**
