"""
Configuration File (config.yaml) and Main Pipeline Orchestrator
"""

# ============================================================================
# config.yaml
# ============================================================================

CONFIG_YAML = """
# CBCT Dental Image Analysis Pipeline Configuration

# Project metadata
project:
  name: "CBCT_Dental_Analysis"
  version: "1.0.0"
  description: "Deep learning pipeline for CBCT dental image analysis"

# Data paths
paths:
  data_dir: "./data/raw"
  cache_dir: "./data/processed/cache"
  processed_dir: "./data/processed"
  checkpoint_dir: "./saved_models"
  log_dir: "./logs"
  results_dir: "./results"

# Data configuration
data:
  expected_shape: [401, 401, 251]  # Z, Y, X
  expected_dtype: "int16"
  target_size_2d: [256, 256]
  target_size_3d: [128, 128, 128]
  split_ratios: [0.7, 0.15, 0.15]  # train, val, test
  use_cache: true
  num_workers: 4

# Preprocessing configuration
preprocessing:
  normalization:
    method: "min_max"  # Options: min_max, z_score, window
    clip_percentile: [1, 99]
  
  denoising:
    method: "gaussian"  # Options: gaussian, bilateral, median, none
    sigma: 1.0
    d: 9
    sigma_color: 75
    sigma_space: 75
  
  contrast:
    method: "clahe"  # Options: clahe, histogram_eq, none
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  
  windowing:
    bone_window: [-400, 1800]
    soft_tissue_window: [-160, 240]
  
  edge_enhancement:
    enable: false
    method: "sobel"  # Options: sobel, canny

# Data augmentation configuration
augmentation:
  rotation:
    enabled: true
    range: [-15, 15]
  flip:
    enabled: true
    horizontal: true
    vertical: false
  zoom:
    enabled: true
    range: [0.9, 1.1]
  shift:
    enabled: true
    range: [-0.1, 0.1]
  elastic:
    enabled: false
    alpha: 100
    sigma: 10
  noise:
    enabled: false
    std: 0.01

# Model configuration
model:
  name: "resnet18_2d"  # Options: resnet18_2d, resnet50_2d, unet_2d, cnn_3d
  num_classes: 32  # Number of teeth or classification classes
  pretrained: true
  backbone: "resnet18"
  
  # Model-specific parameters
  unet:
    base_channels: 64
  cnn_3d:
    channels: [32, 64, 128]

# Training configuration
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"  # Options: adam, sgd
  momentum: 0.9  # For SGD
  loss: "cross_entropy"  # Options: cross_entropy, dice, focal
  
  # Learning rate scheduling
  scheduler_patience: 5
  scheduler_factor: 0.5
  
  # Early stopping
  early_stopping_patience: 10
  
  # Mixed precision training
  mixed_precision: true
  
  # Gradient clipping
  gradient_clip: 1.0
  
  # Checkpointing
  save_frequency: 5  # Save every N epochs
  keep_best_only: false

# Evaluation configuration
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "iou"]
  confusion_matrix: true
  roc_curve: true
  save_predictions: true
  visualize_results: true

# Inference configuration
inference:
  checkpoint: "./saved_models/best_model.pth"
  output_format: "json"  # Options: json, csv
  batch_size: 16
  visualize: true
  save_overlays: true

# Logging configuration
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
  log_file: "./logs/pipeline.log"
  tensorboard: true

# Hardware configuration
hardware:
  use_gpu: true
  gpu_id: 0
  num_gpus: 1
  pin_memory: true
  
# Random seed for reproducibility
seed: 42
"""

# ============================================================================
# main.py
# ============================================================================

"""
Main Pipeline Orchestrator
Command-line interface for the complete CBCT analysis pipeline
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import yaml
import torch

# Import all modules (assuming they're in the src directory)
from src.data_loader import DICOMLoader
from src.preprocessing import CBCTPreprocessor, DataAugmenter
from src.feature_extraction import FeatureExtractor
from src.dataset import CBCTDataset2D, CBCTDataset3D, create_data_splits, get_dataloaders
from src.models import get_model
from src.train import Trainer
from src.evaluate import Evaluator
from src.inference import InferenceEngine
from src.utils import (
    set_seed, setup_logging, get_device,
    print_model_summary, plot_slice, plot_3d_volume
)

logger = logging.getLogger(__name__)


class CBCTPipeline:
    """Main pipeline orchestrator for CBCT analysis."""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        setup_logging(self.config['logging'].get('log_file'))
        
        # Set random seed
        set_seed(self.config['seed'])
        
        # Setup device
        self.device = get_device(self.config['hardware']['use_gpu'])
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        
        logger.info(f"Pipeline initialized: {self.config['project']['name']}")
    
    def _create_directories(self):
        """Create necessary directories."""
        for key, path in self.config['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
        logger.info("Directories created")
    
    def load_data(self):
        """Load and prepare data."""
        logger.info("Loading DICOM data...")
        
        self.data_loader = DICOMLoader(
            data_dir=self.config['paths']['data_dir'],
            cache_dir=self.config['paths']['cache_dir'],
            use_cache=self.config['data']['use_cache'],
            expected_shape=tuple(self.config['data']['expected_shape']),
            expected_dtype=self.config['data']['expected_dtype']
        )
        
        # Load batch of patient scans
        patient_list = self.data_loader.get_patient_list()
        logger.info(f"Found {len(patient_list)} patients")
        
        # For demo, we'll create dummy data list
        # In production, this would load actual data
        data_list = []
        for i, patient in enumerate(patient_list[:10]):  # Limit for demo
            data_list.append({
                'patient_id': patient,
                'volume_path': f"{self.config['paths']['cache_dir']}/{patient}_volume.pkl",
                'label': i % self.config['model']['num_classes']  # Dummy label
            })
        
        # Create train/val/test splits
        self.data_splits = create_data_splits(
            data_list,
            split_ratios=tuple(self.config['data']['split_ratios']),
            shuffle=True,
            seed=self.config['seed']
        )
        
        logger.info("Data loading completed")
    
    def preprocess_data(self):
        """Setup preprocessing pipeline."""
        logger.info("Setting up preprocessing...")
        
        self.preprocessor = CBCTPreprocessor(self.config['preprocessing'])
        self.augmenter = DataAugmenter(self.config['augmentation'])
        
        logger.info("Preprocessing setup completed")
    
    def create_dataloaders(self):
        """Create PyTorch dataloaders."""
        logger.info("Creating dataloaders...")
        
        # Determine dataset type based on model
        dataset_type = '3d' if 'cnn_3d' in self.config['model']['name'] else '2d'
        
        self.dataloaders = get_dataloaders(
            data_splits=self.data_splits,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            dataset_type=dataset_type,
            target_size=tuple(self.config['data'][f'target_size_{dataset_type}'])
        )
        
        logger.info(f"Dataloaders created ({dataset_type})")
    
    def build_model(self):
        """Build the model."""
        logger.info("Building model...")
        
        model_config = {
            'num_classes': self.config['model']['num_classes'],
            'pretrained': self.config['model']['pretrained']
        }
        
        # Add model-specific configs
        if 'unet' in self.config['model']['name']:
            model_config['base_channels'] = self.config['model']['unet']['base_channels']
        
        self.model = get_model(self.config['model']['name'], **model_config)
        
        # Print model summary
        if '3d' in self.config['model']['name']:
            input_size = (1, 1, *self.config['data']['target_size_3d'])
        else:
            input_size = (1, 1, *self.config['data']['target_size_2d'])
        
        print_model_summary(self.model, input_size)
        
        logger.info("Model built successfully")
    
    def train(self, resume_from: str = None):
        """Train the model."""
        logger.info("Starting training...")
        
        # Ensure model and dataloaders are ready
        if self.model is None:
            self.build_model()
        if not hasattr(self, 'dataloaders'):
            self.create_dataloaders()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            train_loader=self.dataloaders['train'],
            val_loader=self.dataloaders['val'],
            config=self.config,
            device=self.device
        )
        
        # Train
        trainer.train(
            num_epochs=self.config['training']['num_epochs'],
            resume_from=resume_from
        )
        
        logger.info("Training completed")
    
    def evaluate(self, checkpoint_path: str = None):
        """Evaluate the model."""
        logger.info("Starting evaluation...")
        
        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = f"{self.config['paths']['checkpoint_dir']}/best_model.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluator
        evaluator = Evaluator(
            model=self.model,
            test_loader=self.dataloaders['test'],
            device=self.device
        )
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        # Print results
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Save results
        results_path = f"{self.config['paths']['results_dir']}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}, f, indent=2)
        
        # Plot confusion matrix
        if self.config['evaluation']['confusion_matrix']:
            cm_path = f"{self.config['paths']['results_dir']}/confusion_matrix.png"
            evaluator.plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
        
        logger.info("Evaluation completed")
    
    def inference(self, input_path: str, output_path: str = None):
        """Run inference on new data."""
        logger.info(f"Running inference on: {input_path}")
        
        # Load checkpoint
        checkpoint_path = self.config['inference']['checkpoint']
        
        # Create inference engine
        inference_engine = InferenceEngine(
            model=self.model,
            checkpoint_path=checkpoint_path,
            device=self.device,
            preprocessor=self.preprocessor
        )
        
        # Load input
        if Path(input_path).is_file():
            # Single file
            result = self.data_loader.load_single_dicom(input_path)
            image = result['image']
            
            # Predict
            prediction = inference_engine.predict_single(image)
            
            logger.info(f"Prediction: {prediction['prediction']}")
            logger.info(f"Confidence: {prediction['confidence']:.4f}")
            
            # Save results
            if output_path:
                inference_engine.export_results([prediction], output_path)
        
        else:
            # Directory of files
            logger.info("Batch inference not yet implemented")
        
        logger.info("Inference completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CBCT Dental Image Analysis Pipeline")
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'inference', 'preprocess'],
        help='Pipeline mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for evaluation/inference)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file or directory (for inference)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file or directory'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = CBCTPipeline(args.config)
        
        if args.mode == 'train':
            pipeline.load_data()
            pipeline.preprocess_data()
            pipeline.create_dataloaders()
            pipeline.build_model()
            pipeline.train(resume_from=args.resume)
        
        elif args.mode == 'evaluate':
            pipeline.load_data()
            pipeline.create_dataloaders()
            pipeline.build_model()
            pipeline.evaluate(checkpoint_path=args.checkpoint)
        
        elif args.mode == 'inference':
            if not args.input:
                logger.error("--input is required for inference mode")
                sys.exit(1)
            
            pipeline.preprocess_data()
            pipeline.build_model()
            pipeline.inference(args.input, args.output)
        
        elif args.mode == 'preprocess':
            pipeline.load_data()
            pipeline.preprocess_data()
            logger.info("Data preprocessing completed")
        
        logger.info(f"Pipeline {args.mode} completed successfully")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================================
# Save config.yaml to file
# ============================================================================

def save_config_yaml():
    """Save the configuration to a YAML file."""
    config_dir = Path("./configs")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(CONFIG_YAML)
    
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    # Uncomment to save config
    # save_config_yaml()
    pass
