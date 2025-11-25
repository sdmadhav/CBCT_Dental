"""
Training, Evaluation, Inference and Utility Modules
"""

# ============================================================================
# train.py
# ============================================================================

"""
Training Module with Complete Training Loop
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Trainer:
    """Complete training pipeline with all bells and whistles."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        loss_name = config['training']['loss']
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'dice':
            self.criterion = DiceLoss()
        elif loss_name == 'focal':
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Optimizer
        opt_name = config['training']['optimizer']
        lr = config['training']['learning_rate']
        if opt_name == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config['training'].get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['training'].get('scheduler_patience', 5),
            factor=0.5
        )
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config['paths']['log_dir'])
        self.writer = SummaryWriter(self.log_dir)
        
        self.epoch = 0
        self.global_step = 0
        
        logger.info("Trainer initialized")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('train/epoch_loss', avg_loss, self.epoch)
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/loss', avg_loss, self.epoch)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        start_epoch = self.epoch
        
        logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        self.writer.close()
        logger.info("Training completed")


# ============================================================================
# evaluate.py
# ============================================================================

"""
Evaluation Module with Comprehensive Metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)


class Evaluator:
    """Model evaluation with comprehensive metrics."""
    
    def __init__(self, model: nn.Module, test_loader, device: torch.device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        
        logger.info("Evaluator initialized")
    
    def evaluate(self) -> Dict:
        """Run full evaluation."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        return metrics
    
    def calculate_metrics(self, labels, preds, probs) -> Dict:
        """Calculate all metrics."""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(labels, preds)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # Classification report
        metrics['classification_report'] = classification_report(labels, preds)
        
        logger.info(f"Evaluation metrics: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved: {save_path}")
    
    def calculate_iou(self, pred_mask, true_mask) -> float:
        """Calculate IoU for segmentation."""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union


# ============================================================================
# inference.py
# ============================================================================

"""
Inference Module for Production Deployment
"""


class InferenceEngine:
    """Production inference engine."""
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
        preprocessor=None
    ):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"InferenceEngine loaded: {checkpoint_path}")
    
    def predict_single(self, image: np.ndarray) -> Dict:
        """Predict on single image."""
        start_time = time.time()
        
        # Preprocess
        if self.preprocessor:
            image = self.preprocessor.preprocess_pipeline(image)
        
        # Convert to tensor
        if len(image.shape) == 2:
            image = image[np.newaxis, np.newaxis, ...]
        else:
            image = image[np.newaxis, ...]
        
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        inference_time = time.time() - start_time
        
        return {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy()[0],
            'inference_time': inference_time
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict on batch of images."""
        results = []
        for image in tqdm(images, desc="Inference"):
            result = self.predict_single(image)
            results.append(result)
        return results
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results to JSON."""
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results exported: {output_path}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ============================================================================
# utils.py
# ============================================================================

"""
Utility Functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set: {seed}")


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def plot_slice(image: np.ndarray, title: str = "Slice", save_path: Optional[str] = None):
    """Plot 2D slice."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_volume(volume: np.ndarray, threshold: float = 0.5, save_path: Optional[str] = None):
    """Simple 3D visualization."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization
    z, y, x = np.where(volume > threshold)
    ax.scatter(x, y, z, c=volume[z, y, x], cmap='hot', marker='.')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Volume Visualization')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_device(use_gpu: bool = True) -> torch.device:
    """Get PyTorch device."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: Tuple):
    """Print model summary."""
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    logger.info(f"Input size: {input_size}")


# Example usage
if __name__ == "__main__":
    setup_logging()
    set_seed(42)
    
    device = get_device()
    print(f"Device: {device}")
