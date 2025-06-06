#!/usr/bin/env python3
"""
Custom fine-tuning script for HuBERT-ECG on MIMIC IV and Physionet 2021 datasets
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from custom_dataset import MIMICPhysionetDataset, create_dataloaders
from preprocessing import ECGPreprocessor, create_preprocessor_config
from hubert_ecg_classification import HuBERTForECGClassification
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from transformers import AutoModel

class ECGTrainer:
    """
    Trainer class for fine-tuning HuBERT-ECG on custom datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Initialize model, datasets, and optimizers
        self.setup_model()
        self.setup_datasets()
        self.setup_optimizer()
        self.setup_metrics()
        
        # Training state
        self.best_val_auroc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
    def setup_model(self):
        """Initialize the HuBERT-ECG model"""
        logger.info("Setting up model...")
        
        # Load pretrained model
        if self.config.model_size in ['small', 'base', 'large']:
            logger.info(f"Loading HuBERT-ECG-{self.config.model_size} from HuggingFace")
            self.model = AutoModel.from_pretrained(
                f"Edoardo-BS/hubert-ecg-{self.config.model_size}", 
                trust_remote_code=True
            )
        else:
            # Load from local checkpoint
            logger.info(f"Loading model from {self.config.model_path}")
            checkpoint = torch.load(self.config.model_path, map_location='cpu')
            config = checkpoint['model_config']
            if hasattr(config, 'to_dict'):
                config = HuBERTECGConfig(**config.to_dict())
            self.model = HuBERTECG(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Add classification head
        num_labels = len(self.config.conditions)
        self.model = HuBERTForECGClassification(
            self.model, 
            num_labels=num_labels,
            classifier_hidden_size=self.config.classifier_hidden_size,
            dropout=self.config.classifier_dropout
        )
        
        self.model.to(self.device)
        
        # Freeze layers if specified
        if self.config.freeze_feature_extractor:
            self.model.hubert_ecg.feature_extractor.requires_grad_(False)
            
        if self.config.freeze_transformer_layers > 0:
            for i in range(self.config.freeze_transformer_layers):
                self.model.hubert_ecg.encoder.layers[i].requires_grad_(False)
        
        logger.info(f"Model setup complete. Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_datasets(self):
        """Setup datasets and dataloaders"""
        logger.info("Setting up datasets...")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            dataset_paths=self.config.dataset_paths,
            conditions=self.config.conditions,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            target_length=self.config.target_length,
            downsampling_factor=self.config.downsampling_factor,
            random_crop=self.config.random_crop,
            augment=self.config.augment,
            seed=self.config.seed
        )
        
        # Get class weights for balanced training
        train_dataset = self.train_loader.dataset
        self.class_weights = train_dataset.get_class_weights().to(self.device)
        
        logger.info(f"Dataset sizes - Train: {len(self.train_loader.dataset)}, "
                   f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        logger.info(f"Conditions: {self.config.conditions}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Different learning rates for different parts of the model
        param_groups = []
        
        if self.config.layer_wise_lr:
            # Feature extractor
            if not self.config.freeze_feature_extractor:
                param_groups.append({
                    'params': self.model.hubert_ecg.feature_extractor.parameters(),
                    'lr': self.config.lr * 0.1  # Lower LR for feature extractor
                })
            
            # Transformer layers
            n_layers = len(self.model.hubert_ecg.encoder.layers)
            for i, layer in enumerate(self.model.hubert_ecg.encoder.layers):
                if i >= self.config.freeze_transformer_layers:
                    # Higher LR for later layers
                    lr_mult = 0.5 + 0.5 * (i / n_layers)
                    param_groups.append({
                        'params': layer.parameters(),
                        'lr': self.config.lr * lr_mult
                    })
            
            # Classification head (highest LR)
            param_groups.append({
                'params': self.model.classifier.parameters(),
                'lr': self.config.lr * 2.0
            })
        else:
            # Single learning rate for all parameters
            param_groups.append({
                'params': [p for p in self.model.parameters() if p.requires_grad],
                'lr': self.config.lr
            })
        
        # Optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.epochs // self.config.accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer setup complete. Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    def setup_metrics(self):
        """Setup loss function and metrics"""
        # Multi-label classification loss with class weights
        if self.config.use_class_weights:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        logger.info("Metrics setup complete")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (signals, attention_masks, labels) in enumerate(progress_bar):
            signals = signals.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(signals, attention_mask=attention_masks)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = self.criterion(logits, labels)
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item() * self.config.accumulation_steps,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
        
        return total_loss / num_batches
    
    def validate(self, epoch=None):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for signals, attention_masks, labels in tqdm(self.val_loader, desc="Validating"):
                signals = signals.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(signals, attention_mask=attention_masks)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                predictions = torch.sigmoid(logits).cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        metrics = self.calculate_metrics(all_predictions, all_labels)
        
        logger.info(f"Validation - Loss: {val_loss:.4f}, AUROC: {metrics['auroc']:.4f}, "
                   f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}")
        
        # Log to wandb
        if self.config.use_wandb:
            log_dict = {
                'val_loss': val_loss,
                'val_auroc': metrics['auroc'],
                'val_f1': metrics['f1'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall']
            }
            if epoch is not None:
                log_dict['epoch'] = epoch
            wandb.log(log_dict)
        
        return val_loss, metrics
    
    def calculate_metrics(self, predictions, labels):
        """Calculate evaluation metrics"""
        try:
            # AUROC (macro average)
            auroc = roc_auc_score(labels, predictions, average='macro')
        except:
            auroc = 0.0
        
        # Convert predictions to binary (threshold = 0.5)
        binary_preds = (predictions > 0.5).astype(int)
        
        # Other metrics
        f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
        precision = precision_score(labels, binary_preds, average='macro', zero_division=0)
        recall = recall_score(labels, binary_preds, average='macro', zero_division=0)
        
        return {
            'auroc': auroc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.config.output_dir, 'latest_checkpoint.pt'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'best_checkpoint.pt'))
            logger.info(f"Saved best checkpoint with AUROC: {metrics['auroc']:.4f}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            
            # Check for improvement
            is_best = metrics['auroc'] > self.best_val_auroc
            if is_best:
                self.best_val_auroc = metrics['auroc']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final test evaluation
        self.test()
        
        if self.config.use_wandb:
            wandb.finish()
    
    def test(self):
        """Test the model on test set"""
        logger.info("Running final test evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for signals, attention_masks, labels in tqdm(self.test_loader, desc="Testing"):
                signals = signals.to(self.device)
                attention_masks = attention_masks.to(self.device)
                
                # Forward pass
                outputs = self.model(signals, attention_mask=attention_masks)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Store predictions and labels
                predictions = torch.sigmoid(logits).cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(labels.numpy())
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(all_predictions, all_labels)
        
        logger.info(f"Test Results - AUROC: {test_metrics['auroc']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, "
                   f"Recall: {test_metrics['recall']:.4f}")
        
        # Save test results
        results = {
            'test_metrics': test_metrics,
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'conditions': self.config.conditions
        }
        
        with open(os.path.join(self.config.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune HuBERT-ECG on custom datasets")
    
    # Dataset arguments
    parser.add_argument('--dataset_paths', nargs='+', required=True,
                       help='Paths to dataset folders')
    parser.add_argument('--conditions', nargs='+', required=True,
                       help='List of condition names to include')
    
    # Model arguments
    parser.add_argument('--model_size', default='small', choices=['small', 'base', 'large'],
                       help='HuBERT-ECG model size')
    parser.add_argument('--model_path', default=None,
                       help='Path to local model checkpoint (overrides model_size)')
    parser.add_argument('--classifier_hidden_size', type=int, default=256,
                       help='Hidden size for classification head')
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                       help='Dropout for classification head')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data processing arguments
    parser.add_argument('--target_length', type=int, default=2500,
                       help='Target signal length (samples)')
    parser.add_argument('--downsampling_factor', type=int, default=None,
                       help='Downsampling factor')
    parser.add_argument('--random_crop', action='store_true',
                       help='Use random cropping during training')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation during training')
    
    # Model freezing arguments
    parser.add_argument('--freeze_feature_extractor', action='store_true',
                       help='Freeze feature extractor layers')
    parser.add_argument('--freeze_transformer_layers', type=int, default=0,
                       help='Number of transformer layers to freeze')
    parser.add_argument('--layer_wise_lr', action='store_true',
                       help='Use layer-wise learning rates')
    
    # Loss and metrics arguments
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for balanced training')
    
    # Logging and output arguments
    parser.add_argument('--output_dir', default='./outputs',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', default='hubert-ecg-finetune',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', default=None,
                       help='Wandb run name')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (steps)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logger.add("training.log", rotation="500 MB")
    
    # Create trainer and start training
    trainer = ECGTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 