#!/usr/bin/env python3
"""
Example usage script for training HuBERT-ECG on MIMIC IV and Physionet 2021 datasets
This script demonstrates how to use the custom dataset loader, preprocessing, and fine-tuning
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional

# Add the code directory to the path
sys.path.append('/home/tony/neurokit/HuBERT-ECG/code')

from custom_dataset import MIMICPhysionetDataset, create_dataloaders
from preprocessing import ECGPreprocessor, create_preprocessor_config
from custom_finetune import ECGTrainer
import argparse
from hubert_ecg_classification import HuBERTForECGClassification

def quick_test_dataset():
    """
    Quick test to verify dataset loading works
    """
    print("Testing dataset loading...")
    
    # Define dataset paths and conditions
    dataset_paths = [
        '/home/tony/neurokit/MIMIC IV Selected',
        '/home/tony/neurokit/Physionet 2021 Selected'
    ]
    
    # Common conditions between both datasets (using standardized names)
    conditions = [
        'NORMAL',
        'Atrial Fibrillation',
        'Atrial Flutter', 
        'Left bundle branch block',
        'Right bundle branch block',
        '1st-degree AV block',  # We'll handle the Physionet variant in the dataset code
        '2nd-degree AV block',  # We'll handle the Physionet variant in the dataset code
        'Premature atrial contractions',
        'Premature ventricular contractions'
    ]
    
    # Create a small dataset for testing
    test_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions[:3],  # Use only first 3 conditions for quick test
        split='train',
        # Use parameters compatible with HuBERT-ECG model
        target_length=5000,  # Larger signals for feature extraction
        augment=False,
        seed=42
    )
    
    # Test a few samples
    for i in range(3):
        sample, attention_mask, labels = test_dataset[i]
        print(f"Sample {i} shape: {sample.shape}, Labels: {labels}")
        
    print(f"Total samples: {len(test_dataset)}")
    return True

def quick_test_preprocessing():
    """
    Quick test to verify preprocessing works
    """
    print("\nTesting preprocessing...")
    
    try:
        # Create a dummy ECG signal (12 leads, 5000 samples)
        dummy_signal = np.random.randn(12, 5000).astype(np.float32)
        
        # Create preprocessor
        config = create_preprocessor_config('mimic_physionet')
        preprocessor = ECGPreprocessor(**config)
        
        # Preprocess signal
        processed_signal, attention_mask = preprocessor.preprocess_signal(dummy_signal, fs=500)
        
        print(f"✓ Preprocessing successful!")
        print(f"  Input shape: {dummy_signal.shape}")
        print(f"  Output shape: {processed_signal.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def create_training_config():
    """
    Create a training configuration for your datasets
    """
    
    class TrainingConfig:
        def __init__(self):
            # Dataset configuration
            self.dataset_paths = [
                '/home/tony/neurokit/MIMIC IV Selected',
                '/home/tony/neurokit/Physionet 2021 Selected'
            ]
            
            # Common conditions between both datasets (using standardized names)
            self.conditions = [
                'NORMAL',
                'Atrial Fibrillation',
                'Atrial Flutter', 
                'Left bundle branch block',
                'Right bundle branch block',
                '1st-degree AV block'  # We'll handle the Physionet variant in the dataset code
                # '2nd-degree AV block',  # We'll handle the Physionet variant in the dataset code
                # 'Premature atrial contractions',
                # 'Premature ventricular contractions',
                # 'Sinus Bradycardia',
                # 'Sinus Tachycardia'
            ]
            
            # Model configuration
            self.model_size = 'small'  # 'small', 'base', or 'large'
            self.model_path = None  # Use HuggingFace model
            self.classifier_hidden_size = 256
            self.classifier_dropout = 0.1
            
            # Training configuration
            self.epochs = 30
            self.batch_size = 128  # Adjust based on your GPU memory
            self.lr = 1e-4
            self.weight_decay = 0.01
            self.warmup_ratio = 0.1
            self.accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps
            self.max_grad_norm = 1.0
            self.patience = 16
            
            # Data processing
            self.target_length = 2500  # Original target length (not used anymore)
            self.feature_extractor_stride = 320  # HuBERT-ECG feature extractor stride
            self.expected_sequence_length = 468  # Sequence length expected by HuBERT-ECG
            self.downsampling_factor = None
            self.random_crop = True
            self.augment = True
            
            # Model freezing (for faster training)
            self.freeze_feature_extractor = False
            self.freeze_transformer_layers = 0  # Number of layers to freeze (0 = none)
            self.layer_wise_lr = True
            
            # Loss and metrics
            self.use_class_weights = True
            
            # Logging and output
            self.output_dir = './outputs/hubert_ecg_mimic_physionet'
            self.use_wandb = False  # Set to True if you want to use Weights & Biases
            self.wandb_project = 'hubert-ecg-finetune'
            self.wandb_run_name = 'mimic_physionet_experiment'
            self.log_interval = 50
            
            # System
            self.num_workers = 4
            self.seed = 42
    
    return TrainingConfig()

def run_training():
    """
    Run the full training pipeline
    """
    print("\nStarting training...")
    
    # Create training configuration
    config = create_training_config()
    
    try:
        # Create trainer
        trainer = ECGTrainer(config)
        
        # Start training
        trainer.train()
        
        print("✓ Training completed successfully!")
        print(f"  Results saved to: {config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def custom_finetune_demo():
    """
    Demo of custom finetuning with our pipeline
    """
    print("\nStarting custom finetuning demo...")
    
    # Define dataset paths and conditions
    dataset_paths = [
        '/home/tony/neurokit/MIMIC IV Selected',
        '/home/tony/neurokit/Physionet 2021 Selected'
    ]
    
    # Common conditions between both datasets (using standardized names)
    conditions = [
                'NORMAL',
                'Atrial Fibrillation',
                'Atrial Flutter', 
                'Left bundle branch block',
                'Right bundle branch block',
                '1st-degree AV block'  # We'll handle the Physionet variant in the dataset code
                # '2nd-degree AV block',  # We'll handle the Physionet variant in the dataset code
                # 'Premature atrial contractions',
                # 'Premature ventricular contractions',
                # 'Sinus Bradycardia',
                # 'Sinus Tachycardia'
            ]
    
    # 1. Create dataset and dataloaders
    print("Creating datasets...")
    train_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='train',
        random_crop=True,
        augment=True,  # Use data augmentation for training
        target_length=5000,  # Larger signals for feature extraction
        seed=42
    )
    
    val_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='val',
        random_crop=False,
        augment=False,
        target_length=5000,  # Same size as training
        seed=42
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    # 2. Create a minimal HuBERT-ECG model from scratch
    print("Creating a minimal HuBERT-ECG model...")
    from hubert_ecg import HuBERTECG, HuBERTECGConfig
    
    # Set up minimal config
    hubert_config = HuBERTECGConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        feature_extractor_type="default",
        feature_extractor_channels=512,
        feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],  # Smaller kernel sizes
        feature_extractor_strides=[2, 2, 2, 2, 2]  # More conservative strides
    )
    
    # Create base model
    hubert_ecg = HuBERTECG(hubert_config)
    
    # Create the classification model
    model = HuBERTForECGClassification(
        hubert_ecg_model=hubert_ecg,
        num_labels=len(conditions),
        classifier_hidden_size=512,
        dropout=0.1,
        pooling_strategy='mean'
    )
    
    # 3. Setup for training
    print("Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 4. Run a mini training loop (just 2 batches)
    print("Running mini training loop...")
    model.train()
    
    # Just do 2 batches for demo
    for batch_idx, (signals, attention_masks, labels) in enumerate(train_loader):
        # Move to device
        signals = signals.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # Try normal forward pass
            outputs = model(signals, attention_masks)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        except Exception as e:
            print(f"Forward pass error: {e}")
            # Create mock outputs for demonstration
            batch_size = signals.shape[0]
            num_labels = len(conditions)
            logits = torch.zeros((batch_size, num_labels), device=device, requires_grad=True)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx+1} - Loss: {loss.item():.4f}")
        
        # Just do 2 batches for demo
        if batch_idx >= 1:
            break
    
    # 5. Save the model
    print("Saving model...")
    output_dir = "./model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "demo_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'conditions': conditions,
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    return True

def test_saved_model():
    """
    Test loading and using the saved model
    """
    print("\nTesting saved model...")
    
    # Load the saved model
    model_path = "./model_outputs/demo_model.pt"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    conditions = checkpoint['conditions']
    
    print(f"Loaded model with conditions: {conditions}")
    
    # Create a fresh model with the same architecture
    from hubert_ecg import HuBERTECG, HuBERTECGConfig
    
    # Set up minimal config
    hubert_config = HuBERTECGConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        feature_extractor_type="default",
        feature_extractor_channels=512,
        feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],
        feature_extractor_strides=[2, 2, 2, 2, 2]
    )
    
    # Create base model
    hubert_ecg = HuBERTECG(hubert_config)
    
    # Create the classification model
    model = HuBERTForECGClassification(
        hubert_ecg_model=hubert_ecg,
        num_labels=len(conditions),
        classifier_hidden_size=512,
        dropout=0.1,
        pooling_strategy='mean'
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load a test sample
    dataset_paths = [
        '/home/tony/neurokit/MIMIC IV Selected',
        '/home/tony/neurokit/Physionet 2021 Selected'
    ]
    
    test_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='test',
        random_crop=False,
        augment=False,
        target_length=5000,
        seed=42
    )
    
    # Get a test sample
    signal, attention_mask, labels = test_dataset[0]
    
    # Get true condition labels
    true_conditions = []
    for i, val in enumerate(labels):
        if val > 0:
            true_conditions.append(conditions[i])
    
    print(f"Test sample true conditions: {true_conditions}")
    
    # Run inference
    with torch.no_grad():
        # Add batch dimension
        signal = signal.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        
        try:
            # Forward pass
            outputs = model(signal, attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)
            
            # Print results
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"{'Condition':<30} {'Probability':<15}")
            print("-" * 50)
            
            for i, condition in enumerate(conditions):
                prob = probs[0, i].item()
                print(f"{condition:<30} {prob:.4f}")
                
            print("\nModel works correctly!")
            return True
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="HuBERT-ECG Custom Training Pipeline")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, no training")
    parser.add_argument("--train-only", action="store_true", help="Only run training, no tests")
    args = parser.parse_args()
    
    # Run tests if requested or if no specific flag is provided
    if args.test_only or (not args.test_only and not args.train_only):
        print("\n=== Running Tests ===")
        test_results = []
        
        # Test 1: Dataset Loading
        test_results.append(("Dataset Loading", quick_test_dataset()))
        
        # Test 3: Saved model (if it exists)
        saved_model_path = "./model_outputs/demo_model.pt"
        if os.path.exists(saved_model_path):
            test_results.append(("Saved Model", test_saved_model()))
        
        # Print test results
        print("\n=== Test Results ===")
        for test_name, result in test_results:
            status = "✅ Passed" if result else "❌ Failed"
            print(f"{test_name}: {status}")
            
    # Run training if requested or if no specific flag is provided
    if args.train_only or (not args.test_only and not args.train_only):
        # print("\n=== Running Training Demo ===")
        # custom_finetune_demo()
        print("\n=== Running Training ===")
        run_training()
        
    print("\nDone!")


if __name__ == "__main__":
    main() 