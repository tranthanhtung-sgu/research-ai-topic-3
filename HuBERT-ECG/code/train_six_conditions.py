#!/usr/bin/env python3
"""
Script to train HuBERT-ECG model with 6 conditions for 20 epochs
with enhanced preprocessing including heartbeat outlier removal
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import time
import pickle
from tqdm import tqdm

# Add the code directory to the path
sys.path.append('/home/tony/neurokit/HuBERT-ECG/code')

from custom_dataset import MIMICPhysionetDataset
from hubert_ecg_classification import HuBERTForECGClassification
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from preprocessing import ECGPreprocessor, create_preprocessor_config
from ecg_outlier_removal import heartbeat_outlier_removal

class PreprocessedDataset(Dataset):
    """Dataset for preprocessed ECG signals"""
    
    def __init__(self, signals, attention_masks, labels):
        self.signals = signals
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.attention_masks[idx], self.labels[idx]

# Add early stopping class
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after validation loss stops improving
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Call after each epoch to check if training should stop
        
        Args:
            val_loss: Validation loss from current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: Validation loss did not improve for {self.counter} epochs")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  EarlyStopping: Stopping training after {self.patience} epochs without improvement")
                self.early_stop = True
                
        return self.early_stop

# Enhanced data augmentation function
def enhanced_augmentation(signal, fs=500):
    """
    Apply enhanced data augmentation techniques to ECG signal
    
    Args:
        signal: ECG signal with shape (leads, samples)
        fs: Sampling frequency in Hz
        
    Returns:
        Augmented signal
    """
    augmented_signal = signal.copy()
    n_leads, n_samples = signal.shape
    
    # Apply augmentation with higher probability
    for lead_idx in range(n_leads):
        lead_signal = signal[lead_idx]
        
        # Random noise addition (increased probability and variance)
        if np.random.random() < 0.5:  # Increased from 0.3
            noise_std = 0.02 * np.std(lead_signal)  # Increased from 0.01
            noise = np.random.normal(0, noise_std, lead_signal.shape)
            augmented_signal[lead_idx] = lead_signal + noise
        
        # Random scaling (increased range)
        if np.random.random() < 0.5:  # Increased from 0.3
            scale_factor = np.random.uniform(0.8, 1.2)  # Wider range
            augmented_signal[lead_idx] = lead_signal * scale_factor
        
        # Random time shift (increased range)
        if np.random.random() < 0.5:  # Increased from 0.3
            shift = np.random.randint(-200, 200)  # Increased from -100, 100
            augmented_signal[lead_idx] = np.roll(lead_signal, shift)
            
        # NEW: Random baseline wander
        if np.random.random() < 0.3:
            # Create low-frequency baseline wander
            t = np.arange(n_samples)
            freq = np.random.uniform(0.1, 0.5)  # Low frequency
            amplitude = np.random.uniform(0.05, 0.15) * np.std(lead_signal)
            baseline = amplitude * np.sin(2 * np.pi * freq * t / fs)
            augmented_signal[lead_idx] = lead_signal + baseline
            
        # NEW: Random dropout (simulate electrode issues)
        if np.random.random() < 0.2:
            # Zero out small segments
            segment_length = int(np.random.uniform(0.01, 0.05) * n_samples)
            start_idx = np.random.randint(0, n_samples - segment_length)
            augmented_signal[lead_idx, start_idx:start_idx + segment_length] = 0
    
    return augmented_signal

def preprocess_all_data(dataset, preprocessor, device='cpu', batch_size=32, augment=False):
    """
    Preprocess all data in the dataset and save to disk
    
    Args:
        dataset: Dataset to preprocess
        preprocessor: ECG preprocessor
        device: Device to use for preprocessing
        batch_size: Batch size for preprocessing
        augment: Whether to apply enhanced augmentation
        
    Returns:
        Preprocessed dataset
    """
    print(f"Preprocessing all data ({len(dataset)} samples)...")
    
    all_signals = []
    all_attention_masks = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    for batch_idx, (signals, _, labels) in enumerate(tqdm(dataloader)):
        # Process batch
        batch_processed_signals = []
        batch_attention_masks = []
        
        for signal in signals:
            # Convert to numpy
            signal_np = signal.numpy().reshape(12, -1)  # Reshape to (leads, samples)
            
            # Apply enhanced augmentation if requested
            if augment:
                signal_np = enhanced_augmentation(signal_np)
            
            # Apply heartbeat outlier removal
            try:
                signal_np = heartbeat_outlier_removal(signal_np)
            except Exception as e:
                print(f"Warning: Error in heartbeat outlier removal: {e}")
            
            # Apply full preprocessing
            processed_signal, attention_mask = preprocessor.preprocess_signal(signal_np)
            
            batch_processed_signals.append(processed_signal)
            batch_attention_masks.append(attention_mask)
        
        # Convert to tensors
        processed_signals_tensor = torch.FloatTensor(np.stack(batch_processed_signals))
        attention_masks_tensor = torch.LongTensor(np.stack(batch_attention_masks))
        
        # Append to lists
        all_signals.append(processed_signals_tensor)
        all_attention_masks.append(attention_masks_tensor)
        all_labels.append(labels)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples")
    
    # Concatenate all batches
    all_signals = torch.cat(all_signals)
    all_attention_masks = torch.cat(all_attention_masks)
    all_labels = torch.cat(all_labels)
    
    return PreprocessedDataset(all_signals, all_attention_masks, all_labels)

def train_six_conditions():
    """
    Train the model with 6 conditions for 20 epochs
    Using enhanced preprocessing with heartbeat outlier removal
    """
    print("Training HuBERT-ECG model with 6 conditions for 20 epochs")
    print("Using enhanced preprocessing with heartbeat outlier removal")
    print("Anti-overfitting measures: Early stopping, increased regularization, reduced complexity")
    
    # Define dataset paths and conditions
    dataset_paths = [
        '/home/tony/neurokit/MIMIC IV Selected',
        '/home/tony/neurokit/Physionet 2021 Selected'
    ]
    
    # Use only 6 conditions
    conditions = [
        'NORMAL',
        'Atrial Fibrillation',
        'Atrial Flutter',
        'Left bundle branch block',
        'Right bundle branch block',
        '1st-degree AV block'
    ]
    
    print(f"Using {len(conditions)} conditions: {conditions}")
    
    # Create enhanced preprocessor configuration
    preprocessor_config = create_preprocessor_config('mimic_physionet')
    preprocessor_config['remove_heartbeat_outliers'] = True
    print("\nPreprocessing configuration:")
    for key, value in preprocessor_config.items():
        print(f"  {key}: {value}")
    
    # Create ECG preprocessor for the dataset
    preprocessor = ECGPreprocessor(**preprocessor_config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "./outputs/six_conditions_model_anti_overfitting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create preprocessed data directory
    preprocessed_dir = os.path.join(output_dir, "preprocessed_data")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Check if preprocessed data exists
    train_preprocessed_path = os.path.join(preprocessed_dir, "train_preprocessed.pkl")
    val_preprocessed_path = os.path.join(preprocessed_dir, "val_preprocessed.pkl")
    
    if os.path.exists(train_preprocessed_path) and os.path.exists(val_preprocessed_path):
        print("\nLoading preprocessed data...")
        with open(train_preprocessed_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_preprocessed_path, 'rb') as f:
            val_dataset = pickle.load(f)
        
        print(f"Loaded preprocessed data - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        # Create datasets
        print("\nCreating datasets...")
        train_dataset_raw = MIMICPhysionetDataset(
            dataset_paths=dataset_paths,
            conditions=conditions,
            split='train',
            random_crop=True,
            augment=True,  # Use data augmentation for training
            target_length=5000,  # Larger signals for feature extraction
            seed=42
        )
        
        val_dataset_raw = MIMICPhysionetDataset(
            dataset_paths=dataset_paths,
            conditions=conditions,
            split='val',
            random_crop=False,
            augment=False,
            target_length=5000,  # Same size as training
            seed=42
        )
        
        print(f"Raw dataset sizes - Train: {len(train_dataset_raw)}, Val: {len(val_dataset_raw)}")
        
        # Preprocess all data
        print("\nPreprocessing training data with enhanced augmentation...")
        train_dataset = preprocess_all_data(train_dataset_raw, preprocessor, device=device, augment=True)
        
        print("\nPreprocessing validation data...")
        val_dataset = preprocess_all_data(val_dataset_raw, preprocessor, device=device, augment=False)
        
        # Save preprocessed data
        print("\nSaving preprocessed data...")
        with open(train_preprocessed_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(val_preprocessed_path, 'wb') as f:
            pickle.dump(val_dataset, f)
        
        print(f"Saved preprocessed data to {preprocessed_dir}")
    
    # Create dataloaders
    batch_size = 128  # Reduced from 256 to improve generalization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model with reduced complexity
    print("\nCreating model with reduced complexity...")
    
    # Set up config with reduced complexity
    hubert_config = HuBERTECGConfig(
        hidden_size=384,  # Reduced from 512
        num_hidden_layers=6,  # Reduced from 8
        num_attention_heads=6,  # Reduced from 8
        intermediate_size=1536,  # Reduced from 2048
        hidden_act="gelu",
        hidden_dropout_prob=0.2,  # Increased from 0.1
        attention_probs_dropout_prob=0.2,  # Increased from 0.1
        layer_norm_eps=1e-12,
        feature_extractor_type="default",
        feature_extractor_channels=384,  # Reduced from 512
        feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],
        feature_extractor_strides=[2, 2, 2, 2, 2]
    )
    
    # Create base model
    hubert_ecg = HuBERTECG(hubert_config)
    
    # Create the classification model with increased dropout
    model = HuBERTForECGClassification(
        hubert_ecg_model=hubert_ecg,
        num_labels=len(conditions),
        classifier_hidden_size=384,  # Reduced from 512
        dropout=0.3,  # Increased from 0.1
        pooling_strategy='mean'
    )
    
    model = model.to(device)
    
    # Create optimizer and loss with increased weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)  # Increased weight decay from 0.01
    
    # Use class weights for imbalanced data
    class_weights = torch.ones(len(conditions), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    # Training loop
    num_epochs = 40  # Increased max epochs, but with early stopping
    print(f"\nTraining for up to {num_epochs} epochs (with early stopping)...")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    # Create a file to log metrics
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy,val_precision,val_f1,val_auroc,timestamp,learning_rate\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
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
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / max(1, train_steps)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (signals, attention_masks, labels) in enumerate(val_loader):
                # Move to device
                signals = signals.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                try:
                    # Forward pass
                    outputs = model(signals, attention_masks)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # Calculate loss
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    val_steps += 1
                    
                    # Get probabilities for AUROC
                    probs = torch.sigmoid(logits)
                    
                    # Convert logits to predictions (binary)
                    preds = (probs > 0.5).float()
                    
                    # Store predictions, probabilities and labels for metric calculation
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        all_preds = np.vstack(all_preds) if all_preds else np.array([])
        all_labels = np.vstack(all_labels) if all_labels else np.array([])
        all_probs = np.vstack(all_probs) if all_probs else np.array([])
        
        avg_val_loss = val_loss / max(1, val_steps)
        
        # Calculate accuracy (exact match)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate precision (micro-average for multi-label)
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        
        # Calculate F1 score (micro-average for multi-label)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        # Calculate ROC AUC (one-vs-rest for multi-label)
        try:
            # Calculate ROC AUC for each class and average
            auroc_scores = []
            for i in range(len(conditions)):
                if len(np.unique(all_labels[:, i])) > 1:  # Check if both classes are present
                    auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    auroc_scores.append(auroc)
            
            # Calculate average AUROC across all classes
            if auroc_scores:
                avg_auroc = np.mean(auroc_scores)
                # Calculate per-class AUROC
                per_class_auroc = {}
                for i, condition in enumerate(conditions):
                    if len(np.unique(all_labels[:, i])) > 1:
                        per_class_auroc[condition] = roc_auc_score(all_labels[:, i], all_probs[:, i])
            else:
                avg_auroc = 0.0
                per_class_auroc = {condition: 0.0 for condition in conditions}
        except Exception as e:
            print(f"Error calculating AUROC: {e}")
            avg_auroc = 0.0
            per_class_auroc = {condition: 0.0 for condition in conditions}
        
        # Get epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Validation loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {accuracy:.4f}")
        print(f"  Validation precision: {precision:.4f}")
        print(f"  Validation F1 score: {f1:.4f}")
        print(f"  Validation AUROC: {avg_auroc:.4f}")
        print(f"  Per-class AUROC:")
        for condition, score in per_class_auroc.items():
            print(f"    - {condition}: {score:.4f}")
        print(f"  Epoch duration: {epoch_duration:.2f} seconds")
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log metrics
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(metrics_file, "w" if epoch == 0 else "a") as f:
            if epoch == 0:
                # Write header
                f.write("epoch,train_loss,val_loss,val_accuracy,val_precision,val_f1,val_auroc,timestamp,learning_rate\n")
            # Write metrics
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{accuracy:.4f},{precision:.4f},{f1:.4f},{avg_auroc:.4f},{timestamp},{current_lr:.6f}\n")
        
        # Save per-class AUROC to a separate file
        auroc_file = os.path.join(output_dir, f"auroc_epoch_{epoch+1}.csv")
        with open(auroc_file, "w") as f:
            f.write("condition,auroc\n")
            for condition, score in per_class_auroc.items():
                f.write(f"{condition},{score:.4f}\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_f1': f1,
            'val_auroc': avg_auroc,
            'per_class_auroc': per_class_auroc,
            'conditions': conditions,
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")
        
        # Save best model based on loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model_loss.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_f1': f1,
                'val_auroc': avg_auroc,
                'per_class_auroc': per_class_auroc,
                'conditions': conditions,
                'best_metric': 'loss',
            }, best_model_path)
            print(f"  Best loss model saved to {best_model_path}")
            
        # Save best model based on accuracy
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_path = os.path.join(output_dir, f"best_model_accuracy_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_f1': f1,
                'val_auroc': avg_auroc,
                'per_class_auroc': per_class_auroc,
                'conditions': conditions,
                'best_metric': 'accuracy',
            }, best_model_path)
            print(f"  Best accuracy model saved to {best_model_path}")
            
            # Also save as the overall best model
            best_model_path_overall = os.path.join(output_dir, "best_model_accuracy.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_f1': f1,
                'val_auroc': avg_auroc,
                'per_class_auroc': per_class_auroc,
                'conditions': conditions,
                'best_metric': 'accuracy',
            }, best_model_path_overall)
        
        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Models saved to {output_dir}")
    
    # Return path to the best accuracy model
    return os.path.join(output_dir, "best_model_accuracy.pt")

if __name__ == "__main__":
    best_model_path = train_six_conditions() 