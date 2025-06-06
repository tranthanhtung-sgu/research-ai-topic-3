import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import wfdb
from scipy import signal
from typing import List, Dict, Tuple, Optional
import glob
from loguru import logger
import random

# Import our enhanced preprocessor
from preprocessing import ECGPreprocessor, create_preprocessor_config

class MIMICPhysionetDataset(Dataset):
    """
    Custom dataset loader for MIMIC IV and Physionet 2021 datasets
    Handles WFDB format files (.dat, .hea) and multi-label classification
    Now with enhanced preprocessing including heartbeat outlier removal
    """
    
    def __init__(
        self,
        dataset_paths: List[str],  # List of paths to dataset folders
        conditions: List[str],     # List of condition names to include
        split: str = 'train',      # 'train', 'val', or 'test'
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # train, val, test
        target_length: int = 2500, # 5 seconds at 500Hz
        downsampling_factor: Optional[int] = None,
        random_crop: bool = False,
        augment: bool = False,
        seed: int = 42,
        # HuBERT-ECG specific parameters
        feature_extractor_stride: int = 320,  # Default stride in HuBERT-ECG
        expected_sequence_length: int = 468,  # Expected by HuBERT-ECG model
        # NEW: Enhanced preprocessing parameters
        use_enhanced_preprocessing: bool = True,
        preprocessor_config: Optional[dict] = None
    ):
        """
        Args:
            dataset_paths: List of paths to dataset folders (e.g., ['/path/to/MIMIC IV Selected', '/path/to/Physionet 2021 Selected'])
            conditions: List of condition folder names to include
            split: Which split to use ('train', 'val', 'test')
            split_ratios: Ratios for train/val/test splits
            target_length: Target signal length in samples (2500 = 5 seconds at 500Hz)
            downsampling_factor: Optional downsampling factor
            random_crop: Whether to use random cropping
            augment: Whether to apply data augmentation
            seed: Random seed for reproducibility
            feature_extractor_stride: Stride of the HuBERT-ECG feature extractor
            expected_sequence_length: Sequence length expected by the HuBERT-ECG model
            use_enhanced_preprocessing: Whether to use enhanced preprocessing
            preprocessor_config: Configuration for ECG preprocessor
        """
        self.dataset_paths = dataset_paths
        self.conditions = conditions
        self.target_length = target_length
        self.downsampling_factor = downsampling_factor
        self.random_crop = random_crop
        self.augment = augment
        self.split = split
        
        # HuBERT-ECG specific parameters
        self.feature_extractor_stride = feature_extractor_stride
        self.expected_sequence_length = expected_sequence_length
        
        # Enhanced preprocessing parameters
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Calculate required input length to match expected sequence length
        # HuBERT-ECG uses a feature extractor with stride, so input length must be compatible
        self.required_input_length = self.expected_sequence_length * self.feature_extractor_stride
        
        # Set up ECG preprocessor
        if use_enhanced_preprocessing:
            if preprocessor_config is None:
                # Use default configuration for MIMIC/Physionet data
                preprocessor_config = create_preprocessor_config('mimic_physionet')
                # Ensure we have heartbeat outlier removal enabled
                preprocessor_config['remove_heartbeat_outliers'] = True
                # Set the target length according to our dataset requirements
                preprocessor_config['target_length'] = self.target_length
            
            self.ecg_preprocessor = ECGPreprocessor(**preprocessor_config)
            logger.info(f"Using enhanced ECG preprocessing with heartbeat outlier removal")
        
        # Load and organize data
        self.data_info = self._load_data_info()
        self.train_data, self.val_data, self.test_data = self._create_splits(split_ratios, seed)
        
        # Select current split
        if split == 'train':
            self.current_data = self.train_data
        elif split == 'val':
            self.current_data = self.val_data
        else:
            self.current_data = self.test_data
            
        logger.info(f"Loaded {len(self.current_data)} samples for {split} split")
        
        # Compute class weights for balanced training
        self.class_weights = self._compute_class_weights()
        
    def _load_data_info(self) -> List[Dict]:
        """Load information about all ECG files and their labels"""
        data_info = []
        
        # Define condition name variants (to handle slight differences between datasets)
        condition_variants = {
            '1st-degree AV block': ['1st-degree AV block', '1st Degree AV Block'],
            '2nd-degree AV block': ['2nd-degree AV block', '2nd Degree AV Block'],
            # Add other variants if needed
        }
        
        for dataset_path in self.dataset_paths:
            dataset_name = os.path.basename(dataset_path)
            
            for condition in self.conditions:
                # Check for condition variants
                condition_folder_names = [condition]
                if condition in condition_variants:
                    condition_folder_names = condition_variants[condition]
                
                # Try each possible folder name
                condition_found = False
                for condition_folder in condition_folder_names:
                    condition_path = os.path.join(dataset_path, condition_folder)
                    
                    if os.path.exists(condition_path):
                        condition_found = True
                        # Find all .dat files in the condition folder
                        dat_files = glob.glob(os.path.join(condition_path, "*.dat"))
                        
                        for dat_file in dat_files:
                            # Get corresponding .hea file
                            hea_file = dat_file.replace('.dat', '.hea')
                            
                            if os.path.exists(hea_file):
                                # Create labels (one-hot encoding for multi-label)
                                labels = [0] * len(self.conditions)
                                condition_idx = self.conditions.index(condition)
                                labels[condition_idx] = 1
                                
                                data_info.append({
                                    'dat_file': dat_file,
                                    'hea_file': hea_file,
                                    'labels': labels,
                                    'condition': condition,
                                    'dataset': dataset_name,
                                    'record_id': os.path.basename(dat_file).replace('.dat', '')
                                })
                
                if not condition_found:
                    logger.warning(f"Condition {condition} not found in {dataset_path}")
        
        logger.info(f"Found {len(data_info)} ECG records across {len(self.conditions)} conditions")
        return data_info
    
    def _create_splits(self, split_ratios: Tuple[float, float, float], seed: int) -> Tuple[List, List, List]:
        """Create train/val/test splits ensuring balanced distribution"""
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # Group by condition to ensure balanced splits
        condition_groups = {}
        for item in self.data_info:
            condition = item['condition']
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(item)
        
        train_data, val_data, test_data = [], [], []
        
        for condition, items in condition_groups.items():
            # Shuffle items for this condition
            random.Random(seed).shuffle(items)
            
            n_items = len(items)
            n_train = int(n_items * train_ratio)
            n_val = int(n_items * val_ratio)
            
            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train + n_val])
            test_data.extend(items[n_train + n_val:])
        
        # Shuffle the final splits
        random.Random(seed).shuffle(train_data)
        random.Random(seed).shuffle(val_data)
        random.Random(seed).shuffle(test_data)
        
        logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training"""
        # Count positive samples for each class
        label_counts = np.zeros(len(self.conditions))
        
        for item in self.current_data:
            labels = np.array(item['labels'])
            label_counts += labels
        
        # Compute weights (inverse frequency)
        total_samples = len(self.current_data)
        weights = []
        
        for count in label_counts:
            if count > 0:
                weight = (total_samples - count) / count
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def _load_wfdb_record(self, dat_file: str) -> np.ndarray:
        """Load ECG signal from WFDB files"""
        # Remove .dat extension to get record name
        record_name = dat_file.replace('.dat', '')
        
        try:
            # Load using wfdb
            record = wfdb.rdrecord(record_name)
            signal = record.p_signal  # Shape: (samples, leads)
            
            # Transpose to get shape: (leads, samples)
            signal = signal.T
            
            # Handle NaN values
            if np.any(np.isnan(signal)):
                # Replace NaN with mean of non-NaN values
                for lead_idx in range(signal.shape[0]):
                    lead_signal = signal[lead_idx]
                    if np.any(np.isnan(lead_signal)):
                        mean_val = np.nanmean(lead_signal)
                        signal[lead_idx] = np.where(np.isnan(lead_signal), mean_val, lead_signal)
            
            return signal.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading {dat_file}: {e}")
            # Return zeros if loading fails
            return np.zeros((12, 5000), dtype=np.float32)
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal"""
        # Check if we should use enhanced preprocessing
        if hasattr(self, 'use_enhanced_preprocessing') and self.use_enhanced_preprocessing and hasattr(self, 'ecg_preprocessor'):
            # Use our enhanced ECG preprocessor
            try:
                processed_signal, _ = self.ecg_preprocessor.preprocess_signal(signal)
                return processed_signal
            except Exception as e:
                logger.warning(f"Error in enhanced preprocessing: {e}. Falling back to basic preprocessing.")
                # Fall back to basic preprocessing
                pass
        
        # Basic preprocessing (original implementation)
        
        # Ensure we have 12 leads
        if signal.shape[0] != 12:
            logger.warning(f"Expected 12 leads, got {signal.shape[0]}")
            if signal.shape[0] < 12:
                # Pad with zeros
                padding = np.zeros((12 - signal.shape[0], signal.shape[1]))
                signal = np.vstack([signal, padding])
            else:
                # Take first 12 leads
                signal = signal[:12]
        
        # Determine target size based on model requirements
        input_length_per_lead = 2000  # Set a reasonable number for ECG sequence length
        
        # Handle signal length to a reasonable size first
        if self.random_crop and signal.shape[1] > input_length_per_lead:
            # Random crop
            start_idx = np.random.randint(0, signal.shape[1] - input_length_per_lead + 1)
            signal = signal[:, start_idx:start_idx + input_length_per_lead]
        else:
            # Center crop or pad to required length
            if signal.shape[1] > input_length_per_lead:
                # Center crop
                start_idx = (signal.shape[1] - input_length_per_lead) // 2
                signal = signal[:, start_idx:start_idx + input_length_per_lead]
            elif signal.shape[1] < input_length_per_lead:
                # Pad with last value
                padding = np.repeat(signal[:, -1:], input_length_per_lead - signal.shape[1], axis=1)
                signal = np.hstack([signal, padding])
        
        # Normalize each lead (z-score normalization)
        for lead_idx in range(signal.shape[0]):
            lead_signal = signal[lead_idx]
            if np.std(lead_signal) > 0:
                signal[lead_idx] = (lead_signal - np.mean(lead_signal)) / np.std(lead_signal)
        
        # Data augmentation
        if self.augment and self.split == 'train':
            signal = self._apply_augmentation_per_lead(signal)
        
        # Flatten the signal (leads * time_samples)
        flattened_signal = signal.reshape(-1)
        
        return flattened_signal.astype(np.float32)
    
    def _apply_augmentation_per_lead(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques to each lead"""
        augmented_signal = signal.copy()
        n_leads = signal.shape[0]
        
        for lead_idx in range(n_leads):
            lead_signal = signal[lead_idx]
            
            # Random noise addition
            if np.random.random() < 0.3:
                noise_std = 0.01 * np.std(lead_signal)
                noise = np.random.normal(0, noise_std, lead_signal.shape)
                augmented_signal[lead_idx] = lead_signal + noise
            
            # Random scaling
            if np.random.random() < 0.3:
                scale_factor = np.random.uniform(0.9, 1.1)
                augmented_signal[lead_idx] = lead_signal * scale_factor
            
            # Random time shift (circular shift)
            if np.random.random() < 0.3:
                shift = np.random.randint(-100, 100)
                augmented_signal[lead_idx] = np.roll(lead_signal, shift)
        
        return augmented_signal
    
    def _compute_attention_mask(self, signal: np.ndarray) -> np.ndarray:
        """Compute attention mask for the signal"""
        # For now, return all ones (attend to all positions)
        # You can implement more sophisticated masking based on signal quality
        return np.ones(len(signal), dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.current_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset"""
        item = self.current_data[idx]
        
        # Load and preprocess signal
        signal = self._load_wfdb_record(item['dat_file'])
        signal = self._preprocess_signal(signal)
        
        # Create attention mask
        attention_mask = self._compute_attention_mask(signal)
        
        # Get labels
        labels = np.array(item['labels'], dtype=np.float32)
        
        return (
            torch.from_numpy(signal),
            torch.from_numpy(attention_mask),
            torch.from_numpy(labels)
        )
    
    def get_condition_names(self) -> List[str]:
        """Get list of condition names"""
        return self.conditions
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced training"""
        return self.class_weights


def create_dataloaders(
    dataset_paths: List[str],
    conditions: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets for each split
    train_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='train',
        **dataset_kwargs
    )
    
    val_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='val',
        **dataset_kwargs
    )
    
    test_dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='test',
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader 