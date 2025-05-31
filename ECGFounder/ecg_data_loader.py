import os
import numpy as np
import pandas as pd
import wfdb
import scipy.io
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
from typing import Dict, List, Tuple

class ECGDataset(Dataset):
    """Dataset class for ECG data from MIMIC IV and Physionet 2021"""
    
    def __init__(self, data: List[np.ndarray], labels: List[int], transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ecg_data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            ecg_data = self.transform(ecg_data)
            
        return torch.tensor(ecg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class ECGDataLoader:
    """Data loader for ECG datasets with support for MIMIC IV and Physionet 2021"""
    
    def __init__(self, mimic_path: str, physionet_path: str):
        self.mimic_path = mimic_path
        self.physionet_path = physionet_path
        
        # Define the 6 conditions mapping
        self.condition_mapping = {
            'NORM': ['NORMAL', 'Sinus_Rhythm'],
            'AFIB': ['Atrial Fibrillation'],
            'AFLT': ['Atrial Flutter'],
            '1dAVb': ['1st-degree AV block', '1st Degree AV Block'],
            'RBBB': ['Right bundle branch block'],
            'LBBB': ['Left bundle branch block']
        }
        
        # Create label encoder for 6 classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB'])
        
    def load_mimic_ecg(self, file_path: str) -> np.ndarray:
        """Load MIMIC IV ECG data (.dat files)"""
        try:
            # Remove extension and load with wfdb
            record_name = os.path.splitext(file_path)[0]
            record = wfdb.rdrecord(record_name)
            return record.p_signal.T  # Transpose to get (n_leads, n_samples)
        except Exception as e:
            print(f"Error loading MIMIC file {file_path}: {e}")
            return None
    
    def load_physionet_ecg(self, file_path: str) -> np.ndarray:
        """Load Physionet 2021 ECG data (.mat files)"""
        try:
            mat_data = scipy.io.loadmat(file_path)
            # Find the ECG data in the mat file
            ecg_data = None
            for key in mat_data.keys():
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    if len(mat_data[key].shape) == 2:
                        ecg_data = mat_data[key]
                        break
            
            if ecg_data is not None:
                # Ensure shape is (n_leads, n_samples)
                if ecg_data.shape[0] > ecg_data.shape[1]:
                    ecg_data = ecg_data.T
                return ecg_data
            else:
                print(f"No valid ECG data found in {file_path}")
                return None
        except Exception as e:
            print(f"Error loading Physionet file {file_path}: {e}")
            return None
    
    def normalize_ecg_length(self, ecg_data: np.ndarray, target_length: int = 5000) -> np.ndarray:
        """Normalize ECG length to target_length samples"""
        n_leads, n_samples = ecg_data.shape
        
        if n_samples == target_length:
            normalized_data = ecg_data
        elif n_samples > target_length:
            # Truncate from center
            start_idx = (n_samples - target_length) // 2
            normalized_data = ecg_data[:, start_idx:start_idx + target_length]
        else:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_length - n_samples))
            normalized_data = np.pad(ecg_data, pad_width, mode='constant', constant_values=0)
        
        # Add data normalization to prevent NaN losses
        # Check for extreme values
        if np.any(np.isnan(normalized_data)) or np.any(np.isinf(normalized_data)):
            print(f"Warning: Found NaN or Inf values in ECG data")
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to reasonable range (z-score normalization per lead)
        for lead in range(normalized_data.shape[0]):
            lead_data = normalized_data[lead, :]
            if np.std(lead_data) > 0:  # Avoid division by zero
                normalized_data[lead, :] = (lead_data - np.mean(lead_data)) / np.std(lead_data)
            else:
                normalized_data[lead, :] = lead_data - np.mean(lead_data)
        
        # Clip extreme values to prevent gradient explosion
        normalized_data = np.clip(normalized_data, -10, 10)
        
        return normalized_data
    
    def load_condition_data(self, condition: str, max_samples_per_condition: int = None) -> Tuple[List[np.ndarray], List[int]]:
        """Load ECG data for a specific condition from both datasets"""
        data_list = []
        labels = []
        
        condition_folders = self.condition_mapping[condition]
        label = self.label_encoder.transform([condition])[0]
        
        # Load from MIMIC IV
        for folder_name in condition_folders:
            mimic_folder = os.path.join(self.mimic_path, folder_name)
            if os.path.exists(mimic_folder):
                print(f"Loading MIMIC IV data from {mimic_folder}")
                dat_files = glob.glob(os.path.join(mimic_folder, "*.dat"))
                
                for dat_file in dat_files:
                    ecg_data = self.load_mimic_ecg(dat_file)
                    if ecg_data is not None and ecg_data.shape[0] == 12:  # Ensure 12-lead
                        ecg_data = self.normalize_ecg_length(ecg_data)
                        data_list.append(ecg_data)
                        labels.append(label)
                        
                        if max_samples_per_condition and len(data_list) >= max_samples_per_condition // 2:
                            break
        
        # Load from Physionet 2021
        for folder_name in condition_folders:
            physionet_folder = os.path.join(self.physionet_path, folder_name)
            if os.path.exists(physionet_folder):
                print(f"Loading Physionet data from {physionet_folder}")
                mat_files = glob.glob(os.path.join(physionet_folder, "*.mat"))
                
                for mat_file in mat_files:
                    ecg_data = self.load_physionet_ecg(mat_file)
                    if ecg_data is not None and ecg_data.shape[0] == 12:  # Ensure 12-lead
                        ecg_data = self.normalize_ecg_length(ecg_data)
                        data_list.append(ecg_data)
                        labels.append(label)
                        
                        if max_samples_per_condition and len(data_list) >= max_samples_per_condition:
                            break
        
        print(f"Loaded {len(data_list)} samples for condition {condition}")
        return data_list, labels
    
    def load_all_data(self, max_samples_per_condition: int = 1000) -> Tuple[List[np.ndarray], List[int]]:
        """Load data for all 6 conditions"""
        all_data = []
        all_labels = []
        
        for condition in self.condition_mapping.keys():
            data, labels = self.load_condition_data(condition, max_samples_per_condition)
            all_data.extend(data)
            all_labels.extend(labels)
        
        print(f"\nTotal loaded: {len(all_data)} samples")
        print("Class distribution:")
        for i, condition in enumerate(['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB']):
            count = all_labels.count(i)
            print(f"  {condition}: {count} samples")
        
        return all_data, all_labels
    
    def create_dataloaders(self, batch_size: int = 32, test_size: float = 0.2, 
                          max_samples_per_condition: int = 1000) -> Tuple[DataLoader, DataLoader]:
        """Create train and test dataloaders"""
        # Load all data
        all_data, all_labels = self.load_all_data(max_samples_per_condition)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_data, all_labels, test_size=test_size, 
            stratify=all_labels, random_state=42
        )
        
        # Create datasets
        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
    
    def get_class_names(self) -> List[str]:
        """Get the class names in order"""
        return ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB'] 