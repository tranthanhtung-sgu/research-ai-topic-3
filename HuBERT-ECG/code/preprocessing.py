import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Optional, List
import warnings
from loguru import logger

# Import our new heartbeat outlier removal module
from ecg_outlier_removal import heartbeat_outlier_removal

class ECGPreprocessor:
    """
    Comprehensive ECG preprocessing pipeline for HuBERT-ECG
    """
    
    def __init__(
        self,
        target_fs: int = 500,          # Target sampling frequency
        target_length: int = 2500,     # Target signal length (5 seconds at 500Hz)
        leads_order: List[str] = None, # Order of ECG leads
        normalize_method: str = 'zscore',  # 'zscore', 'minmax', or 'robust'
        filter_config: dict = None,    # Bandpass filter configuration
        remove_baseline: bool = True,  # Remove baseline wander
        remove_powerline: bool = True, # Remove powerline interference
        clip_outliers: bool = True,    # Clip extreme outliers
        remove_heartbeat_outliers: bool = True,  # NEW: Apply heartbeat outlier removal
    ):
        """
        Initialize the ECG preprocessor
        
        Args:
            target_fs: Target sampling frequency in Hz
            target_length: Target signal length in samples
            leads_order: Order of ECG leads (default: standard 12-lead order)
            normalize_method: Normalization method ('zscore', 'minmax', 'robust')
            filter_config: Bandpass filter configuration
            remove_baseline: Whether to remove baseline wander
            remove_powerline: Whether to remove powerline interference
            clip_outliers: Whether to clip extreme outliers
            remove_heartbeat_outliers: Whether to apply heartbeat outlier removal
        """
        self.target_fs = target_fs
        self.target_length = target_length
        self.normalize_method = normalize_method
        self.remove_baseline = remove_baseline
        self.remove_powerline = remove_powerline
        self.clip_outliers = clip_outliers
        self.remove_heartbeat_outliers = remove_heartbeat_outliers
        
        # Standard 12-lead ECG order
        if leads_order is None:
            self.leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        else:
            self.leads_order = leads_order
        
        # Default filter configuration
        if filter_config is None:
            self.filter_config = {
                'lowcut': 0.5,    # High-pass cutoff (removes baseline wander)
                'highcut': 40.0,  # Low-pass cutoff (removes high-freq noise)
                'order': 4        # Filter order
            }
        else:
            self.filter_config = filter_config
    
    def preprocess_signal(
        self, 
        signal: np.ndarray, 
        fs: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main preprocessing pipeline
        
        Args:
            signal: ECG signal array, shape (leads, samples) or (samples, leads)
            fs: Original sampling frequency (if None, assumes target_fs)
            metadata: Optional metadata about the signal
            
        Returns:
            processed_signal: Preprocessed signal (flattened)
            attention_mask: Attention mask for the signal
        """
        if fs is None:
            fs = self.target_fs
        
        # Ensure correct shape (leads, samples)
        signal = self._ensure_correct_shape(signal)
        
        # Step 1: Handle missing leads
        signal = self._handle_missing_leads(signal)
        
        # Step 2: Handle NaN and infinite values
        signal = self._handle_invalid_values(signal)
        
        # Step 3: Resample if necessary
        if fs != self.target_fs:
            signal = self._resample_signal(signal, fs, self.target_fs)
        
        # NEW STEP: Apply heartbeat outlier removal (before temporal processing)
        if self.remove_heartbeat_outliers:
            signal = self._remove_heartbeat_outliers(signal)
        
        # Step 4: Apply temporal cropping/padding
        signal = self._temporal_processing(signal)
        
        # Step 5: Apply filters
        if self.remove_baseline or self.remove_powerline:
            signal = self._apply_filters(signal)
        
        # Step 6: Remove artifacts and outliers
        if self.clip_outliers:
            signal = self._remove_outliers(signal)
        
        # Step 7: Normalize signal
        signal = self._normalize_signal(signal)
        
        # Step 8: Flatten signal for model input
        flattened_signal = self._flatten_signal(signal)
        
        # Step 9: Create attention mask
        attention_mask = self._create_attention_mask(flattened_signal)
        
        return flattened_signal, attention_mask
    
    def _ensure_correct_shape(self, signal: np.ndarray) -> np.ndarray:
        """Ensure signal has shape (leads, samples)"""
        if signal.ndim == 1:
            # Single lead, reshape to (1, samples)
            signal = signal.reshape(1, -1)
        elif signal.ndim == 2:
            # Check if we need to transpose
            if signal.shape[0] > signal.shape[1]:
                # Likely (samples, leads), transpose to (leads, samples)
                signal = signal.T
        else:
            raise ValueError(f"Invalid signal shape: {signal.shape}")
        
        return signal
    
    def _handle_missing_leads(self, signal: np.ndarray) -> np.ndarray:
        """Handle missing ECG leads"""
        n_leads, n_samples = signal.shape
        
        if n_leads < 12:
            # Pad with zeros for missing leads
            padding = np.zeros((12 - n_leads, n_samples))
            signal = np.vstack([signal, padding])
            logger.warning(f"Padded {12 - n_leads} missing leads with zeros")
        elif n_leads > 12:
            # Take first 12 leads
            signal = signal[:12]
            logger.warning(f"Truncated from {n_leads} to 12 leads")
        
        return signal
    
    def _handle_invalid_values(self, signal: np.ndarray) -> np.ndarray:
        """Handle NaN and infinite values"""
        # Find invalid values
        invalid_mask = ~np.isfinite(signal)
        
        if np.any(invalid_mask):
            # Replace with interpolated values or zeros
            for lead_idx in range(signal.shape[0]):
                lead_signal = signal[lead_idx]
                lead_invalid = invalid_mask[lead_idx]
                
                if np.any(lead_invalid):
                    # Try interpolation first
                    valid_indices = np.where(~lead_invalid)[0]
                    if len(valid_indices) > 1:
                        # Linear interpolation
                        signal[lead_idx] = np.interp(
                            np.arange(len(lead_signal)),
                            valid_indices,
                            lead_signal[valid_indices]
                        )
                    else:
                        # If no valid values, set to zero
                        signal[lead_idx] = 0.0
                        
            logger.warning("Replaced invalid values with interpolated values")
        
        return signal
    
    def _resample_signal(self, signal: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
        """Resample signal to target sampling frequency"""
        if original_fs == target_fs:
            return signal
        
        n_leads, n_samples = signal.shape
        target_samples = int(n_samples * target_fs / original_fs)
        
        resampled_signal = np.zeros((n_leads, target_samples))
        
        for lead_idx in range(n_leads):
            resampled_signal[lead_idx] = resample(signal[lead_idx], target_samples)
        
        logger.info(f"Resampled from {original_fs}Hz to {target_fs}Hz")
        return resampled_signal
    
    def _temporal_processing(self, signal: np.ndarray) -> np.ndarray:
        """Handle temporal cropping and padding"""
        n_leads, n_samples = signal.shape
        
        if n_samples > self.target_length:
            # Crop to target length (take center portion)
            start_idx = (n_samples - self.target_length) // 2
            signal = signal[:, start_idx:start_idx + self.target_length]
        elif n_samples < self.target_length:
            # Pad to target length
            padding_needed = self.target_length - n_samples
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            
            # Pad with edge values
            signal = np.pad(signal, ((0, 0), (left_pad, right_pad)), mode='edge')
        
        return signal
    
    def _apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filtering"""
        n_leads, n_samples = signal.shape
        filtered_signal = np.zeros_like(signal)
        
        # Design bandpass filter
        nyquist = self.target_fs / 2
        low = self.filter_config['lowcut'] / nyquist
        high = self.filter_config['highcut'] / nyquist
        
        # Ensure filter frequencies are valid
        low = max(low, 0.001)  # Avoid very low frequencies
        high = min(high, 0.99)  # Avoid Nyquist frequency
        
        try:
            b, a = butter(self.filter_config['order'], [low, high], btype='band')
            
            for lead_idx in range(n_leads):
                # Apply zero-phase filtering
                filtered_signal[lead_idx] = filtfilt(b, a, signal[lead_idx])
                
        except Exception as e:
            logger.warning(f"Filter application failed: {e}. Using original signal.")
            filtered_signal = signal
        
        return filtered_signal
    
    def _remove_outliers(self, signal: np.ndarray) -> np.ndarray:
        """Remove extreme outliers using statistical thresholding"""
        n_leads, n_samples = signal.shape
        cleaned_signal = signal.copy()
        
        for lead_idx in range(n_leads):
            lead_signal = signal[lead_idx]
            
            # Calculate robust statistics
            q25, q75 = np.percentile(lead_signal, [25, 75])
            iqr = q75 - q25
            
            # Define outlier thresholds
            lower_bound = q25 - 3 * iqr
            upper_bound = q75 + 3 * iqr
            
            # Clip outliers
            cleaned_signal[lead_idx] = np.clip(lead_signal, lower_bound, upper_bound)
        
        return cleaned_signal
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize the signal"""
        if self.normalize_method == 'zscore':
            # Z-score normalization (per lead)
            normalized_signal = np.zeros_like(signal)
            for lead_idx in range(signal.shape[0]):
                lead_signal = signal[lead_idx]
                mean_val = np.mean(lead_signal)
                std_val = np.std(lead_signal)
                if std_val > 1e-8:  # Avoid division by zero
                    normalized_signal[lead_idx] = (lead_signal - mean_val) / std_val
                else:
                    normalized_signal[lead_idx] = lead_signal - mean_val
                    
        elif self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            normalized_signal = np.zeros_like(signal)
            for lead_idx in range(signal.shape[0]):
                lead_signal = signal[lead_idx]
                min_val = np.min(lead_signal)
                max_val = np.max(lead_signal)
                if max_val > min_val:
                    normalized_signal[lead_idx] = (lead_signal - min_val) / (max_val - min_val)
                else:
                    normalized_signal[lead_idx] = lead_signal - min_val
                    
        elif self.normalize_method == 'robust':
            # Robust normalization using median and MAD
            normalized_signal = np.zeros_like(signal)
            for lead_idx in range(signal.shape[0]):
                lead_signal = signal[lead_idx]
                median_val = np.median(lead_signal)
                mad_val = np.median(np.abs(lead_signal - median_val))
                if mad_val > 1e-8:
                    normalized_signal[lead_idx] = (lead_signal - median_val) / mad_val
                else:
                    normalized_signal[lead_idx] = lead_signal - median_val
        else:
            # No normalization
            normalized_signal = signal
        
        return normalized_signal
    
    def _flatten_signal(self, signal: np.ndarray) -> np.ndarray:
        """Flatten signal for model input"""
        # Reshape from (12, target_length) to (12 * target_length,)
        return signal.reshape(-1)
    
    def _create_attention_mask(self, signal: np.ndarray) -> np.ndarray:
        """Create attention mask for the signal"""
        # For now, create a mask that attends to all positions
        # You can implement more sophisticated masking based on signal quality
        attention_mask = np.ones(len(signal), dtype=np.int64)
        
        # Example: mask positions with very low signal energy
        # signal_energy = np.abs(signal)
        # threshold = np.percentile(signal_energy, 5)  # Bottom 5%
        # attention_mask[signal_energy < threshold] = 0
        
        return attention_mask
    
    def process_batch(self, signals: List[np.ndarray], fs_list: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of signals"""
        if fs_list is None:
            fs_list = [self.target_fs] * len(signals)
        
        processed_signals = []
        attention_masks = []
        
        for i, signal in enumerate(signals):
            proc_signal, attention_mask = self.preprocess_signal(signal, fs_list[i])
            processed_signals.append(proc_signal)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        signals_tensor = torch.FloatTensor(np.stack(processed_signals))
        masks_tensor = torch.LongTensor(np.stack(attention_masks))
        
        return signals_tensor, masks_tensor
    
    def get_signal_quality_score(self, signal: np.ndarray) -> float:
        """Compute a signal quality score (0-1, higher is better)"""
        # Simple signal quality assessment
        try:
            # Check for flat lines
            std_vals = np.std(signal, axis=1)
            if np.any(std_vals < 1e-6):
                return 0.0
            
            # Check for excessive noise
            mean_std = np.mean(std_vals)
            if mean_std > 10.0:  # Very noisy
                return 0.3
            
            # Check for artifacts (sudden jumps)
            diff_signal = np.diff(signal, axis=1)
            max_diff = np.max(np.abs(diff_signal))
            if max_diff > 5.0:  # Large artifacts
                return 0.5
            
            # Good quality signal
            return 1.0
            
        except Exception:
            return 0.5  # Unknown quality
    
    def _remove_heartbeat_outliers(self, signal: np.ndarray) -> np.ndarray:
        """Apply heartbeat outlier removal on each lead of the signal"""
        try:
            # Process with heartbeat outlier removal algorithm
            cleaned_signal = heartbeat_outlier_removal(signal, fs=self.target_fs)
            logger.info("Applied heartbeat outlier removal")
            return cleaned_signal
        except Exception as e:
            logger.warning(f"Error in heartbeat outlier removal: {e}. Using original signal.")
            return signal


def create_preprocessor_config(dataset_type: str = 'mimic_physionet') -> dict:
    """Create preprocessing configuration for specific datasets"""
    
    if dataset_type == 'mimic_physionet':
        return {
            'target_fs': 500,
            'target_length': 2500,  # 5 seconds
            'normalize_method': 'zscore',
            'filter_config': {
                'lowcut': 0.5,
                'highcut': 40.0,
                'order': 4
            },
            'remove_baseline': True,
            'remove_powerline': True,
            'clip_outliers': True,
            'remove_heartbeat_outliers': True  # NEW: Enable heartbeat outlier removal
        }
    else:
        # Default configuration
        return {
            'target_fs': 500,
            'target_length': 2500,
            'normalize_method': 'zscore',
            'filter_config': {
                'lowcut': 0.5,
                'highcut': 40.0,
                'order': 4
            },
            'remove_baseline': True,
            'remove_powerline': True,
            'clip_outliers': True,
            'remove_heartbeat_outliers': True  # NEW: Enable heartbeat outlier removal
        } 