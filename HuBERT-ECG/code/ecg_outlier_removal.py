import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import warnings
from typing import Tuple, List, Dict, Optional

class ECGOutlierRemoval:
    """
    Implementation of real-time heartbeat outlier removal in ECG signals.
    Based on the paper "Real-time heartbeat outlier removal in electrocardiogram (ECG) biometric system"
    """
    
    def __init__(
        self,
        fs: int = 500,              # Sampling frequency in Hz
        window_size: int = 5000,    # Window size for processing (10 seconds at 500Hz)
        r_peak_threshold: float = 0.5,  # Adaptive threshold multiplier for R peak detection
        diff_threshold: float = 0.3,    # Threshold for template dissimilarity detection
        template_length: int = 160,     # Template window length (320ms at 500Hz)
        min_bpm: int = 40,          # Minimum expected heart rate (bpm)
        max_bpm: int = 200,         # Maximum expected heart rate (bpm)
        use_adaptive_threshold: bool = True,  # Use adaptive thresholding
        correlation_threshold: float = 0.7  # Minimum correlation for template matching
    ):
        """
        Initialize the ECG heartbeat outlier removal processor
        
        Args:
            fs: Sampling frequency in Hz
            window_size: Window size for processing in samples
            r_peak_threshold: Adaptive threshold multiplier for R peak detection
            diff_threshold: Threshold for template dissimilarity detection
            template_length: Template window length in samples
            min_bpm: Minimum expected heart rate (bpm)
            max_bpm: Maximum expected heart rate (bpm)
            use_adaptive_threshold: Whether to use adaptive thresholding
            correlation_threshold: Minimum correlation for template matching
        """
        self.fs = fs
        self.window_size = window_size
        self.r_peak_threshold = r_peak_threshold
        self.diff_threshold = diff_threshold
        self.template_length = template_length
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.use_adaptive_threshold = use_adaptive_threshold
        self.correlation_threshold = correlation_threshold
        
        # Calculate minimum and maximum RR interval in samples
        self.min_rr = int(60 * fs / max_bpm)
        self.max_rr = int(60 * fs / min_bpm)
        
        # For storing heartbeat templates
        self.templates = []
        self.median_template = None
        
    def preprocess_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess the ECG signal before outlier detection
        
        Args:
            ecg_signal: Raw ECG signal (single lead)
            
        Returns:
            preprocessed_signal: Filtered and normalized ECG signal
        """
        # Apply bandpass filter (0.5-40Hz)
        sos = signal.butter(4, [0.5, 40], 'bandpass', fs=self.fs, output='sos')
        filtered = signal.sosfilt(sos, ecg_signal)
        
        # Apply notch filter to remove power line interference (50/60Hz)
        notch_freq = 50  # or 60 depending on region
        quality_factor = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.fs)
        notched = signal.filtfilt(b_notch, a_notch, filtered)
        
        # Normalize the signal
        mean_val = np.mean(notched)
        std_val = np.std(notched)
        if std_val > 1e-10:
            normalized = (notched - mean_val) / std_val
        else:
            normalized = notched - mean_val
            
        return normalized
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R peaks in the ECG signal using an adaptive threshold
        
        Args:
            ecg_signal: Preprocessed ECG signal
            
        Returns:
            r_peaks: Array of R peak indices
        """
        # Compute signal derivative to enhance QRS complexes
        diff_signal = np.diff(ecg_signal)
        squared_diff = diff_signal ** 2
        
        # Smooth the signal with a moving average filter
        window_size = int(0.08 * self.fs)  # 80ms window
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
            
        smoothed = signal.savgol_filter(squared_diff, window_size, 2)
        
        # Determine threshold
        if self.use_adaptive_threshold:
            # Adaptive threshold based on signal statistics
            signal_mean = np.mean(smoothed)
            signal_std = np.std(smoothed)
            threshold = signal_mean + self.r_peak_threshold * signal_std
        else:
            # Fixed threshold
            threshold = self.r_peak_threshold * np.max(smoothed)
        
        # Find peaks above threshold
        peaks, _ = find_peaks(smoothed, height=threshold, distance=self.min_rr)
        
        # Adjust peak positions to the original signal
        r_peaks = []
        for peak in peaks:
            # Look for the actual R peak in a small window around the detected peak
            window_start = max(0, peak - 10)
            window_end = min(len(ecg_signal) - 1, peak + 10)
            max_pos = np.argmax(ecg_signal[window_start:window_end]) + window_start
            r_peaks.append(max_pos)
        
        return np.array(r_peaks)
    
    def extract_heartbeat_templates(self, ecg_signal: np.ndarray, r_peaks: np.ndarray) -> List[np.ndarray]:
        """
        Extract heartbeat templates around R peaks
        
        Args:
            ecg_signal: Preprocessed ECG signal
            r_peaks: Array of R peak indices
            
        Returns:
            templates: List of heartbeat templates
        """
        templates = []
        half_length = self.template_length // 2
        
        for peak in r_peaks:
            # Extract a window centered at the R peak
            start = peak - half_length
            end = peak + half_length
            
            # Handle boundary conditions
            if start < 0 or end >= len(ecg_signal):
                continue
                
            template = ecg_signal[start:end]
            templates.append(template)
        
        return templates
    
    def compute_median_template(self, templates: List[np.ndarray]) -> np.ndarray:
        """
        Compute the median template from a list of templates
        
        Args:
            templates: List of heartbeat templates
            
        Returns:
            median_template: Median heartbeat template
        """
        if not templates:
            return None
            
        # Stack templates and compute median along the first axis
        templates_array = np.stack(templates)
        median_template = np.median(templates_array, axis=0)
        
        return median_template
    
    def detect_outliers(self, templates: List[np.ndarray], median_template: np.ndarray) -> List[int]:
        """
        Detect outlier heartbeats based on template correlation
        
        Args:
            templates: List of heartbeat templates
            median_template: Median heartbeat template
            
        Returns:
            outlier_indices: Indices of outlier templates
        """
        outlier_indices = []
        
        for i, template in enumerate(templates):
            # Compute correlation with median template
            correlation = np.corrcoef(template, median_template)[0, 1]
            
            # Compute normalized difference
            diff = np.mean(np.abs(template - median_template)) / np.max(np.abs(median_template))
            
            # Classify as outlier if correlation is low or difference is high
            if correlation < self.correlation_threshold or diff > self.diff_threshold:
                outlier_indices.append(i)
        
        return outlier_indices
    
    def remove_outliers(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove outlier heartbeats from the ECG signal
        
        Args:
            ecg_signal: Raw ECG signal (single lead)
            
        Returns:
            cleaned_signal: ECG signal with outliers removed
            metadata: Dictionary with metadata about the cleaning process
        """
        # Preprocess the signal
        preprocessed = self.preprocess_signal(ecg_signal)
        
        # Detect R peaks
        r_peaks = self.detect_r_peaks(preprocessed)
        
        if len(r_peaks) < 3:
            # Not enough heartbeats to perform outlier detection
            return ecg_signal, {"outliers_removed": 0, "total_beats": len(r_peaks)}
        
        # Extract heartbeat templates
        templates = self.extract_heartbeat_templates(preprocessed, r_peaks)
        
        if len(templates) < 3:
            # Not enough templates to perform outlier detection
            return ecg_signal, {"outliers_removed": 0, "total_beats": len(templates)}
        
        # Compute median template
        median_template = self.compute_median_template(templates)
        
        # Detect outliers
        outlier_indices = self.detect_outliers(templates, median_template)
        
        # Create cleaned signal (replace outliers with interpolated values)
        cleaned_signal = ecg_signal.copy()
        
        for idx in outlier_indices:
            if idx < len(r_peaks):
                peak = r_peaks[idx]
                half_length = self.template_length // 2
                
                start = peak - half_length
                end = peak + half_length
                
                # Handle boundary conditions
                if start < 0 or end >= len(ecg_signal):
                    continue
                
                # Replace outlier with interpolated values
                if idx > 0 and idx < len(templates) - 1:
                    # Use the average of previous and next templates
                    prev_template = templates[idx - 1]
                    next_template = templates[idx + 1]
                    replacement = (prev_template + next_template) / 2
                    
                    # Scale to match the signal amplitude
                    scaling_factor = np.max(ecg_signal[start:end]) / np.max(replacement) if np.max(replacement) > 0 else 1
                    replacement = replacement * scaling_factor
                    
                    # Replace the segment
                    cleaned_signal[start:end] = replacement
                else:
                    # Use the median template if we can't interpolate
                    if median_template is not None:
                        # Scale to match the signal amplitude
                        scaling_factor = np.max(ecg_signal[start:end]) / np.max(median_template) if np.max(median_template) > 0 else 1
                        replacement = median_template * scaling_factor
                        
                        # Replace the segment
                        cleaned_signal[start:end] = replacement
        
        metadata = {
            "outliers_removed": len(outlier_indices),
            "total_beats": len(templates),
            "outlier_percentage": (len(outlier_indices) / len(templates)) * 100 if templates else 0
        }
        
        return cleaned_signal, metadata
    
    def process_multi_lead_signal(self, multi_lead_signal: np.ndarray) -> np.ndarray:
        """
        Process a multi-lead ECG signal to remove outliers in each lead
        
        Args:
            multi_lead_signal: Multi-lead ECG signal with shape (leads, samples)
            
        Returns:
            cleaned_signal: Multi-lead ECG signal with outliers removed
        """
        n_leads = multi_lead_signal.shape[0]
        cleaned_signal = np.zeros_like(multi_lead_signal)
        
        for lead_idx in range(n_leads):
            lead_signal = multi_lead_signal[lead_idx]
            cleaned_lead, _ = self.remove_outliers(lead_signal)
            cleaned_signal[lead_idx] = cleaned_lead
        
        return cleaned_signal
    
def heartbeat_outlier_removal(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Wrapper function to perform heartbeat outlier removal on an ECG signal
    
    Args:
        signal: ECG signal with shape (leads, samples) or (samples,)
        fs: Sampling frequency in Hz
        
    Returns:
        cleaned_signal: ECG signal with outliers removed
    """
    # Ensure signal has correct shape
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)  # Convert to 1 lead
    
    # Create processor
    processor = ECGOutlierRemoval(fs=fs)
    
    # Process signal
    cleaned_signal = processor.process_multi_lead_signal(signal)
    
    return cleaned_signal 