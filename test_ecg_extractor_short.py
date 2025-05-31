import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis

# Standard 12-lead ECG names
STANDARD_12_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def get_valid_indices(info_dict, key):
    """Extract valid indices from NeuroKit info dictionary."""
    indices = info_dict.get(key, [])
    if indices is None: 
        return []
    if not hasattr(indices, '__iter__'): 
        return []
    return [int(p) for p in indices if pd.notna(p) and isinstance(p, (int, float, np.number)) and p < 1e12]

def extract_basic_features(ecg_signal, sampling_rate=500):
    """Extract basic statistical and morphological features from short ECG signals."""
    features = {}
    
    # Basic statistical features
    features['mean'] = float(np.mean(ecg_signal))
    features['std'] = float(np.std(ecg_signal))
    features['var'] = float(np.var(ecg_signal))
    features['min'] = float(np.min(ecg_signal))
    features['max'] = float(np.max(ecg_signal))
    features['range'] = float(np.max(ecg_signal) - np.min(ecg_signal))
    features['skewness'] = float(skew(ecg_signal))
    features['kurtosis'] = float(kurtosis(ecg_signal))
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        features[f'percentile_{p}'] = float(np.percentile(ecg_signal, p))
    
    # Zero crossings
    zero_crossings = np.where(np.diff(np.signbit(ecg_signal)))[0]
    features['zero_crossings'] = len(zero_crossings)
    features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)
    
    # Signal energy and power
    features['signal_energy'] = float(np.sum(ecg_signal**2))
    features['signal_power'] = float(np.mean(ecg_signal**2))
    features['rms'] = float(np.sqrt(np.mean(ecg_signal**2)))
    
    return features

def extract_frequency_features(ecg_signal, sampling_rate=500):
    """Extract frequency domain features."""
    features = {}
    
    # Compute power spectral density
    freqs, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=min(256, len(ecg_signal)//4))
    
    # Define frequency bands (Hz)
    bands = {
        'very_low': (0.003, 0.04),
        'low': (0.04, 0.15),
        'high': (0.15, 0.4),
        'very_high': (0.4, 0.5)
    }
    
    total_power = np.trapz(psd, freqs)
    
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        features[f'{band_name}_freq_power'] = float(band_power)
        features[f'{band_name}_freq_power_ratio'] = float(band_power / total_power) if total_power > 0 else 0.0
    
    # Spectral centroid
    features['spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(psd)
    features['dominant_frequency'] = float(freqs[dominant_freq_idx])
    features['dominant_power'] = float(psd[dominant_freq_idx])
    
    return features

def extract_morphological_features(ecg_signal, sampling_rate=500):
    """Extract morphological features from ECG signal."""
    features = {}
    
    # First and second derivatives
    first_derivative = np.diff(ecg_signal)
    second_derivative = np.diff(first_derivative)
    
    features['first_derivative_mean'] = float(np.mean(first_derivative))
    features['first_derivative_std'] = float(np.std(first_derivative))
    features['second_derivative_mean'] = float(np.mean(second_derivative))
    features['second_derivative_std'] = float(np.std(second_derivative))
    
    # Peak detection with lower threshold for short signals
    try:
        peaks, _ = signal.find_peaks(ecg_signal, height=np.std(ecg_signal)*0.5, distance=int(0.3*sampling_rate))
        features['num_peaks'] = len(peaks)
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sampling_rate * 1000  # in ms
            features['peak_intervals_mean'] = float(np.mean(peak_intervals))
            features['peak_intervals_std'] = float(np.std(peak_intervals))
            features['estimated_heart_rate'] = float(60000 / np.mean(peak_intervals)) if np.mean(peak_intervals) > 0 else 0.0
        
        if len(peaks) > 0:
            peak_amplitudes = ecg_signal[peaks]
            features['peak_amplitudes_mean'] = float(np.mean(peak_amplitudes))
            features['peak_amplitudes_std'] = float(np.std(peak_amplitudes))
    
    except Exception as e:
        print(f"Error in peak detection: {e}")
        features['num_peaks'] = 0
    
    return features

def extract_neurokit_features(ecg_signal, sampling_rate=500):
    """Extract features using NeuroKit2 with error handling for short signals."""
    features = {}
    
    try:
        # Try basic ECG processing
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
        
        # Extract R-peaks
        rpeaks = get_valid_indices(info, 'ECG_R_Peaks')
        features['neurokit_rpeaks_count'] = len(rpeaks)
        
        if len(rpeaks) > 1:
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # in ms
            features['neurokit_rr_mean'] = float(np.mean(rr_intervals))
            features['neurokit_rr_std'] = float(np.std(rr_intervals))
            features['neurokit_heart_rate'] = float(60000 / np.mean(rr_intervals))
        
        # Try to extract other peaks
        peak_types = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
        for peak_type in peak_types:
            peaks = get_valid_indices(info, peak_type)
            features[f'neurokit_{peak_type.lower()}_count'] = len(peaks)
        
        # Extract cleaned signal features
        if 'ECG_Clean' in signals.columns:
            cleaned_signal = signals['ECG_Clean'].values
            features['neurokit_cleaned_mean'] = float(np.mean(cleaned_signal))
            features['neurokit_cleaned_std'] = float(np.std(cleaned_signal))
    
    except Exception as e:
        print(f"NeuroKit processing failed: {e}")
        features['neurokit_processing_failed'] = True
    
    return features

def save_plot(fig, filename, tight_layout=True):
    """Save matplotlib figure."""
    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    
    try:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Failed to save plot {filename}: {str(e)}")
    
    plt.close(fig)

def plot_ecg_signal(ecg_signal, sampling_rate, output_dir, test_name, lead_name):
    """Plot ECG signal with basic analysis."""
    
    time_axis = np.arange(len(ecg_signal)) / sampling_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Raw signal
    axes[0].plot(time_axis, ecg_signal, 'b-', linewidth=1)
    axes[0].set_title(f'{test_name} - {lead_name} - Raw ECG Signal')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.3)
    
    # Try to detect peaks
    try:
        peaks, _ = signal.find_peaks(ecg_signal, height=np.std(ecg_signal)*0.5, distance=int(0.3*sampling_rate))
        if len(peaks) > 0:
            axes[0].scatter(time_axis[peaks], ecg_signal[peaks], color='red', s=50, zorder=5, 
                          label=f'Detected peaks (n={len(peaks)})')
            axes[0].legend()
    except:
        pass
    
    # Frequency spectrum
    freqs, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=min(256, len(ecg_signal)//4))
    axes[1].semilogy(freqs, psd)
    axes[1].set_title(f'{test_name} - {lead_name} - Power Spectral Density')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].grid(True, alpha=0.3)
    
    # Signal statistics
    axes[2].hist(ecg_signal, bins=30, alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(ecg_signal), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(ecg_signal):.3f}')
    axes[2].axvline(np.median(ecg_signal), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(ecg_signal):.3f}')
    axes[2].set_title(f'{test_name} - {lead_name} - Amplitude Distribution')
    axes[2].set_xlabel('Amplitude (mV)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    save_plot(fig, os.path.join(output_dir, f'{test_name}_{lead_name}_analysis.png'))

def process_test_sample(test_dir, output_base_dir, sampling_rate=500):
    """Process a single test sample and extract all possible features."""
    
    test_name = os.path.basename(test_dir)
    npy_file = os.path.join(test_dir, f"{test_name}.npy")
    
    if not os.path.exists(npy_file):
        print(f"Warning: {npy_file} not found, skipping...")
        return None
    
    print(f"\n--- Processing {test_name} ---")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, test_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load ECG data
        ecg_all_leads = np.load(npy_file)
        print(f"Loaded ECG data shape: {ecg_all_leads.shape}")
        
        # Handle different data shapes
        if ecg_all_leads.ndim == 1:
            ecg_all_leads = ecg_all_leads.reshape(1, -1)
        
        num_leads = ecg_all_leads.shape[0]
        signal_length = ecg_all_leads.shape[1]
        duration = signal_length / sampling_rate
        
        print(f"Number of leads: {num_leads}")
        print(f"Signal length: {signal_length} samples ({duration:.2f} seconds)")
        
        # Process each lead
        all_results = {}
        
        for lead_idx in range(min(num_leads, len(STANDARD_12_LEAD_NAMES))):
            lead_name = STANDARD_12_LEAD_NAMES[lead_idx]
            print(f"Processing Lead {lead_name} (index {lead_idx})...")
            
            ecg_lead = ecg_all_leads[lead_idx, :]
            
            # Extract comprehensive features
            basic_features = extract_basic_features(ecg_lead, sampling_rate)
            freq_features = extract_frequency_features(ecg_lead, sampling_rate)
            morph_features = extract_morphological_features(ecg_lead, sampling_rate)
            neurokit_features = extract_neurokit_features(ecg_lead, sampling_rate)
            
            # Combine all features for this lead
            lead_results = {
                'lead_name': lead_name,
                'lead_index': lead_idx,
                'sampling_rate': sampling_rate,
                'signal_length_samples': signal_length,
                'signal_duration_seconds': duration,
                **basic_features,
                **freq_features,
                **morph_features,
                **neurokit_features
            }
            
            all_results[lead_name] = lead_results
            
            # Generate plots for this lead
            plot_ecg_signal(ecg_lead, sampling_rate, output_dir, test_name, lead_name)
        
        # Save comprehensive results
        if all_results:
            # Save as JSON
            results_file = os.path.join(output_dir, f"{test_name}_ecg_features.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved features to: {results_file}")
            
            # Save summary report
            summary_file = os.path.join(output_dir, f"{test_name}_summary_report.txt")
            with open(summary_file, 'w') as f:
                f.write(f"ECG Analysis Report for {test_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Signal Duration: {duration:.2f} seconds\n")
                f.write(f"Sampling Rate: {sampling_rate} Hz\n")
                f.write(f"Number of Leads: {num_leads}\n\n")
                
                for lead_name, results in all_results.items():
                    f.write(f"Lead {lead_name}:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"  Mean Amplitude: {results.get('mean', 'N/A'):.4f} mV\n")
                    f.write(f"  Std Deviation: {results.get('std', 'N/A'):.4f} mV\n")
                    f.write(f"  Signal Range: {results.get('range', 'N/A'):.4f} mV\n")
                    f.write(f"  Detected Peaks: {results.get('num_peaks', 'N/A')}\n")
                    
                    if 'estimated_heart_rate' in results and results['estimated_heart_rate'] > 0:
                        f.write(f"  Estimated HR: {results['estimated_heart_rate']:.1f} BPM\n")
                    
                    if 'neurokit_heart_rate' in results:
                        f.write(f"  NeuroKit HR: {results['neurokit_heart_rate']:.1f} BPM\n")
                    
                    f.write(f"  Spectral Centroid: {results.get('spectral_centroid', 'N/A'):.2f} Hz\n")
                    f.write(f"  Dominant Frequency: {results.get('dominant_frequency', 'N/A'):.2f} Hz\n")
                    f.write("\n")
            
            print(f"Saved summary to: {summary_file}")
            
        return all_results
        
    except Exception as e:
        print(f"Error processing {test_name}: {e}")
        return None

def create_feature_matrix(all_test_results, output_dir):
    """Create a feature matrix suitable for machine learning."""
    
    if not all_test_results:
        print("No results to create feature matrix")
        return
    
    # Collect all feature names
    all_features = set()
    for test_name, test_results in all_test_results.items():
        for lead_name, lead_results in test_results.items():
            for feature_name in lead_results.keys():
                if feature_name not in ['lead_name', 'lead_index']:
                    all_features.add(f"{lead_name}_{feature_name}")
    
    all_features = sorted(list(all_features))
    
    # Create feature matrix
    feature_matrix = []
    test_names = []
    
    for test_name, test_results in all_test_results.items():
        row = []
        for feature_name in all_features:
            lead_name, feature = feature_name.split('_', 1)
            if lead_name in test_results and feature in test_results[lead_name]:
                value = test_results[lead_name][feature]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row.append(value)
                else:
                    row.append(0.0)  # Default value for missing/invalid data
            else:
                row.append(0.0)  # Default value for missing features
        
        feature_matrix.append(row)
        test_names.append(test_name)
    
    # Save as CSV
    feature_df = pd.DataFrame(feature_matrix, columns=all_features, index=test_names)
    csv_file = os.path.join(output_dir, "test_feature_matrix.csv")
    feature_df.to_csv(csv_file)
    print(f"Saved feature matrix to: {csv_file}")
    
    # Save feature names
    feature_names_file = os.path.join(output_dir, "feature_names.txt")
    with open(feature_names_file, 'w') as f:
        for feature in all_features:
            f.write(f"{feature}\n")
    print(f"Saved feature names to: {feature_names_file}")
    
    return feature_df

def main():
    """Main function to process all test samples."""
    
    # Configuration
    test_base_dir = "test"
    output_base_dir = "test_extracted"
    sampling_rate = 500  # Adjust if needed
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all test directories
    test_dirs = [d for d in os.listdir(test_base_dir) 
                 if os.path.isdir(os.path.join(test_base_dir, d)) and d.startswith('test')]
    test_dirs.sort()
    
    print(f"Found {len(test_dirs)} test directories: {test_dirs}")
    
    # Process each test sample
    all_test_results = {}
    
    for test_dir_name in test_dirs:
        test_dir_path = os.path.join(test_base_dir, test_dir_name)
        results = process_test_sample(test_dir_path, output_base_dir, sampling_rate)
        
        if results:
            all_test_results[test_dir_name] = results
    
    # Save combined results
    combined_results_file = os.path.join(output_base_dir, "all_test_features.json")
    with open(combined_results_file, 'w') as f:
        json.dump(all_test_results, f, indent=2)
    
    # Create feature matrix for ML
    feature_df = create_feature_matrix(all_test_results, output_base_dir)
    
    print(f"\n--- Processing Complete ---")
    print(f"Processed {len(all_test_results)} test samples")
    print(f"Combined results saved to: {combined_results_file}")
    print(f"Individual results and plots saved in: {output_base_dir}")
    if feature_df is not None:
        print(f"Feature matrix shape: {feature_df.shape}")

if __name__ == "__main__":
    main() 