import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from pathlib import Path

# Standard 12-lead ECG names
STANDARD_12_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Focus on Lead II only (index 1)
TARGET_LEAD_INDEX = 1
TARGET_LEAD_NAME = 'II'

def get_valid_indices(info_dict, key):
    """Extract valid indices from NeuroKit info dictionary."""
    indices = info_dict.get(key, [])
    if indices is None: 
        return []
    if not hasattr(indices, '__iter__'): 
        return []
    return [int(p) for p in indices if pd.notna(p) and isinstance(p, (int, float, np.number)) and p < 1e12]

def calculate_intervals_and_amplitudes(signals_df, info_dict, rpeaks_indices, sampling_rate):
    """Calculate comprehensive ECG intervals and amplitudes."""
    results = {}
    
    if not isinstance(signals_df, pd.DataFrame) or 'ECG_Clean' not in signals_df.columns:
        print("Error: signals_df is not a DataFrame or missing 'ECG_Clean'.")
        return results
    
    cleaned_ecg = signals_df['ECG_Clean'].values
    if len(cleaned_ecg) == 0:
        print("Error: Cleaned ECG signal is empty.")
        return results

    # --- RR Intervals ---
    if len(rpeaks_indices) > 1:
        rr_samples = np.diff(rpeaks_indices)
        rr_ms = (rr_samples / sampling_rate) * 1000
        results['RR_intervals_ms'] = rr_ms.tolist()  # Convert to list for JSON serialization
        if len(rr_ms) > 0:
            results['RR_mean_ms'] = float(np.mean(rr_ms))
            results['RR_std_ms'] = float(np.std(rr_ms))
            results['RR_median_ms'] = float(np.median(rr_ms))
            results['RR_min_ms'] = float(np.min(rr_ms))
            results['RR_max_ms'] = float(np.max(rr_ms))

    # --- PP Intervals ---
    p_peaks = get_valid_indices(info_dict, 'ECG_P_Peaks')
    if len(p_peaks) > 1:
        pp_samples = np.diff(p_peaks)
        pp_ms = (pp_samples / sampling_rate) * 1000
        if len(pp_ms) > 0:
            results['PP_intervals_ms'] = pp_ms.tolist()
            results['PP_mean_ms'] = float(np.mean(pp_ms))
            results['PP_std_ms'] = float(np.std(pp_ms))

    # --- PR Intervals ---
    p_onsets = get_valid_indices(info_dict, 'ECG_P_Onsets')
    qrs_onsets_candidates = get_valid_indices(info_dict, 'ECG_R_Onsets')
    if not qrs_onsets_candidates:
        q_peaks = get_valid_indices(info_dict, 'ECG_Q_Peaks')
        qrs_onsets_candidates = sorted(list(set(q_peaks))) if q_peaks else []
    
    pr_intervals_ms_list = []
    if p_onsets and qrs_onsets_candidates:
        p_onsets_sorted = sorted(list(set(p_onsets)))
        qrs_onsets_sorted = sorted(list(set(qrs_onsets_candidates)))
        for q_on in qrs_onsets_sorted:
            relevant_p_onsets_before_q = [p_on for p_on in p_onsets_sorted if p_on < q_on]
            if relevant_p_onsets_before_q:
                closest_p_onset = max(relevant_p_onsets_before_q)
                pr_duration_ms = (q_on - closest_p_onset) / sampling_rate * 1000
                if 80 <= pr_duration_ms <= 350:
                    pr_intervals_ms_list.append(pr_duration_ms)
        
        if pr_intervals_ms_list:
            results['PR_intervals_ms'] = pr_intervals_ms_list
            results['PR_mean_ms'] = float(np.mean(pr_intervals_ms_list))
            results['PR_std_ms'] = float(np.std(pr_intervals_ms_list))

    # --- QRS Durations ---
    r_onsets = get_valid_indices(info_dict, 'ECG_R_Onsets')
    s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_S_Offsets')
    if not s_offsets_or_r_offsets:
        s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_R_Offsets')
    
    qrs_durations_ms_list = []
    if r_onsets and s_offsets_or_r_offsets:
        r_onsets_sorted = sorted(list(set(r_onsets)))
        qrs_offsets_sorted = sorted(list(set(s_offsets_or_r_offsets)))
        used_offsets_indices = [False] * len(qrs_offsets_sorted)
        
        for r_on_val in r_onsets_sorted:
            best_qrs_off_val = -1
            min_duration_diff = float('inf')
            best_offset_idx = -1
            
            for i, r_off_val in enumerate(qrs_offsets_sorted):
                if not used_offsets_indices[i] and r_off_val > r_on_val:
                    duration_ms = (r_off_val - r_on_val) / sampling_rate * 1000
                    if 40 <= duration_ms <= 200 and (r_off_val - r_on_val) < min_duration_diff:
                        min_duration_diff = (r_off_val - r_on_val)
                        best_qrs_off_val = r_off_val
                        best_offset_idx = i
            
            if best_qrs_off_val != -1 and best_offset_idx != -1:
                qrs_durations_ms_list.append(min_duration_diff / sampling_rate * 1000)
                used_offsets_indices[best_offset_idx] = True
        
        if qrs_durations_ms_list:
            results['QRS_durations_ms'] = qrs_durations_ms_list
            results['QRS_mean_ms'] = float(np.mean(qrs_durations_ms_list))
            results['QRS_std_ms'] = float(np.std(qrs_durations_ms_list))

    # --- QT Intervals ---
    t_offsets = get_valid_indices(info_dict, 'ECG_T_Offsets')
    qt_intervals_ms_list = []
    qtc_intervals_list = []
    
    if r_onsets and t_offsets and 'RR_intervals_ms' in results and len(results.get('RR_intervals_ms', [])) > 0:
        qrs_onsets_sorted = sorted(list(set(r_onsets)))
        t_offsets_sorted = sorted(list(set(t_offsets)))
        rr_intervals_sec = np.array(results['RR_intervals_ms']) / 1000.0
        used_t_offsets = [False] * len(t_offsets_sorted)
        
        for q_on_idx, q_on_val in enumerate(qrs_onsets_sorted):
            best_t_off_val = -1
            min_qt_duration = float('inf')
            best_t_off_idx = -1
            
            for i, t_off_val in enumerate(t_offsets_sorted):
                if not used_t_offsets[i] and t_off_val > q_on_val:
                    qt_duration_ms_candidate = (t_off_val - q_on_val) / sampling_rate * 1000
                    if 200 <= qt_duration_ms_candidate <= 700 and (t_off_val - q_on_val) < min_qt_duration:
                        min_qt_duration = (t_off_val - q_on_val)
                        best_t_off_val = t_off_val
                        best_t_off_idx = i
            
            if best_t_off_val != -1 and best_t_off_idx != -1:
                qt_duration_ms = min_qt_duration / sampling_rate * 1000
                qt_intervals_ms_list.append(qt_duration_ms)
                used_t_offsets[best_t_off_idx] = True
                
                # Calculate QTc (corrected QT)
                if q_on_idx < len(rr_intervals_sec):
                    rr_sec_for_this_beat = rr_intervals_sec[q_on_idx]
                    if pd.notna(rr_sec_for_this_beat) and rr_sec_for_this_beat > 0.1:
                        qtc_intervals_list.append(qt_duration_ms / np.sqrt(rr_sec_for_this_beat))
        
        if qt_intervals_ms_list:
            results['QT_intervals_ms'] = qt_intervals_ms_list
            results['QT_mean_ms'] = float(np.mean(qt_intervals_ms_list))
            results['QT_std_ms'] = float(np.std(qt_intervals_ms_list))
        
        if qtc_intervals_list:
            results['QTc_intervals'] = qtc_intervals_list
            results['QTc_mean'] = float(np.mean(qtc_intervals_list))
            results['QTc_std'] = float(np.std(qtc_intervals_list))

    # --- Wave Amplitudes ---
    peak_keys_amplitudes = {
        'P': 'ECG_P_Peaks',
        'Q': 'ECG_Q_Peaks', 
        'R': 'ECG_R_Peaks',
        'S': 'ECG_S_Peaks',
        'T': 'ECG_T_Peaks'
    }
    
    results['Amplitudes_mV'] = {}
    for wave_name, peak_key in peak_keys_amplitudes.items():
        peaks = get_valid_indices(info_dict, peak_key)
        if peaks and len(peaks) > 0 and max(peaks) < len(cleaned_ecg):
            amplitudes = cleaned_ecg[peaks]
            if len(amplitudes) > 0:
                results['Amplitudes_mV'][f'{wave_name}_peak_amps'] = amplitudes.tolist()
                results['Amplitudes_mV'][f'{wave_name}_peak_amp_mean'] = float(np.mean(amplitudes))
                results['Amplitudes_mV'][f'{wave_name}_peak_amp_std'] = float(np.std(amplitudes))

    # --- ST Deviations ---
    st_deviations_mv = []
    if r_onsets and s_offsets_or_r_offsets and p_onsets:
        j_points = sorted(list(set(s_offsets_or_r_offsets)))
        qrs_onsets_sorted = sorted(list(set(r_onsets)))
        p_onsets_sorted = sorted(list(set(p_onsets)))
        
        for j_point_val in j_points:
            current_q_on = max([qon for qon in qrs_onsets_sorted if qon < j_point_val], default=None)
            baseline_start_p_onset = max([pon for pon in p_onsets_sorted if pon < (current_q_on if current_q_on is not None else -1)], default=None)
            
            if (current_q_on is not None and baseline_start_p_onset is not None and 
                baseline_start_p_onset < current_q_on and 
                (current_q_on - baseline_start_p_onset) > int(0.04 * sampling_rate)):
                
                pr_segment_for_baseline = cleaned_ecg[baseline_start_p_onset:current_q_on]
                if len(pr_segment_for_baseline) > 0:
                    isoelectric_level = np.median(pr_segment_for_baseline)
                    st_measurement_point_sample = j_point_val + int(0.06 * sampling_rate)
                    if st_measurement_point_sample < len(cleaned_ecg):
                        st_deviations_mv.append(cleaned_ecg[st_measurement_point_sample] - isoelectric_level)
        
        if st_deviations_mv:
            results['ST_deviations_mV'] = st_deviations_mv
            results['ST_mean_deviation_mV'] = float(np.mean(st_deviations_mv))
            results['ST_std_deviation_mV'] = float(np.std(st_deviations_mv))

    return results

def calculate_heart_rate_variability(rpeaks_indices, sampling_rate):
    """Calculate HRV metrics."""
    hrv_results = {}
    
    if len(rpeaks_indices) < 2:
        return hrv_results
    
    try:
        # Calculate HRV using NeuroKit2
        hrv_time = nk.hrv_time(rpeaks_indices, sampling_rate=sampling_rate, show=False)
        hrv_freq = nk.hrv_frequency(rpeaks_indices, sampling_rate=sampling_rate, show=False)
        hrv_nonlinear = nk.hrv_nonlinear(rpeaks_indices, sampling_rate=sampling_rate, show=False)
        
        # Extract key metrics
        if not hrv_time.empty:
            for col in hrv_time.columns:
                if not hrv_time[col].isna().all():
                    hrv_results[f'HRV_time_{col}'] = float(hrv_time[col].iloc[0])
        
        if not hrv_freq.empty:
            for col in hrv_freq.columns:
                if not hrv_freq[col].isna().all():
                    hrv_results[f'HRV_freq_{col}'] = float(hrv_freq[col].iloc[0])
        
        if not hrv_nonlinear.empty:
            for col in hrv_nonlinear.columns:
                if not hrv_nonlinear[col].isna().all():
                    hrv_results[f'HRV_nonlinear_{col}'] = float(hrv_nonlinear[col].iloc[0])
                    
    except Exception as e:
        print(f"Error calculating HRV: {e}")
    
    return hrv_results

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

def plot_ecg_analysis(ecg_signal, signals_df, info_dict, rpeaks_indices, sampling_rate, output_dir, test_name):
    """Generate comprehensive ECG analysis plots for Lead II only."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Raw and processed ECG
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    time_axis = np.arange(len(ecg_signal)) / sampling_rate
    
    # Raw ECG
    ax1.plot(time_axis, ecg_signal, 'b-', linewidth=0.8, label='Raw ECG')
    ax1.set_title(f'{test_name} - Raw ECG Signal (Lead II)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Processed ECG with R-peaks
    cleaned_ecg = signals_df['ECG_Clean'].values
    ax2.plot(time_axis, cleaned_ecg, 'g-', linewidth=0.8, label='Cleaned ECG')
    
    if rpeaks_indices:
        r_times = np.array(rpeaks_indices) / sampling_rate
        r_amplitudes = cleaned_ecg[rpeaks_indices]
        ax2.scatter(r_times, r_amplitudes, color='red', s=50, zorder=5, label=f'R-peaks (n={len(rpeaks_indices)})')
    
    ax2.set_title(f'{test_name} - Processed ECG with R-peaks (Lead II)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    save_plot(fig, os.path.join(output_dir, f'{test_name}_ecg_analysis_Lead_II.png'))
    
    # 2. RR interval analysis
    if len(rpeaks_indices) > 1:
        rr_intervals = np.diff(rpeaks_indices) / sampling_rate * 1000  # in ms
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # RR interval time series
        beat_numbers = np.arange(1, len(rr_intervals) + 1)
        ax1.plot(beat_numbers, rr_intervals, 'bo-', markersize=4, linewidth=1)
        ax1.set_title(f'{test_name} - RR Interval Time Series (Lead II)')
        ax1.set_xlabel('Beat Number')
        ax1.set_ylabel('RR Interval (ms)')
        ax1.grid(True, alpha=0.3)
        
        # RR interval histogram
        ax2.hist(rr_intervals, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(np.mean(rr_intervals), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rr_intervals):.1f} ms')
        ax2.set_title(f'{test_name} - RR Interval Distribution (Lead II)')
        ax2.set_xlabel('RR Interval (ms)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        save_plot(fig, os.path.join(output_dir, f'{test_name}_rr_analysis_Lead_II.png'))
    
    # 3. Heart rate analysis
    if len(rpeaks_indices) > 1:
        # Calculate instantaneous heart rate
        rr_intervals_sec = np.diff(rpeaks_indices) / sampling_rate
        heart_rates = 60 / rr_intervals_sec  # BPM
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        beat_times = np.array(rpeaks_indices[1:]) / sampling_rate
        ax.plot(beat_times, heart_rates, 'ro-', markersize=4, linewidth=1)
        ax.set_title(f'{test_name} - Heart Rate Over Time (Lead II)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heart Rate (BPM)')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_hr = np.mean(heart_rates)
        std_hr = np.std(heart_rates)
        ax.axhline(mean_hr, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Mean HR: {mean_hr:.1f} ± {std_hr:.1f} BPM')
        ax.legend()
        
        save_plot(fig, os.path.join(output_dir, f'{test_name}_heart_rate_Lead_II.png'))

def process_test_sample(test_dir, output_base_dir, sampling_rate=100):
    """Process a single test sample and extract ECG features from Lead II only."""
    
    test_name = os.path.basename(test_dir)
    npy_file = os.path.join(test_dir, f"{test_name}.npy")
    
    if not os.path.exists(npy_file):
        print(f"Warning: {npy_file} not found, skipping...")
        return None
    
    print(f"\n--- Processing {test_name} (Lead II only) ---")
    
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
        print(f"Number of leads: {num_leads}")
        
        # Check if Lead II exists
        if TARGET_LEAD_INDEX >= num_leads:
            print(f"Error: Lead II (index {TARGET_LEAD_INDEX}) not available. Only {num_leads} leads found.")
            return None
        
        # Process Lead II only
        print(f"Processing Lead {TARGET_LEAD_NAME} (index {TARGET_LEAD_INDEX})...")
        
        ecg_lead_ii = ecg_all_leads[TARGET_LEAD_INDEX, :]
        
        # Process ECG with NeuroKit2
        try:
            signals, info = nk.ecg_process(ecg_lead_ii, sampling_rate=sampling_rate, method='neurokit')
            rpeaks = get_valid_indices(info, 'ECG_R_Peaks')
            
            if not rpeaks:
                print(f"Warning: No R-peaks detected in Lead {TARGET_LEAD_NAME}")
                return None
            
            print(f"Found {len(rpeaks)} R-peaks in Lead {TARGET_LEAD_NAME}")
            
            # Calculate comprehensive features
            intervals_amplitudes = calculate_intervals_and_amplitudes(signals, info, rpeaks, sampling_rate)
            hrv_metrics = calculate_heart_rate_variability(rpeaks, sampling_rate)
            
            # Combine all results for Lead II
            lead_results = {
                'lead_name': TARGET_LEAD_NAME,
                'lead_index': TARGET_LEAD_INDEX,
                'sampling_rate': sampling_rate,
                'signal_length_samples': len(ecg_lead_ii),
                'signal_duration_seconds': len(ecg_lead_ii) / sampling_rate,
                'num_rpeaks': len(rpeaks),
                'mean_heart_rate_bpm': 60 / (np.mean(np.diff(rpeaks)) / sampling_rate) if len(rpeaks) > 1 else None,
                **intervals_amplitudes,
                **hrv_metrics
            }
            
            # Generate plots for Lead II
            plot_ecg_analysis(ecg_lead_ii, signals, info, rpeaks, sampling_rate, 
                            output_dir, test_name)
            
            # Save results
            # Save as JSON
            results_file = os.path.join(output_dir, f"{test_name}_lead_II_features.json")
            with open(results_file, 'w') as f:
                json.dump(lead_results, f, indent=2)
            print(f"Saved features to: {results_file}")
            
            # Save summary report
            summary_file = os.path.join(output_dir, f"{test_name}_lead_II_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"ECG Analysis Report for {test_name} - Lead II Only\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Signal Duration: {lead_results.get('signal_duration_seconds', 'N/A'):.2f} seconds\n")
                f.write(f"Sampling Rate: {sampling_rate} Hz\n")
                f.write(f"Number of R-peaks: {lead_results.get('num_rpeaks', 'N/A')}\n")
                f.write(f"Mean Heart Rate: {lead_results.get('mean_heart_rate_bpm', 'N/A'):.1f} BPM\n\n")
                
                if 'RR_mean_ms' in lead_results:
                    f.write(f"Mean RR Interval: {lead_results['RR_mean_ms']:.1f} ± {lead_results.get('RR_std_ms', 0):.1f} ms\n")
                
                if 'QRS_mean_ms' in lead_results:
                    f.write(f"Mean QRS Duration: {lead_results['QRS_mean_ms']:.1f} ± {lead_results.get('QRS_std_ms', 0):.1f} ms\n")
                
                if 'PR_mean_ms' in lead_results:
                    f.write(f"Mean PR Interval: {lead_results['PR_mean_ms']:.1f} ± {lead_results.get('PR_std_ms', 0):.1f} ms\n")
                
                if 'QT_mean_ms' in lead_results:
                    f.write(f"Mean QT Interval: {lead_results['QT_mean_ms']:.1f} ± {lead_results.get('QT_std_ms', 0):.1f} ms\n")
                
                if 'QTc_mean' in lead_results:
                    f.write(f"Mean QTc: {lead_results['QTc_mean']:.1f} ± {lead_results.get('QTc_std', 0):.1f}\n")
                
                # Add amplitude information
                if 'Amplitudes_mV' in lead_results:
                    f.write(f"\nWave Amplitudes (mV):\n")
                    for wave_type in ['P', 'Q', 'R', 'S', 'T']:
                        mean_key = f'{wave_type}_peak_amp_mean'
                        std_key = f'{wave_type}_peak_amp_std'
                        if mean_key in lead_results['Amplitudes_mV']:
                            mean_amp = lead_results['Amplitudes_mV'][mean_key]
                            std_amp = lead_results['Amplitudes_mV'].get(std_key, 0)
                            f.write(f"  {wave_type}-wave: {mean_amp:.3f} ± {std_amp:.3f} mV\n")
            
            print(f"Saved summary to: {summary_file}")
            
            return {TARGET_LEAD_NAME: lead_results}
            
        except Exception as e:
            print(f"Error processing Lead {TARGET_LEAD_NAME}: {e}")
            return None
        
    except Exception as e:
        print(f"Error processing {test_name}: {e}")
        return None

def main():
    """Main function to process all test samples focusing on Lead II only."""
    
    # Configuration
    test_base_dir = "test"
    output_base_dir = "test_extracted_lead_II"
    sampling_rate = 100  # Updated sampling rate
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all test directories
    test_dirs = [d for d in os.listdir(test_base_dir) 
                 if os.path.isdir(os.path.join(test_base_dir, d)) and d.startswith('test')]
    test_dirs.sort()
    
    print(f"Found {len(test_dirs)} test directories: {test_dirs}")
    print(f"Processing Lead II only with sampling rate: {sampling_rate} Hz")
    
    # Process each test sample
    all_test_results = {}
    
    for test_dir_name in test_dirs:
        test_dir_path = os.path.join(test_base_dir, test_dir_name)
        results = process_test_sample(test_dir_path, output_base_dir, sampling_rate)
        
        if results:
            all_test_results[test_dir_name] = results
    
    # Save combined results
    combined_results_file = os.path.join(output_base_dir, "all_test_lead_II_features.json")
    with open(combined_results_file, 'w') as f:
        json.dump(all_test_results, f, indent=2)
    
    # Create a simplified feature matrix for Lead II only
    if all_test_results:
        feature_matrix = []
        test_names = []
        feature_names = []
        
        # Get feature names from first sample
        first_sample = next(iter(all_test_results.values()))
        if TARGET_LEAD_NAME in first_sample:
            feature_names = [k for k in first_sample[TARGET_LEAD_NAME].keys() 
                           if k not in ['lead_name', 'lead_index'] and 
                           not isinstance(first_sample[TARGET_LEAD_NAME][k], (list, dict))]
        
        # Create feature matrix
        for test_name, test_results in all_test_results.items():
            if TARGET_LEAD_NAME in test_results:
                row = []
                for feature_name in feature_names:
                    value = test_results[TARGET_LEAD_NAME].get(feature_name, 0.0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        row.append(value)
                    else:
                        row.append(0.0)
                feature_matrix.append(row)
                test_names.append(test_name)
        
        # Save as CSV
        if feature_matrix:
            feature_df = pd.DataFrame(feature_matrix, columns=feature_names, index=test_names)
            csv_file = os.path.join(output_base_dir, "test_lead_II_feature_matrix.csv")
            feature_df.to_csv(csv_file)
            print(f"Saved Lead II feature matrix to: {csv_file}")
            print(f"Feature matrix shape: {feature_df.shape}")
    
    print(f"\n--- Processing Complete ---")
    print(f"Processed {len(all_test_results)} test samples (Lead II only)")
    print(f"Combined results saved to: {combined_results_file}")
    print(f"Individual results and plots saved in: {output_base_dir}")

if __name__ == "__main__":
    main() 