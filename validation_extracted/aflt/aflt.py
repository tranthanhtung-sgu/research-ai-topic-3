import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import os # Added for path joining if needed

# --- Helper function to save plots ---
def save_plot(fig, filename_base, lead_name_sanitized_case, tight_layout=True, is_comparative=False): # Added is_comparative
    """Saves the given matplotlib figure and closes it."""
    if not fig.get_size_inches()[0] > 1 or not fig.get_size_inches()[1] > 1:
        fig.set_size_inches(12, 10 if is_comparative else 6) # Larger default for comparative
    if tight_layout:
        try:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        except Exception:
            pass
    filepath = f"{filename_base}_{lead_name_sanitized_case}.png"
    try:
        fig.savefig(filepath)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Failed to save plot {filepath}: {str(e)}")
    plt.close(fig)

def get_valid_indices(info_dict, key):
    indices = info_dict.get(key, [])
    if indices is None: return []
    if not hasattr(indices, '__iter__'): return []
    return [int(p) for p in indices if pd.notna(p) and isinstance(p, (int, float, np.number)) and p < 1e12]

# calculate_intervals_and_amplitudes remains the same as in your afib.py
def calculate_intervals_and_amplitudes(signals_df, info_dict, rpeaks_indices, sampling_rate):
    results = {}
    if not isinstance(signals_df, pd.DataFrame) or 'ECG_Clean' not in signals_df.columns:
        print("Error: signals_df is not a DataFrame or missing 'ECG_Clean'.")
        return results
    cleaned_ecg = signals_df['ECG_Clean'].values
    if len(cleaned_ecg) == 0:
        print("Error: Cleaned ECG signal is empty.")
        return results

    if len(rpeaks_indices) > 1:
        rr_samples = np.diff(rpeaks_indices); rr_ms = (rr_samples / sampling_rate) * 1000
        results['RR_intervals_ms'] = rr_ms
        if len(rr_ms) > 0: results['RR_mean_ms'] = np.mean(rr_ms); results['RR_std_ms'] = np.std(rr_ms)
    p_peaks = get_valid_indices(info_dict, 'ECG_P_Peaks')
    if len(p_peaks) > 1:
        pp_samples = np.diff(p_peaks); pp_ms = (pp_samples / sampling_rate) * 1000
        if len(pp_ms) > 0: results['PP_intervals_ms'] = pp_ms; results['PP_mean_ms'] = np.mean(pp_ms); results['PP_std_ms'] = np.std(pp_ms)
    p_onsets = get_valid_indices(info_dict, 'ECG_P_Onsets'); qrs_onsets_candidates = get_valid_indices(info_dict, 'ECG_R_Onsets')
    if not qrs_onsets_candidates: q_peaks = get_valid_indices(info_dict, 'ECG_Q_Peaks'); qrs_onsets_candidates = sorted(list(set(q_peaks))) if q_peaks else []
    pr_intervals_ms_list = []
    if p_onsets and qrs_onsets_candidates:
        p_onsets_sorted = sorted(list(set(p_onsets))); qrs_onsets_sorted = sorted(list(set(qrs_onsets_candidates)))
        for q_on in qrs_onsets_sorted:
            relevant_p_onsets_before_q = [p_on for p_on in p_onsets_sorted if p_on < q_on]
            if relevant_p_onsets_before_q:
                closest_p_onset = max(relevant_p_onsets_before_q)
                pr_duration_ms = (q_on - closest_p_onset) / sampling_rate * 1000
                if 80 <= pr_duration_ms <= 350: pr_intervals_ms_list.append(pr_duration_ms)
        if pr_intervals_ms_list: results['PR_intervals_ms'] = np.array(pr_intervals_ms_list); results['PR_mean_ms'] = np.mean(pr_intervals_ms_list); results['PR_std_ms'] = np.std(pr_intervals_ms_list)
    r_onsets = get_valid_indices(info_dict, 'ECG_R_Onsets'); s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_S_Offsets')
    if not s_offsets_or_r_offsets: s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_R_Offsets')
    qrs_durations_ms_list = []
    if r_onsets and s_offsets_or_r_offsets:
        r_onsets_sorted = sorted(list(set(r_onsets))); qrs_offsets_sorted = sorted(list(set(s_offsets_or_r_offsets)))
        used_offsets_indices = [False] * len(qrs_offsets_sorted)
        for r_on_val in r_onsets_sorted:
            best_qrs_off_val = -1; min_duration_diff = float('inf'); best_offset_idx = -1
            for i, r_off_val in enumerate(qrs_offsets_sorted):
                if not used_offsets_indices[i] and r_off_val > r_on_val:
                    duration_ms = (r_off_val - r_on_val) / sampling_rate * 1000
                    if 40 <= duration_ms <= 200 and (r_off_val - r_on_val) < min_duration_diff:
                        min_duration_diff = (r_off_val - r_on_val); best_qrs_off_val = r_off_val; best_offset_idx = i
            if best_qrs_off_val != -1 and best_offset_idx != -1: qrs_durations_ms_list.append(min_duration_diff / sampling_rate * 1000); used_offsets_indices[best_offset_idx] = True
        if qrs_durations_ms_list: results['QRS_durations_ms'] = np.array(qrs_durations_ms_list); results['QRS_mean_ms'] = np.mean(qrs_durations_ms_list); results['QRS_std_ms'] = np.std(qrs_durations_ms_list)
    t_offsets = get_valid_indices(info_dict, 'ECG_T_Offsets'); qt_intervals_ms_list = []; qtc_intervals_list = []
    if r_onsets and t_offsets and 'RR_intervals_ms' in results and len(results.get('RR_intervals_ms', [])) > 0:
        qrs_onsets_sorted = sorted(list(set(r_onsets))); t_offsets_sorted = sorted(list(set(t_offsets)))
        rr_intervals_sec = results['RR_intervals_ms'] / 1000.0; beat_idx = 0; used_t_offsets = [False] * len(t_offsets_sorted)
        for q_on_idx, q_on_val in enumerate(qrs_onsets_sorted):
            best_t_off_val = -1; min_qt_duration = float('inf'); best_t_off_idx = -1
            for i, t_off_val in enumerate(t_offsets_sorted):
                if not used_t_offsets[i] and t_off_val > q_on_val:
                    qt_duration_ms_candidate = (t_off_val - q_on_val) / sampling_rate * 1000
                    if 200 <= qt_duration_ms_candidate <= 700 and (t_off_val - q_on_val) < min_qt_duration:
                        min_qt_duration = (t_off_val - q_on_val); best_t_off_val = t_off_val; best_t_off_idx = i
            if best_t_off_val != -1 and best_t_off_idx != -1:
                qt_duration_ms = min_qt_duration / sampling_rate * 1000; qt_intervals_ms_list.append(qt_duration_ms); used_t_offsets[best_t_off_idx] = True
                r_peak_for_this_qrs = min([r for r in rpeaks_indices if r >= q_on_val], default=-1) if rpeaks_indices else -1
                rr_sec_for_this_beat = np.nan
                if r_peak_for_this_qrs != -1:
                    try: r_peak_list_idx = rpeaks_indices.index(r_peak_for_this_qrs)
                    except ValueError: r_peak_list_idx = -1 # Not found
                    if r_peak_list_idx > 0: rr_sec_for_this_beat = (rpeaks_indices[r_peak_list_idx] - rpeaks_indices[r_peak_list_idx-1]) / sampling_rate
                    elif q_on_idx < len(rr_intervals_sec): rr_sec_for_this_beat = rr_intervals_sec[q_on_idx]
                if pd.notna(rr_sec_for_this_beat) and rr_sec_for_this_beat > 0.1: qtc_intervals_list.append(qt_duration_ms / np.sqrt(rr_sec_for_this_beat))
            beat_idx +=1
        if qt_intervals_ms_list: results['QT_intervals_ms']=np.array(qt_intervals_ms_list); results['QT_mean_ms']=np.mean(qt_intervals_ms_list); results['QT_std_ms']=np.std(qt_intervals_ms_list)
        if qtc_intervals_list: results['QTc_intervals']=np.array(qtc_intervals_list); results['QTc_mean']=np.mean(qtc_intervals_list); results['QTc_std']=np.std(qtc_intervals_list)
    peak_keys_amplitudes = {'P':'ECG_P_Peaks','Q':'ECG_Q_Peaks','R':'ECG_R_Peaks','S':'ECG_S_Peaks','T':'ECG_T_Peaks'}
    results['Amplitudes_mV'] = {}
    for wave_name, peak_key in peak_keys_amplitudes.items():
        peaks = get_valid_indices(info_dict, peak_key)
        if peaks and len(peaks) > 0 and max(peaks) < len(cleaned_ecg):
            amplitudes = cleaned_ecg[peaks]
            if len(amplitudes)>0: results['Amplitudes_mV'][f'{wave_name}_peak_amps']=amplitudes; results['Amplitudes_mV'][f'{wave_name}_peak_amp_mean']=np.mean(amplitudes); results['Amplitudes_mV'][f'{wave_name}_peak_amp_std']=np.std(amplitudes)
    st_deviations_mv = []
    if r_onsets and s_offsets_or_r_offsets and p_onsets:
        j_points = sorted(list(set(s_offsets_or_r_offsets))); qrs_onsets_sorted = sorted(list(set(r_onsets))); p_onsets_sorted = sorted(list(set(p_onsets)))
        for j_point_val in j_points:
            current_q_on = max([qon for qon in qrs_onsets_sorted if qon < j_point_val], default=None)
            baseline_start_p_onset = max([pon for pon in p_onsets_sorted if pon < (current_q_on if current_q_on is not None else -1)], default=None)
            if current_q_on is not None and baseline_start_p_onset is not None and baseline_start_p_onset < current_q_on and (current_q_on - baseline_start_p_onset) > int(0.04 * sampling_rate):
                pr_segment_for_baseline = cleaned_ecg[baseline_start_p_onset:current_q_on]
                if len(pr_segment_for_baseline) > 0:
                    isoelectric_level = np.median(pr_segment_for_baseline)
                    st_measurement_point_sample = j_point_val + int(0.06 * sampling_rate)
                    if st_measurement_point_sample < len(cleaned_ecg): st_deviations_mv.append(cleaned_ecg[st_measurement_point_sample] - isoelectric_level)
        if st_deviations_mv: results['ST_deviations_mV']=np.array(st_deviations_mv); results['ST_mean_deviation_mV']=np.mean(st_deviations_mv); results['ST_std_deviation_mV']=np.std(st_deviations_mv)
    return results

# --- Modified Plotting Functions for Comparative Display ---

def plot_rr_interval_distribution_comparative(
    rpeaks_case, sampling_rate_case, lead_name_case,
    rpeaks_ref, sampling_rate_ref, lead_name_ref,
    lead_name_sanitized_case, _plot_filename_prefix_case,
    interpolate=False, detrend=None, interpolation_rate=100
):
    fig, (ax_case, ax_ref) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    fig.suptitle("Comparative RR Interval Distribution", fontsize=16, y=0.98)

    def _plot_single_rr_dist(ax, rpeaks, s_rate, lead_nm, label_suffix):
        if len(rpeaks) < 2:
            ax.text(0.5,0.5, f"Not enough R-peaks for {label_suffix}", ha='center', va='center')
            ax.set_title(f'RR Distribution - {lead_nm} ({label_suffix}) - No Data')
            return
        rri = np.diff(rpeaks) / s_rate * 1000
        # ... (rest of the processing logic from original plot_rr_interval_distribution)
        intervals_for_hist = rri # Simplified for this example, add interpolate/detrend if needed
        mean_rr, std_rr = np.mean(intervals_for_hist), np.std(intervals_for_hist)
        median_rr, min_rr, max_rr = np.median(intervals_for_hist), np.min(intervals_for_hist), np.max(intervals_for_hist)
        range_rr = max_rr - min_rr; bin_width = 25
        num_bins = int(np.ceil(range_rr / bin_width)) if range_rr > 0 else 10
        num_bins = max(10, min(num_bins, 30))
        ax.hist(intervals_for_hist, bins=num_bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=mean_rr, color='red', linestyle='--', linewidth=2, label=f'Mean RR: {mean_rr:.1f} ms')
        stats_text = (f"Mean: {mean_rr:.1f} ms\nMedian: {median_rr:.1f} ms\nStd Dev: {std_rr:.1f} ms\n"
                      f"Range: {min_rr:.1f}-{max_rr:.1f} ms\nN beats: {len(intervals_for_hist)}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'RR Interval Distribution - {lead_nm} ({label_suffix})')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('RR Interval [ms]')
        ax.grid(True, alpha=0.3); ax.legend()
    
    _plot_single_rr_dist(ax_case, rpeaks_case, sampling_rate_case, lead_name_case, "CASE")
    _plot_single_rr_dist(ax_ref, rpeaks_ref, sampling_rate_ref, lead_name_ref, "REFERENCE")
    save_plot(fig, f"{_plot_filename_prefix_case}_rr_distribution_comparative", lead_name_sanitized_case, is_comparative=True)


def plot_ecg_segments_comparative(
    ecg_cleaned_case, rpeaks_case, sampling_rate_case, lead_name_case,
    ecg_cleaned_ref, rpeaks_ref, sampling_rate_ref, lead_name_ref,
    lead_name_sanitized_case, _plot_filename_prefix_case
):
    # --- 1. Superimposed/Average Heartbeats ---
    fig_superimposed, (ax_case_s, ax_ref_s) = plt.subplots(2, 1, figsize=(12, 14), sharex=False)
    fig_superimposed.suptitle("Comparative Individual Average Heartbeat", fontsize=16, y=0.98)

    def _plot_single_superimposed(ax, ecg_signal, rpeaks, s_rate, lead_nm, label_suffix):
        if len(rpeaks) < 2:
            ax.text(0.5,0.5, f"Not enough R-peaks for {label_suffix}", ha='center', va='center')
            ax.set_title(f'Avg Heartbeat - {lead_nm} ({label_suffix}) - No Data')
            return
        segments = nk.ecg_segment(ecg_signal, rpeaks=rpeaks, sampling_rate=s_rate, show=False)
        if not segments: ax.text(0.5,0.5, "No segments", ha='center', va='center'); return
        
        segment_keys = sorted(list(segments.keys())); num_segments = len(segment_keys)
        colors = plt.cm.viridis(np.linspace(0, 0.8, min(num_segments, 30)))
        beat_lengths = [len(segments[key]['Signal'].values) for key in segment_keys if 'Signal' in segments[key].columns and segments[key] is not None]
        if not beat_lengths: ax.text(0.5,0.5, "No valid beat lengths", ha='center', va='center'); return
        
        median_length = int(np.median(beat_lengths))
        if median_length <=0 : ax.text(0.5,0.5, "Invalid median length", ha='center', va='center'); return
        rpeak_idx_local = median_length // 2
        time_axis = np.linspace(-rpeak_idx_local/s_rate, (median_length-rpeak_idx_local-1)/s_rate, median_length)
        resampled_beats_local = []
        max_beats_to_plot = min(30, num_segments)
        for i in range(max_beats_to_plot):
            key = segment_keys[i]
            if 'Signal' not in segments[key].columns: continue
            beat = segments[key]['Signal'].values
            if len(beat) > 10:
                resampled = np.interp(np.linspace(0,1,median_length), np.linspace(0,1,len(beat)), beat)
                resampled_beats_local.append(resampled)
                ax.plot(time_axis, resampled, color=colors[i % len(colors)], alpha=0.5, linewidth=1)
        avg_beat_local = None
        if resampled_beats_local:
            avg_beat_array = np.array(resampled_beats_local)
            if avg_beat_array.ndim == 2 and avg_beat_array.shape[0]>0:
                avg_beat_local = np.nanmean(avg_beat_array, axis=0)
                if not np.all(np.isnan(avg_beat_local)):
                    ax.plot(time_axis, avg_beat_local, color='red', linewidth=2.5, label='Average Beat') # Thinner for reference
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        avg_hr_str = ""
        if len(rpeaks) > 1:
            rr_ms = np.diff(rpeaks)/s_rate*1000
            if len(rr_ms)>0: avg_hr_str=f" (Avg HR: {60000/np.mean(rr_ms):.1f} BPM)"
        ax.set_title(f'Individual Avg Heartbeat - {lead_nm} ({label_suffix}){avg_hr_str}')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (s) relative to R-peak')
        ax.grid(True, alpha=0.3)
        if avg_beat_local is not None: ax.legend()

    _plot_single_superimposed(ax_case_s, ecg_cleaned_case, rpeaks_case, sampling_rate_case, lead_name_case, "CASE")
    _plot_single_superimposed(ax_ref_s, ecg_cleaned_ref, rpeaks_ref, sampling_rate_ref, lead_name_ref, "REFERENCE")
    save_plot(fig_superimposed, f"{_plot_filename_prefix_case}_segmented_heartbeats_comparative", lead_name_sanitized_case, is_comparative=True)


def generate_summary_report(detailed_params, info_dict, rpeaks_indices, sampling_rate):
    """Generate a concise summary report with specific ECG parameters."""
    report_lines = []
    
    # 1. Heart Rate Parameters
    heart_rate = 60000 / detailed_params.get('RR_mean_ms', 0) if detailed_params.get('RR_mean_ms', 0) > 0 else 0
    atrial_rate = 60000 / detailed_params.get('PP_mean_ms', 0) if detailed_params.get('PP_mean_ms', 0) > 0 else 0
    ventricular_rate = heart_rate  # Same as heart rate in most cases
    
    # 2. Rhythm
    rr_std = detailed_params.get('RR_std_ms', 0)
    pp_std = detailed_params.get('PP_std_ms', 0)
    rhythm_type = "regular" if rr_std < 50 else "irregular" if rr_std < 100 else "irregularly-irregular"
    
    # 3. P Wave Analysis
    p_peaks = get_valid_indices(info_dict, 'ECG_P_Peaks')
    r_peaks = get_valid_indices(info_dict, 'ECG_R_Peaks')
    p_wave_present = 1 if len(p_peaks) > 0 else 0
    p_qrs_ratio = len(p_peaks) / len(r_peaks) if len(r_peaks) > 0 else 0
    
    # Determine P wave morphology
    p_amplitudes = detailed_params.get('Amplitudes_mV', {}).get('P_peak_amps', [])
    p_wave_morphology = "normal"
    if len(p_amplitudes) > 0:
        if np.std(p_amplitudes) > 0.5:
            p_wave_morphology = "variable"
        elif np.mean(p_amplitudes) < 0:
            p_wave_morphology = "inverted"
    
    # 4. PR Interval
    pr_interval = detailed_params.get('PR_mean_ms', 0)
    pr_interval_valid = 1 if pr_interval > 0 and pr_interval < 350 else 0
    
    # 5. QRS Complex
    qrs_duration = detailed_params.get('QRS_mean_ms', 0)
    qrs_duration_class = "normal" if qrs_duration < 120 else "prolonged"
    
    # Determine QRS morphology
    r_amplitudes = detailed_params.get('Amplitudes_mV', {}).get('R_peak_amps', [])
    qrs_morphology = "normal"
    if len(r_amplitudes) > 0:
        if np.std(r_amplitudes) > 0.5:
            qrs_morphology = "variable"
    
    # 6. ST-T Changes
    st_deviations = detailed_params.get('ST_deviations_mV', [])
    st_t_discordance = 1 if len(st_deviations) > 0 and np.mean(st_deviations) < 0 else 0
    
    # Format the report
    report_lines.append("# ECG Analysis Summary Report\n")
    
    # 1. Heart Rate Parameters
    report_lines.append("## 1. Heart Rate Parameters")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"heart_rate\tnumeric\tBeats per minute\t{heart_rate:.1f}")
    report_lines.append(f"atrial_rate\tnumeric\tEstimate from P wave frequency\t{atrial_rate:.1f}")
    report_lines.append(f"ventricular_rate\tnumeric\tEstimate from QRS complex frequency\t{ventricular_rate:.1f}\n")
    
    # 2. Rhythm
    report_lines.append("## 2. Rhythm")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"rhythm_type\tcategory\tRegularity of rhythm\t{rhythm_type}")
    report_lines.append(f"rr_variability\tnumeric\tStd. deviation of R-R intervals\t{rr_std:.1f}")
    report_lines.append(f"pp_variability\tnumeric\tStd. deviation of P-P intervals\t{pp_std:.1f}\n")
    
    # 3. P Wave Analysis
    report_lines.append("## 3. P Wave Analysis")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"p_wave_present\tbinary\t1 if present, 0 if absent\t{p_wave_present}")
    report_lines.append(f"p_wave_morphology\tcategory\tWaveform characteristics\t{p_wave_morphology}")
    report_lines.append(f"p_qrs_ratio\tnumeric\tRatio of P waves to QRS complexes\t{p_qrs_ratio:.2f}\n")
    
    # 4. PR Interval
    report_lines.append("## 4. PR Interval")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"pr_interval_ms\tnumeric\tIn milliseconds\t{pr_interval:.1f}")
    report_lines.append(f"pr_interval_valid\tbinary\t1 if measurable, 0 if not\t{pr_interval_valid}\n")
    
    # 5. QRS Complex
    report_lines.append("## 5. QRS Complex")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"qrs_duration_ms\tnumeric\tIn milliseconds\t{qrs_duration:.1f}")
    report_lines.append(f"qrs_duration_class\tcategory\tDuration classification\t{qrs_duration_class}")
    report_lines.append(f"qrs_morphology\tcategory\tWaveform characteristics\t{qrs_morphology}\n")
    
    # 6. ST-T Changes
    report_lines.append("## 6. ST-T Changes")
    report_lines.append("Feature Name\tType\tDescription\tValue")
    report_lines.append(f"st_t_discordance\tbinary\t1 if ST/T opposite to QRS\t{st_t_discordance}")
    
    return "\n".join(report_lines)

# --- Main Orchestrator ---
def main_orchestrator():
    # --- CASE Condition (e.g., AFLT from validation02.npy) ---
    case_npy_file_path = "../validation/validation02/validation02.npy"
    case_condition_label_short = "AFLT" # For file prefix
    case_condition_label_full = "AFLT CASE" # For titles/reports
    case_sampling_rate = 100
    case_preferred_lead_idx = 1 # e.g. Lead II
    
    # --- NORMAL REFERENCE Condition (validation01.npy) ---
    normal_ref_npy_file_path = "../validation/validation01/validation01.npy"
    # normal_ref_condition_label_short = "NSR" # Not used for file prefix, case prefix is used
    normal_ref_condition_label_full = "NSR REFERENCE"
    normal_ref_sampling_rate = 100
    normal_ref_preferred_lead_idx = 1 # Use same lead index for fair comparison

    STANDARD_12_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # --- Load and Process CASE data ---
    print(f"--- Loading and Processing: {case_condition_label_full} from {case_npy_file_path} ---")
    try:
        ecg_all_leads_case = np.load(case_npy_file_path)
        num_leads_case = ecg_all_leads_case.shape[0] if ecg_all_leads_case.ndim > 1 else 1
        if ecg_all_leads_case.ndim == 1: ecg_all_leads_case = ecg_all_leads_case.reshape(1,-1)
        lead_idx_case = case_preferred_lead_idx if num_leads_case > case_preferred_lead_idx else 0
        ecg_lead_case = ecg_all_leads_case[lead_idx_case, :]
        signals_case, info_case = nk.ecg_process(ecg_lead_case, sampling_rate=case_sampling_rate, method='neurokit')
        rpeaks_case = get_valid_indices(info_case, 'ECG_R_Peaks')
        lead_name_case = f"Lead {STANDARD_12_LEAD_NAMES[lead_idx_case]}" if lead_idx_case < len(STANDARD_12_LEAD_NAMES) else f"Lead {lead_idx_case+1}"
        lead_name_sanitized_case = lead_name_case.replace(" ","_") + f"_idx{lead_idx_case}"
        if not rpeaks_case: raise ValueError(f"No R-peaks in CASE ({lead_name_case})")
    except Exception as e:
        print(f"FATAL ERROR processing CASE data: {e}"); return

    # --- Load and Process NORMAL REFERENCE data ---
    print(f"--- Loading and Processing: {normal_ref_condition_label_full} from {normal_ref_npy_file_path} ---")
    try:
        ecg_all_leads_ref = np.load(normal_ref_npy_file_path)
        num_leads_ref = ecg_all_leads_ref.shape[0] if ecg_all_leads_ref.ndim > 1 else 1
        if ecg_all_leads_ref.ndim == 1: ecg_all_leads_ref = ecg_all_leads_ref.reshape(1,-1)
        lead_idx_ref = normal_ref_preferred_lead_idx if num_leads_ref > normal_ref_preferred_lead_idx else 0
        ecg_lead_ref = ecg_all_leads_ref[lead_idx_ref, :]
        signals_ref, info_ref = nk.ecg_process(ecg_lead_ref, sampling_rate=normal_ref_sampling_rate, method='neurokit')
        rpeaks_ref = get_valid_indices(info_ref, 'ECG_R_Peaks')
        lead_name_ref = f"Lead {STANDARD_12_LEAD_NAMES[lead_idx_ref]}" if lead_idx_ref < len(STANDARD_12_LEAD_NAMES) else f"Lead {lead_idx_ref+1}"
        # lead_name_sanitized_ref is not needed for filenames as we use case's
        if not rpeaks_ref: raise ValueError(f"No R-peaks in REFERENCE ({lead_name_ref})")
    except Exception as e:
        print(f"FATAL ERROR processing REFERENCE data: {e}"); return

    # --- Generate Comparative Plots ---
    print("\n--- Generating Comparative Plots ---")
    
    plot_rr_interval_distribution_comparative(
        rpeaks_case, case_sampling_rate, lead_name_case,
        rpeaks_ref, normal_ref_sampling_rate, lead_name_ref,
        lead_name_sanitized_case, case_condition_label_short
    )

    plot_ecg_segments_comparative(
        signals_case['ECG_Clean'].values, rpeaks_case, case_sampling_rate, lead_name_case,
        signals_ref['ECG_Clean'].values, rpeaks_ref, normal_ref_sampling_rate, lead_name_ref,
        lead_name_sanitized_case, case_condition_label_short
    )
    
    # --- Generate Individual Reports (as before, but now the plots they refer to are comparative) ---
    # You might still want separate reports for detailed parameters of each.
    # For simplicity, I'll just run the reporting part for the CASE condition here,
    # noting that its plots are now comparative.
    
    report_lines = [] # Start a new report for the CASE condition, plots are comparative
    report_lines.insert(0, f"## Ultra Comprehensive ECG Analysis for {lead_name_case} ({case_condition_label_full}) ##")
    report_lines.append(f"## WITH COMPARATIVE PLOTS vs. {normal_ref_condition_label_full} ({lead_name_ref}) ##")
    report_lines.append(f"Analyzing Lead: {lead_name_case}\nSampling Rate: {case_sampling_rate} Hz\nSignal Duration: {len(ecg_lead_case)/case_sampling_rate:.2f} seconds\n" + "-"*30)
    
    # Add messages about the comparative plots generated above
    report_lines.append("\n### Comparative Plots Generated ###")
    report_lines.append(f"- RR Distribution: {case_condition_label_short}_rr_distribution_comparative_{lead_name_sanitized_case}.png")
    report_lines.append(f"- Segmented Heartbeats: {case_condition_label_short}_segmented_heartbeats_comparative_{lead_name_sanitized_case}.png")
    
    detailed_params_case = calculate_intervals_and_amplitudes(signals_case, info_case, rpeaks_case, case_sampling_rate)
    # ... (Sections A-H parameter reporting for CASE data, copied from your afib.py perform_full_analysis...)
    report_lines.append("\n### A. Fiducial Points Detection (CASE) ###")
    fiducial_keys = ['ECG_P_Onsets','ECG_P_Peaks','ECG_P_Offsets','ECG_Q_Peaks','ECG_R_Onsets','ECG_R_Peaks','ECG_R_Offsets','ECG_S_Peaks','ECG_T_Onsets','ECG_T_Peaks','ECG_T_Offsets']
    for key in fiducial_keys: report_lines.append(f"- {key.replace('ECG_', '')} detected: {len(get_valid_indices(info_case, key))}")
    report_lines.append("\n### B. Intervals and Durations (CASE) ###")
    interval_report_keys = {'RR Mean (ms)':('RR_mean_ms','RR_std_ms','RR_intervals_ms'),'PP Mean (ms)':('PP_mean_ms','PP_std_ms','PP_intervals_ms'),'PR Mean (ms)':('PR_mean_ms','PR_std_ms','PR_intervals_ms'),'QRS Mean (ms)':('QRS_mean_ms','QRS_std_ms','QRS_durations_ms'),'QT Mean (ms)':('QT_mean_ms','QT_std_ms','QT_intervals_ms'),'QTc Mean (Bazett)':('QTc_mean','QTc_std','QTc_intervals')}
    for disp_key, (mean_k,std_k,list_k) in interval_report_keys.items():
        mean_v,std_v,list_v = detailed_params_case.get(mean_k),detailed_params_case.get(std_k),detailed_params_case.get(list_k,[])
        if mean_v is not None and len(list_v)>0:
            std_display = f"{std_v:.2f}" if std_v is not None else "N/A"
            report_lines.append(f"- {disp_key}: {mean_v:.2f}{'ms' if 'ms' in disp_key else ''} (StdDev: {std_display}, N: {len(list_v)})")
        else:
            report_lines.append(f"- {disp_key}: Not reliably calculated.")
    report_lines.append("\n### C. Amplitudes (from Cleaned ECG - CASE) ###")
    if 'Amplitudes_mV' in detailed_params_case and detailed_params_case['Amplitudes_mV']:
        for amp_mean_k in detailed_params_case['Amplitudes_mV']:
            if '_amp_mean' in amp_mean_k:
                wave_n=amp_mean_k.split('_')[0]; mean_a=detailed_params_case['Amplitudes_mV'][amp_mean_k]
                std_a=detailed_params_case['Amplitudes_mV'].get(f'{wave_n}_peak_amp_std',np.nan)
                num_a=len(detailed_params_case['Amplitudes_mV'].get(f'{wave_n}_peak_amps',[]))
                std_display = f"{std_a:.3f}" if pd.notna(std_a) else "N/A"
                report_lines.append(f"- {wave_n} Peak Amp: Mean={mean_a:.3f}, Std={std_display} (N={num_a}) arb. units")
    else: report_lines.append("- Waveform amplitudes not calculated.")
    report_lines.append("\n### D. ST-Segment Analysis (Basic - CASE) ###")
    st_m,st_s,st_v = detailed_params_case.get('ST_mean_deviation_mV'),detailed_params_case.get('ST_std_deviation_mV'),detailed_params_case.get('ST_deviations_mV',[])
    if st_m is not None and len(st_v)>0:
        std_display = f"{st_s:.3f}" if st_s is not None else "N/A"
        report_lines.append(f"- ST Mean Dev (J+60ms): {st_m:.3f} arb. units (Std: {std_display}, N: {len(st_v)})")
    else: report_lines.append("- ST-Segment deviation: Not reliably calculated.")
    report_lines.append("\n### E. Axis Information (CASE - General Note) ###") 
    report_lines.append("- QRS Axis, P-wave axis, T-wave axis: Full calculation requires multiple specific leads (e.g., I & aVF). This report focuses on single-lead parameters derived from the chosen lead.")
    report_lines.append("\n### F. Heart Rate Variability (HRV) & Rate (CASE) ###")
    avg_hr_rr = detailed_params_case.get('RR_mean_ms')
    if avg_hr_rr and avg_hr_rr > 0: report_lines.append(f"- Avg HR (from RR): {60000/avg_hr_rr:.2f} bpm")
    else: report_lines.append("- Avg HR: Not calculated from RR.")
    if len(rpeaks_case) > 10:
        try:
            hrv_sum_case = nk.hrv(rpeaks_case, sampling_rate=case_sampling_rate, show=False)
            if not hrv_sum_case.empty:
                report_lines.append("  **HRV Metrics (nk.hrv() - CASE):** (Note: Freq-domain & non-linear metrics need longer, stable recordings)")
                for domain, cols in [("Time",['MeanNN','SDNN','RMSSD','pNN50']),("Freq",['LF','HF','LFHF']),("NonLin",['SD1','SD2','ApEn','SampEn'])]:
                    metrics_found = [f"{c}: {hrv_sum_case[c].iloc[0]:.3f}" for c in cols if c in hrv_sum_case.columns]
                    if metrics_found: report_lines.append(f"    {domain}-Domain: " + ", ".join(metrics_found))
                # Poincare plot is now comparative and generated above.
                report_lines.append(f"  (See Comparative Poincaré plot: {case_condition_label_short}_1_poincare_{lead_name_sanitized_case}.png if HRV params available for both conditions)")

            else: report_lines.append("  HRV summary (nk.hrv() - CASE) empty.")
        except Exception as e_hrv: report_lines.append(f"  HRV Analysis for CASE failed: {e_hrv}")
    else: report_lines.append("  Not enough R-peaks for full HRV (CASE - need >10).")

    report_lines.append("\n### G. Signal Quality (CASE) ###")
    if 'ECG_Quality' in signals_case.columns: report_lines.append(f"- Mean ECG Quality: {signals_case['ECG_Quality'].mean():.3f} (NeuroKit method)")
    else: report_lines.append("- ECG Quality metric not available.")
    
    report_lines.append("\n### H. Cardiac Phase Information (CASE - General Note) ###") 
    report_lines.append("- Cardiac phase plotting is not included in this comparative version for brevity. See NeuroKit examples if specific phase plots are needed.")

    report_lines.append("\n### I. Representative Heartbeats (CASE - Superimposed) ###")
    report_lines.append(f"- See comparative plot: {case_condition_label_short}_segmented_heartbeats_comparative_{lead_name_sanitized_case}.png (shows superimposed beats and average for both conditions).")

    # Clinical Summary Section (for CASE)
    report_lines.append("\n" + "="*30 + f"\n### Clinical Summary Suggestion ({case_condition_label_full}) ###")
    if case_condition_label_short == "AFLT":
        report_lines.append("Key AFLT indicators to check (Compare CASE vs REFERENCE in plots):")
        report_lines.append("  - Atrial Rate: Regular atrial activity at ~300 bpm")
        report_lines.append("  - P Wave Morphology: Sawtooth pattern (F waves)")
        report_lines.append("  - AV Conduction: Usually 2:1 or 4:1 block")
        report_lines.append("  - QRS Complex: Usually normal unless pre-existing conduction abnormality")
        report_lines.append("  - Rhythm: Regular ventricular response if fixed AV block")
    # Add more conditions if needed
    else: # Generic summary
        report_lines.append("Review parameters from Sections A-H and compare visually with REFERENCE in plots.")

    report_lines.append("Please correlate with full clinical picture. This is not a diagnostic tool.")
    report_lines.append("="*30)

    report_filename_case = f"ultra_comprehensive_ecg_report_COMPARATIVE_{case_condition_label_short}_{lead_name_sanitized_case}.txt"
    with open(report_filename_case, "w") as f: f.write("\n".join(report_lines))
    print(f"\n--- Report for {case_condition_label_full} (with comparative plot references) ---")
    print(f"Full report saved to: {report_filename_case}")
    print("\nAll comparative analyses complete. Check generated .png plot files and .txt report file.")

    # Generate the summary report for the case
    summary_report = generate_summary_report(detailed_params_case, info_case, rpeaks_case, case_sampling_rate)
    
    # Save the summary report
    summary_filename = "report_filename_case.txt"
    with open(summary_filename, "w") as f:
        f.write(summary_report)
    print(f"\nSummary report saved to: {summary_filename}")


if __name__ == "__main__":
    main_orchestrator() 