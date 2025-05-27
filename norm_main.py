# import numpy as np
# import neurokit2 as nk
# import matplotlib.pyplot as plt
# import glob
# import os
# import pandas as pd

# # --- Helper function to save plots ---
# def save_plot(fig, filename_base, lead_name_sanitized):
#     """Saves the given matplotlib figure and closes it."""
#     if not fig.get_size_inches()[0] > 1 or not fig.get_size_inches()[1] > 1:
#         fig.set_size_inches(10, 6)
#     filepath = f"{filename_base}_{lead_name_sanitized}.png"
#     try:
#         fig.savefig(filepath)
#         print(f"Saved plot: {filepath}")
#     except Exception as e:
#         print(f"Failed to save plot {filepath}: {str(e)}")
#     plt.close(fig)

# def calculate_pr_intervals(p_onsets, r_onsets, sampling_rate):
#     """
#     Calculates PR intervals by pairing P-onsets with subsequent R-onsets.
#     Returns a list of PR intervals in milliseconds.
#     """
#     pr_intervals_ms = []
#     if not p_onsets or not r_onsets:
#         return pr_intervals_ms

#     p_onsets_valid = sorted([p for p in p_onsets if not np.isnan(p)])
#     r_onsets_valid = sorted([r for r in r_onsets if not np.isnan(r)])

#     p_idx = 0
#     for r_onset_val in r_onsets_valid:
#         # Find the latest P-onset that occurs before the current R-onset
#         # and is within a reasonable physiological window (e.g., PR < 300ms)
#         candidate_p_onsets = [
#             p_onset_val for p_onset_val in p_onsets_valid
#             if p_onset_val < r_onset_val and \
#                ((r_onset_val - p_onset_val) / sampling_rate * 1000) < 300 and \
#                ((r_onset_val - p_onset_val) / sampling_rate * 1000) > 50 # Min PR, e.g. 50ms
#         ]
        
#         if candidate_p_onsets:
#             # Choose the P-onset closest to (but before) the R-onset
#             best_p_onset = max(candidate_p_onsets)
#             pr_interval_samples = r_onset_val - best_p_onset
#             pr_intervals_ms.append((pr_interval_samples / sampling_rate) * 1000)
#             # Try to advance p_idx to avoid reusing the same P-onset for multiple R-onsets
#             # This is a simple heuristic; more complex logic might be needed for very noisy signals
#             try:
#                 p_onsets_valid = [p for p in p_onsets_valid if p > best_p_onset]
#             except: # if p_onsets_valid becomes empty
#                 pass

#     return pr_intervals_ms


# def main_nsr_analysis():
#     # Step 1: Load the .npy file - CHANGED FOR NSR
#     npy_file_path = "/home/tony/neurokit/validation/validation01/validation01.npy"

#     # --- Configuration ---
#     sampling_rate = 100 # Hz
#     preferred_lead_idx = 1 # Try Lead II
#     fallback_lead_idx = 0  # Fallback to Lead I

#     report_lines = []

#     try:
#         ecg_signal_all_leads = np.load(npy_file_path)
#         report_lines.append(f"Successfully loaded ECG data from: {npy_file_path}")
#     except FileNotFoundError:
#         report_lines.append(f"Error: File not found at {npy_file_path}.")
#         # Dummy data for testing if file not found
#         print("Creating a dummy NSR-like signal for testing as file not found.")
#         sampling_rate_dummy = 100
#         duration_dummy = 10
#         ecg_signal_all_leads = np.array([nk.ecg_simulate(duration=duration_dummy, sampling_rate=sampling_rate_dummy, heart_rate=75, random_state=42)]*12)

#     except Exception as e:
#         report_lines.append(f"Error loading .npy file: {str(e)}")
#         print("\n".join(report_lines))
#         exit(1)

#     report_lines.append(f"ECG Signal Shape: {ecg_signal_all_leads.shape}")

#     num_leads = ecg_signal_all_leads.shape[0]
#     if num_leads > preferred_lead_idx:
#         lead_to_analyze_idx = preferred_lead_idx
#         lead_name = f"II (index {lead_to_analyze_idx})"
#     elif num_leads > fallback_lead_idx:
#         lead_to_analyze_idx = fallback_lead_idx
#         lead_name = f"I (index {lead_to_analyze_idx})"
#     elif num_leads > 0:
#         lead_to_analyze_idx = 0
#         lead_name = f"Lead {lead_to_analyze_idx+1}"
#     else:
#         report_lines.append("Error: No leads found.")
#         print("\n".join(report_lines))
#         exit(1)

#     ecg_lead_signal = ecg_signal_all_leads[lead_to_analyze_idx, :]
#     lead_name_sanitized = lead_name.replace(" ", "_").replace("(", "").replace(")", "")

#     # CHANGED REPORT TITLE
#     report_lines.insert(0, f"## Normal Sinus Rhythm (NSR) Analysis for ECG Lead {lead_name} ##")
#     report_lines.append(f"Analyzing Lead: {lead_name}")
#     report_lines.append(f"Sampling Rate: {sampling_rate} Hz")
#     report_lines.append(f"Signal Duration: {len(ecg_lead_signal)/sampling_rate:.2f} seconds")
#     report_lines.append("-" * 30)

#     try:
#         print(f"\nProcessing Lead {lead_name} for NSR analysis...")
#         signals, info = nk.ecg_process(ecg_lead_signal, sampling_rate=sampling_rate, method='neurokit')

#         fig_ecg_plot = nk.ecg_plot(signals, info)
#         if not isinstance(fig_ecg_plot, plt.Figure):
#             fig_ecg_plot = plt.gcf()
#         fig_ecg_plot.suptitle(f"Processed ECG for Lead {lead_name} (NSR Analysis)", y=1)
#         save_plot(fig_ecg_plot, "nsr_processed_ecg", lead_name_sanitized) # CHANGED FILENAME

#         rpeaks_indices = info['ECG_R_Peaks']
#         report_lines.append("\n### 1. Rate (60-100 bpm) ###")
#         if len(rpeaks_indices) > 1:
#             mean_rr_ms = np.mean(np.diff(rpeaks_indices)) / sampling_rate * 1000
#             heart_rate_bpm = 60000 / mean_rr_ms if mean_rr_ms > 0 else 0
#             report_lines.append(f"- Average Heart Rate: {heart_rate_bpm:.2f} bpm (calculated from mean R-R interval).")
#             if 60 <= heart_rate_bpm <= 100:
#                 report_lines.append("  Observation: Heart rate is within the normal range for NSR (60-100 bpm).")
#             else:
#                 report_lines.append("  Observation: Heart rate is outside the typical NSR range (60-100 bpm).")
#         else:
#             report_lines.append("- Not enough R-peaks to calculate average heart rate.")


#         report_lines.append("\n### 2. Rhythm (Regular: consistent P-P and R-R intervals) ###")
#         if len(rpeaks_indices) < 5:
#             report_lines.append("- Not enough R-peaks to reliably assess rhythm regularity.")
#         else:
#             rr_intervals_ms = np.diff(rpeaks_indices) / sampling_rate * 1000

#             fig_tachogram, ax_tachogram = plt.subplots(figsize=(12, 4))
#             ax_tachogram.plot(rr_intervals_ms, marker='o', linestyle='-')
#             ax_tachogram.set_title(f"Tachogram (R-R Intervals) - Lead {lead_name}")
#             ax_tachogram.set_xlabel("Beat Number")
#             ax_tachogram.set_ylabel("R-R Interval (ms)")
#             save_plot(fig_tachogram, "nsr_tachogram", lead_name_sanitized) # CHANGED FILENAME
#             report_lines.append("- Tachogram: Shows beat-to-beat R-R interval variation. For NSR, expect relatively consistent R-R intervals.")
#             report_lines.append(f"  (See nsr_tachogram_{lead_name_sanitized}.png)")

#             try:
#                 hrv_analysis_results = nk.hrv(rpeaks_indices, sampling_rate=sampling_rate, show=False)

#                 fig_poincare, ax_poincare = plt.subplots(figsize=(6,6))
#                 rr_n = rr_intervals_ms[:-1]
#                 rr_n_plus_1 = rr_intervals_ms[1:]
#                 if len(rr_n) > 0 : # Ensure there's data to plot
#                     ax_poincare.scatter(rr_n, rr_n_plus_1, c='blue', alpha=0.75)
#                     min_val = min(np.min(rr_n), np.min(rr_n_plus_1))
#                     max_val = max(np.max(rr_n), np.max(rr_n_plus_1))
#                     ax_poincare.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

#                     if 'HRV_SD1' in hrv_analysis_results.columns and 'HRV_SD2' in hrv_analysis_results.columns:
#                         sd1 = hrv_analysis_results['HRV_SD1'].iloc[0]
#                         sd2 = hrv_analysis_results['HRV_SD2'].iloc[0]
#                         ax_poincare.text(0.05, 0.9, f'SD1: {sd1:.2f} ms', transform=ax_poincare.transAxes)
#                         ax_poincare.text(0.05, 0.85, f'SD2: {sd2:.2f} ms', transform=ax_poincare.transAxes)
                
#                 ax_poincare.set_title(f"Poincaré Plot - Lead {lead_name}")
#                 ax_poincare.set_xlabel("RR_n interval (ms)")
#                 ax_poincare.set_ylabel("RR_n+1 interval (ms)")
#                 ax_poincare.grid(True, alpha=0.3)
#                 save_plot(fig_poincare, "nsr_poincare", lead_name_sanitized) # CHANGED FILENAME
#                 report_lines.append("- Poincaré Plot: Visualizes R-R interval correlation. For NSR, expect a tight cluster along the line of identity.")
#                 report_lines.append(f"  (See nsr_poincare_{lead_name_sanitized}.png)")

#                 report_lines.append("\n  Key HRV Metrics for Regularity (expect lower values for NSR):")
#                 hrv_metrics_to_report = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_CVNN', 'HRV_SD1', 'HRV_SD2']
#                 for metric in hrv_metrics_to_report:
#                     if metric in hrv_analysis_results.columns:
#                         value = hrv_analysis_results[metric].iloc[0]
#                         report_lines.append(f"    {metric}: {value:.2f}")
#                 report_lines.append("  Interpretation: Lower SDNN, RMSSD, pNN50, CVNN, SD1, SD2 suggest more regular rhythm.")
#             except Exception as e_hrv:
#                 report_lines.append(f"- HRV Analysis (including Poincaré) failed or incomplete: {str(e_hrv)}")
#                 print(f"HRV analysis/Poincaré plot failed: {str(e_hrv)}")
        
#         # P-P interval regularity (basic check)
#         p_peaks_indices_valid = [p for p in info.get('ECG_P_Peaks', []) if not np.isnan(p)]
#         if len(p_peaks_indices_valid) > 2:
#             pp_intervals_ms = np.diff(p_peaks_indices_valid) / sampling_rate * 1000
#             mean_pp = np.mean(pp_intervals_ms)
#             std_pp = np.std(pp_intervals_ms)
#             report_lines.append(f"- P-P Intervals: Mean={mean_pp:.2f} ms, StdDev={std_pp:.2f} ms. Low StdDev suggests regular atrial activity.")
#         else:
#             report_lines.append("- P-P Intervals: Not enough P-peaks detected for P-P interval regularity analysis.")


#         report_lines.append("\n### 3. P waves (Uniform morphology, upright in I, II, aVF; one P wave precedes each QRS) ###")
#         num_r_peaks = len(rpeaks_indices)
#         p_peaks_indices = [p for p in info.get('ECG_P_Peaks', []) if not np.isnan(p)]
#         num_p_peaks_detected = len(p_peaks_indices)

#         report_lines.append(f"- Number of R-peaks found: {num_r_peaks}")
#         report_lines.append(f"- Number of P-peaks detected by NeuroKit: {num_p_peaks_detected}")
        
#         if num_r_peaks > 0 and abs(num_p_peaks_detected - num_r_peaks) <= max(1, 0.1 * num_r_peaks) : # Allow for slight misdetection
#              report_lines.append("- Observation: P-peaks are consistently detected before QRS complexes (approx 1:1 ratio). This is consistent with NSR.")
#         else:
#              report_lines.append("- Observation: Number of P-peaks significantly differs from R-peaks. This might indicate issues with P-wave detection or an arrhythmia. Visual inspection is crucial.")
#         report_lines.append("  Guidance: For NSR, expect a clear P-wave before each QRS. Morphology should be uniform (visual check).")
#         # P-wave axis (upright/inverted) is a multi-lead assessment, not covered here with single lead focus.

#         fig_pwave_detail, ax_pwave_detail = plt.subplots(figsize=(15, 6))
#         segment_len_sec = min(5, len(ecg_lead_signal) / sampling_rate - 0.1)
#         plot_end_sample = int(min(len(signals['ECG_Clean']), segment_len_sec * sampling_rate))
#         time_axis_segment = np.arange(plot_end_sample) / sampling_rate

#         ax_pwave_detail.plot(time_axis_segment, signals['ECG_Clean'].iloc[:plot_end_sample], label="Cleaned ECG")
        
#         p_peaks_plot = [p for p in p_peaks_indices if p < plot_end_sample]
#         if p_peaks_plot:
#             ax_pwave_detail.scatter(np.array(p_peaks_plot)/sampling_rate, signals['ECG_Clean'].iloc[p_peaks_plot], 
#                                 color='red', marker='P', s=100, label="Detected P-peaks (NK)", zorder=5)
        
#         r_peaks_plot = [r for r in rpeaks_indices if r < plot_end_sample]
#         if r_peaks_plot:
#              ax_pwave_detail.scatter(np.array(r_peaks_plot)/sampling_rate, signals['ECG_Clean'].iloc[r_peaks_plot],
#                                 color='blue', marker='^', s=100, label="R-peaks", zorder=5)

#         ax_pwave_detail.set_title(f"ECG Segment (Lead {lead_name}) - Detail for P-wave assessment (NSR)")
#         ax_pwave_detail.set_xlabel("Time (s)")
#         ax_pwave_detail.set_ylabel("Amplitude")
#         ax_pwave_detail.legend()
#         ax_pwave_detail.grid(True, linestyle=':', alpha=0.7)
#         save_plot(fig_pwave_detail, "nsr_pwave_detail", lead_name_sanitized) # CHANGED FILENAME
#         report_lines.append(f"  (See nsr_pwave_detail_{lead_name_sanitized}.png for visual inspection of P-waves).")

#         report_lines.append("\n### 4. P:QRS Ratio (1:1) ###")
#         if num_r_peaks > 0:
#             ratio = num_p_peaks_detected / num_r_peaks if num_r_peaks > 0 else 0
#             report_lines.append(f"- Observed P:R-peak ratio: {num_p_peaks_detected}:{num_r_peaks} (approx {ratio:.2f}:1).")
#             if 0.9 <= ratio <= 1.1: # Allowing some tolerance for detection issues
#                 report_lines.append("  Observation: P:QRS ratio is approximately 1:1, consistent with NSR.")
#             else:
#                 report_lines.append("  Observation: P:QRS ratio deviates from 1:1. Further investigation or P-wave detection tuning may be needed.")
#         else:
#             report_lines.append("- Observation: No R-peaks detected, cannot determine P:QRS ratio.")


#         report_lines.append("\n### 5. PR Interval (Constant, 0.12-0.20 seconds) ###")
#         p_onsets_raw = info.get('ECG_P_Onsets', [])
#         r_onsets_raw = info.get('ECG_R_Onsets', []) # Or use ECG_Q_Peaks if R_Onsets are not reliable
        
#         pr_intervals_calculated_ms = calculate_pr_intervals(p_onsets_raw, r_onsets_raw, sampling_rate)

#         if pr_intervals_calculated_ms:
#             avg_pr = np.nanmean(pr_intervals_calculated_ms)
#             std_pr = np.nanstd(pr_intervals_calculated_ms)
#             report_lines.append(f"- Calculated PR Intervals: Average={avg_pr:.2f} ms, StdDev={std_pr:.2f} ms (from {len(pr_intervals_calculated_ms)} intervals).")
#             if 120 <= avg_pr <= 200:
#                 report_lines.append("  Observation: Average PR interval is within the normal range (120-200 ms).")
#             else:
#                 report_lines.append(f"  Observation: Average PR interval ({avg_pr:.2f} ms) is outside the normal range (120-200 ms).")
#             if std_pr < 20: # Threshold for "constant" PR interval, can be adjusted
#                 report_lines.append("  Observation: PR intervals appear relatively constant (StdDev < 20 ms).")
#             else:
#                 report_lines.append("  Observation: PR intervals show some variability (StdDev >= 20 ms).")
#         else:
#             report_lines.append("- Observation: PR intervals could not be reliably calculated (insufficient/inconsistent P-onsets or R-onsets).")


#         report_lines.append("\n### 6. QRS Duration (Narrow, typically <0.10-0.12 seconds) ###")
#         avg_qrs_duration = np.nan
#         r_onsets_calc = [x for x in info.get('ECG_R_Onsets', []) if not np.isnan(x)]
#         r_offsets_calc = [x for x in info.get('ECG_R_Offsets', []) if not np.isnan(x)]

#         qrs_durations_ms_list = []
#         if r_onsets_calc and r_offsets_calc:
#             # Pair QRS onsets and offsets
#             # A simple pairing, assuming they are somewhat ordered and correspond
#             idx_offset = 0
#             for r_on in r_onsets_calc:
#                 # Find the first r_offset after r_on
#                 found_offset = False
#                 for i in range(idx_offset, len(r_offsets_calc)):
#                     if r_offsets_calc[i] > r_on and (r_offsets_calc[i] - r_on)/sampling_rate*1000 < 150 : # Max QRS reasonable duration
#                         qrs_durations_ms_list.append(((r_offsets_calc[i] - r_on) / sampling_rate) * 1000)
#                         idx_offset = i + 1 # Start search for next offset from here
#                         found_offset = True
#                         break
#                 if not found_offset: # If no suitable offset found for an onset
#                     pass # Could log this or handle differently

#         if qrs_durations_ms_list:
#             avg_qrs_duration = np.nanmean(qrs_durations_ms_list)
#             report_lines.append(f"- Average QRS duration (from R-onsets/offsets): {avg_qrs_duration:.2f} ms (from {len(qrs_durations_ms_list)} complexes).")
#             # NSR typically <0.10s (100ms), but up to 0.12s (120ms) can be normal
#             if avg_qrs_duration < 120:
#                 report_lines.append("  Observation: QRS duration is narrow (<120ms), consistent with NSR.")
#             else:
#                 report_lines.append(f"  Observation: QRS duration ({avg_qrs_duration:.2f}ms) is wider than typical for uncomplicated NSR.")

#             fig_qrs, ax_qrs = plt.subplots(figsize=(8, 5))
#             ax_qrs.hist(qrs_durations_ms_list, bins=max(5, min(20, len(qrs_durations_ms_list)//2 if len(qrs_durations_ms_list)>10 else 5)), edgecolor='black') # Dynamic bins
#             ax_qrs.set_title(f"Distribution of QRS Durations - Lead {lead_name}")
#             ax_qrs.set_xlabel("QRS Duration (ms)")
#             ax_qrs.set_ylabel("Frequency")
#             save_plot(fig_qrs, "nsr_qrs_duration_hist", lead_name_sanitized) # CHANGED FILENAME
#             report_lines.append(f"  (See nsr_qrs_duration_hist_{lead_name_sanitized}.png).")
#         else:
#             report_lines.append("- QRS durations could not be calculated (insufficient R-onsets/offsets or pairing issues).")
        
#         report_lines.append("\n" + "="*30)
#         report_lines.append("### Clinical Summary Suggestion (NSR) ###")
#         report_lines.append(f"The analysis of ECG Lead {lead_name} demonstrates features consistent with Normal Sinus Rhythm:")
        
#         rate_check = "within normal limits (60-100 bpm)" if 'heart_rate_bpm' in locals() and 60 <= heart_rate_bpm <= 100 else "outside normal limits or unassessable"
#         report_lines.append(f"1. **Rate**: Average heart rate appears {rate_check}.")
        
#         rhythm_check = "regular" if 'hrv_analysis_results' in locals() and hrv_analysis_results['HRV_SDNN'].iloc[0] < 50 else "showing some variability or unassessable" # Example threshold
#         report_lines.append(f"2. **Rhythm**: Appears generally {rhythm_check}, with relatively consistent R-R intervals noted on tachogram and a clustered Poincaré plot.")
        
#         pwave_check = "present and consistently precede QRS complexes (approx 1:1)" if 'num_r_peaks' in locals() and num_r_peaks > 0 and abs(num_p_peaks_detected - num_r_peaks) <= max(1, 0.1 * num_r_peaks) else "inconsistent or P:QRS ratio not 1:1"
#         report_lines.append(f"3. **P-waves**: Appear {pwave_check}. Visual inspection for uniform morphology is recommended.")
        
#         pr_check = "within normal limits (120-200 ms) and constant" if 'avg_pr' in locals() and 120 <= avg_pr <= 200 and 'std_pr' in locals() and std_pr < 20 else "outside normal limits, variable, or unassessable"
#         report_lines.append(f"4. **PR Interval**: Appears {pr_check}.")
        
#         qrs_dur_check = "narrow (<120ms)" if not np.isnan(avg_qrs_duration) and avg_qrs_duration < 120 else "wide or unassessable"
#         report_lines.append(f"5. **QRS Duration**: Appears {qrs_dur_check}.")
        
#         report_lines.append("\nThese findings support an interpretation of Normal Sinus Rhythm. Correlate with full clinical picture and other leads if available.")
#         report_lines.append("="*30)

#     except Exception as e:
#         error_msg = f"Critical Error during processing Lead {lead_name} for NSR analysis: {str(e)}"
#         print(error_msg)
#         report_lines.append(error_msg)
#         import traceback
#         traceback.print_exc()
#         try:
#             cleaned_signal = nk.ecg_clean(ecg_lead_signal, sampling_rate=sampling_rate)
#             fig_cleaned, ax_cleaned = plt.subplots(figsize=(12,4))
#             ax_cleaned.plot(cleaned_signal)
#             ax_cleaned.set_title(f"Cleaned ECG Signal (Processing Failed) - Lead {lead_name}")
#             save_plot(fig_cleaned, "nsr_cleaned_ecg_fallback", lead_name_sanitized) # CHANGED FILENAME
#         except Exception as e2:
#             print(f"Error during fallback cleaning of Lead {lead_name}: {str(e2)}")

#     report_filename = f"nsr_report_{lead_name_sanitized}.txt" # CHANGED FILENAME
#     with open(report_filename, "w") as f:
#         f.write("\n".join(report_lines))
    
#     print("\n\n" + "="*10 + " FINAL NSR REPORT " + "="*10) # CHANGED
#     print("\n".join(report_lines))
#     print(f"\nFull report saved to: {report_filename}")
#     print(f"Associated plots saved in the current directory with prefix 'nsr_' and lead '{lead_name_sanitized}'.")

# if __name__ == "__main__":
#     main_nsr_analysis()

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- Helper function to save plots ---
def save_plot(fig, filename_base, lead_name_sanitized, tight_layout=True):
    """Saves the given matplotlib figure and closes it."""
    if not fig.get_size_inches()[0] > 1 or not fig.get_size_inches()[1] > 1:
        fig.set_size_inches(10, 6)
    if tight_layout:
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.96]) 
        except: 
            pass
    filepath = f"{filename_base}_{lead_name_sanitized}.png"
    try:
        fig.savefig(filepath)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Failed to save plot {filepath}: {str(e)}")
    plt.close(fig)

def get_valid_indices(info_dict, key):
    """Safely gets valid (non-NaN, integer) indices from the info dictionary."""
    indices = info_dict.get(key, [])
    return [int(p) for p in indices if pd.notna(p) and isinstance(p, (int, float, np.number)) and p < 1e9] 

def calculate_intervals_and_amplitudes(signals_df, info_dict, rpeaks_indices, sampling_rate):
    results = {}
    if not isinstance(signals_df, pd.DataFrame) or 'ECG_Clean' not in signals_df.columns:
        print("Error: signals_df is not a DataFrame or missing 'ECG_Clean'.")
        return results
    cleaned_ecg = signals_df['ECG_Clean'].values
    if len(cleaned_ecg) == 0:
        print("Error: Cleaned ECG signal is empty.")
        return results

    # --- RR Intervals (already in hrv_time, but good to have raw for plotting) ---
    if len(rpeaks_indices) > 1:
        rr_samples = np.diff(rpeaks_indices)
        rr_ms = (rr_samples / sampling_rate) * 1000
        results['RR_intervals_ms'] = rr_ms # For plotting or direct use
        # Mean, min, max, std will be covered by HRV metrics

    # --- PP Intervals ---
    p_peaks = get_valid_indices(info_dict, 'ECG_P_Peaks')
    if len(p_peaks) > 1:
        pp_samples = np.diff(p_peaks)
        pp_ms = (pp_samples / sampling_rate) * 1000
        if len(pp_ms) > 0:
            results['PP_intervals_ms'] = pp_ms
            results['PP_mean_ms'] = np.mean(pp_ms)
            results['PP_std_ms'] = np.std(pp_ms)

    # --- PR Intervals ---
    p_onsets = get_valid_indices(info_dict, 'ECG_P_Onsets')
    qrs_onsets = get_valid_indices(info_dict, 'ECG_R_Onsets') 
    pr_intervals_ms_list = []
    if p_onsets and qrs_onsets:
        p_onsets_sorted = sorted(list(set(p_onsets))) 
        qrs_onsets_sorted = sorted(list(set(qrs_onsets)))
        
        for q_on in qrs_onsets_sorted:
            relevant_p_onset = None
            temp_p_onsets_before_q = [p for p in p_onsets_sorted if p < q_on]
            if temp_p_onsets_before_q:
                relevant_p_onset = max(temp_p_onsets_before_q)
            
            if relevant_p_onset is not None:
                pr_duration_samples = q_on - relevant_p_onset
                pr_duration_ms = (pr_duration_samples / sampling_rate) * 1000
                if 80 <= pr_duration_ms <= 350: 
                    pr_intervals_ms_list.append(pr_duration_ms)
        if pr_intervals_ms_list:
            results['PR_intervals_ms'] = np.array(pr_intervals_ms_list)
            results['PR_mean_ms'] = np.mean(pr_intervals_ms_list)
            results['PR_std_ms'] = np.std(pr_intervals_ms_list)

    # --- QRS Durations ---
    r_onsets = get_valid_indices(info_dict, 'ECG_R_Onsets')
    r_offsets = get_valid_indices(info_dict, 'ECG_R_Offsets') 
    qrs_durations_ms_list = []
    if r_onsets and r_offsets:
        r_onsets_sorted = sorted(list(set(r_onsets)))
        r_offsets_sorted = sorted(list(set(r_offsets)))
        
        used_offsets_indices = [False] * len(r_offsets_sorted)
        for r_on in r_onsets_sorted:
            best_r_off_val = -1; min_duration_diff = float('inf'); best_offset_idx = -1
            for i, r_off_val in enumerate(r_offsets_sorted):
                if not used_offsets_indices[i] and r_off_val > r_on:
                    duration_samples = r_off_val - r_on
                    duration_ms = (duration_samples / sampling_rate) * 1000
                    if 40 <= duration_ms <= 200: 
                        if duration_samples < min_duration_diff:
                             min_duration_diff = duration_samples; best_r_off_val = r_off_val; best_offset_idx = i
            if best_r_off_val != -1 and best_offset_idx != -1:
                qrs_durations_ms_list.append((best_r_off_val - r_on) / sampling_rate * 1000)
                used_offsets_indices[best_offset_idx] = True
        if qrs_durations_ms_list:
            results['QRS_durations_ms'] = np.array(qrs_durations_ms_list)
            results['QRS_mean_ms'] = np.mean(qrs_durations_ms_list)
            results['QRS_std_ms'] = np.std(qrs_durations_ms_list)

    # --- QT Intervals and QTc ---
    t_offsets = get_valid_indices(info_dict, 'ECG_T_Offsets')
    qt_intervals_ms_list = []; qtc_intervals_list = []
    if qrs_onsets and t_offsets and 'RR_intervals_ms' in results and len(results.get('RR_intervals_ms', [])) > 0:
        qrs_onsets_sorted = sorted(list(set(qrs_onsets)))
        t_offsets_sorted = sorted(list(set(t_offsets)))
        rr_intervals_sec = results['RR_intervals_ms'] / 1000.0 
        beat_idx = 0; used_t_offsets = [False] * len(t_offsets_sorted)
        for r_peak_ref in rpeaks_indices: 
            if beat_idx >= len(rr_intervals_sec): break
            current_q_on_candidates = [qon for qon in qrs_onsets_sorted if qon <= r_peak_ref]
            if not current_q_on_candidates: continue
            current_q_on = max(current_q_on_candidates)
            best_t_off_val = -1; min_qt_duration = float('inf'); best_t_off_idx = -1
            for i, t_off_val in enumerate(t_offsets_sorted):
                if not used_t_offsets[i] and t_off_val > current_q_on: 
                    qt_duration_samples = t_off_val - current_q_on
                    qt_duration_ms_candidate = (qt_duration_samples / sampling_rate) * 1000
                    if 200 <= qt_duration_ms_candidate <= 700:
                        if qt_duration_samples < min_qt_duration:
                            min_qt_duration = qt_duration_samples; best_t_off_val = t_off_val; best_t_off_idx = i
            if best_t_off_val != -1 and best_t_off_idx != -1:
                qt_duration_ms = (best_t_off_val - current_q_on) / sampling_rate * 1000
                qt_intervals_ms_list.append(qt_duration_ms); used_t_offsets[best_t_off_idx] = True
                rr_sec_for_this_beat = rr_intervals_sec[beat_idx]
                if rr_sec_for_this_beat > 0.1:
                    qtc = qt_duration_ms / np.sqrt(rr_sec_for_this_beat)
                    qtc_intervals_list.append(qtc)
            beat_idx +=1
        if qt_intervals_ms_list:
            results['QT_intervals_ms'] = np.array(qt_intervals_ms_list); results['QT_mean_ms'] = np.mean(qt_intervals_ms_list); results['QT_std_ms'] = np.std(qt_intervals_ms_list)
        if qtc_intervals_list:
            results['QTc_intervals'] = np.array(qtc_intervals_list); results['QTc_mean'] = np.mean(qtc_intervals_list); results['QTc_std'] = np.std(qtc_intervals_list)

    # --- Amplitudes ---
    peak_keys_amplitudes = {'P':'ECG_P_Peaks','Q':'ECG_Q_Peaks','R':'ECG_R_Peaks','S':'ECG_S_Peaks','T':'ECG_T_Peaks'}
    results['Amplitudes_mV'] = {} 
    for wave_name, peak_key in peak_keys_amplitudes.items():
        peaks = get_valid_indices(info_dict, peak_key)
        if peaks and len(peaks) > 0 and max(peaks) < len(cleaned_ecg): 
            amplitudes = cleaned_ecg[peaks]
            if len(amplitudes) > 0:
                results['Amplitudes_mV'][f'{wave_name}_peak_amps'] = amplitudes 
                results['Amplitudes_mV'][f'{wave_name}_peak_amp_mean'] = np.mean(amplitudes)
                results['Amplitudes_mV'][f'{wave_name}_peak_amp_std'] = np.std(amplitudes)

    # --- ST Segment Deviation ---
    st_deviations_mv = []
    p_offsets = get_valid_indices(info_dict, 'ECG_P_Offsets')
    if qrs_onsets and p_offsets and r_offsets: 
        j_points = sorted(list(set(r_offsets))); qrs_onsets_sorted = sorted(list(set(qrs_onsets)))
        p_onsets_sorted = sorted(list(set(p_onsets))) 
        for j_point_idx in j_points:
            current_q_on = [qon for qon in qrs_onsets_sorted if qon < j_point_idx]
            if not current_q_on: continue
            current_q_on = max(current_q_on)
            baseline_start_options = [pon for pon in p_onsets_sorted if pon < current_q_on]
            if not baseline_start_options: continue
            baseline_start = max(baseline_start_options)
            if baseline_start < current_q_on and (current_q_on - baseline_start) > int(0.04 * sampling_rate): 
                pr_segment = cleaned_ecg[baseline_start:current_q_on]
                if len(pr_segment) > 0:
                    isoelectric_level = np.median(pr_segment)
                    st_measurement_point = j_point_idx + int(0.06 * sampling_rate) 
                    if st_measurement_point < len(cleaned_ecg):
                        st_deviations_mv.append(cleaned_ecg[st_measurement_point] - isoelectric_level)
        if st_deviations_mv:
            results['ST_deviations_mV'] = np.array(st_deviations_mv)
            results['ST_mean_deviation_mV'] = np.mean(st_deviations_mv)
            results['ST_std_deviation_mV'] = np.std(st_deviations_mv)
    return results

# --- Main Analysis Function ---
def main_ultra_comprehensive_analysis():
    npy_file_path = "/home/tony/neurokit/validation/validation01/validation01.npy"
    analysis_type = "NSR"; sampling_rate = 100; preferred_lead_idx = 1; fallback_lead_idx = 0
    report_lines = []
    try:
        ecg_signal_all_leads = np.load(npy_file_path)
        report_lines.append(f"Successfully loaded ECG data from: {npy_file_path}")
    except Exception as e:
        report_lines.append(f"File load error: {str(e)} - Using dummy data."); print(report_lines[-1])
        ecg_signal_all_leads = np.array([nk.ecg_simulate(duration=10, sampling_rate=100, heart_rate=75, random_state=42)]*12)

    report_lines.append(f"ECG Signal Shape: {ecg_signal_all_leads.shape}")
    num_all_leads = ecg_signal_all_leads.shape[0]; STANDARD_12_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lead_to_analyze_idx = preferred_lead_idx if num_all_leads > preferred_lead_idx else fallback_lead_idx
    if num_all_leads == 0 or lead_to_analyze_idx >= num_all_leads : lead_to_analyze_idx = 0 
    if num_all_leads == 0: report_lines.append("Error: No leads found."); print("\n".join(report_lines)); exit(1)
    lead_name = f"Lead {STANDARD_12_LEAD_NAMES[lead_to_analyze_idx]} (index {lead_to_analyze_idx})" if num_all_leads == 12 else f"Lead {lead_to_analyze_idx + 1} (index {lead_to_analyze_idx})"
    ecg_lead_signal = ecg_signal_all_leads[lead_to_analyze_idx, :]
    lead_name_sanitized = lead_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+","")

    report_lines.insert(0, f"## Ultra Comprehensive ECG Analysis for {lead_name} ({analysis_type} focus) ##")
    report_lines.append(f"Analyzing Lead: {lead_name}\nSampling Rate: {sampling_rate} Hz\nSignal Duration: {len(ecg_lead_signal)/sampling_rate:.2f} seconds\n" + "-"*30)

    try:
        print(f"\nProcessing {lead_name} for comprehensive analysis...")
        signals, info = nk.ecg_process(ecg_lead_signal, sampling_rate=sampling_rate, method='neurokit')
        rpeaks_indices = get_valid_indices(info, 'ECG_R_Peaks')
        if not rpeaks_indices: raise ValueError("No R-peaks detected in the selected lead.")

        fig = nk.ecg_plot(signals, info); 
        if not isinstance(fig, plt.Figure): fig = plt.gcf()
        fig.suptitle(f"Processed ECG for {lead_name} with Delineations", y=1.02)
        save_plot(fig, f"{analysis_type.lower()}_0_processed_ecg", lead_name_sanitized)

        detailed_params = calculate_intervals_and_amplitudes(signals, info, rpeaks_indices, sampling_rate)

        report_lines.append("\n### A. Fiducial Points Detection ###")
        fiducial_keys = ['ECG_P_Onsets','ECG_P_Peaks','ECG_P_Offsets','ECG_Q_Peaks','ECG_R_Onsets','ECG_R_Peaks','ECG_R_Offsets','ECG_S_Peaks','ECG_T_Onsets','ECG_T_Peaks','ECG_T_Offsets']
        for key in fiducial_keys: report_lines.append(f"- Number of {key.replace('ECG_', '')} detected: {len(get_valid_indices(info, key))}")

        report_lines.append("\n### B. Intervals and Durations ###")
        interval_report_keys = {'RR Mean (ms)':('RR_mean_ms','RR_std_ms','RR_intervals_ms'),'PP Mean (ms)':('PP_mean_ms','PP_std_ms','PP_intervals_ms'),
                                'PR Mean (ms)':('PR_mean_ms','PR_std_ms','PR_intervals_ms'),'QRS Mean (ms)':('QRS_mean_ms','QRS_std_ms','QRS_durations_ms'),
                                'QT Mean (ms)':('QT_mean_ms','QT_std_ms','QT_intervals_ms'),'QTc Mean':('QTc_mean','QTc_std','QTc_intervals')}
        for disp_key, (mean_k, std_k, list_k) in interval_report_keys.items():
            if mean_k in detailed_params and list_k in detailed_params and len(detailed_params[list_k]) > 0:
                unit = "ms" if "ms" in disp_key else ""
                report_lines.append(f"- {disp_key}: {detailed_params[mean_k]:.2f}{unit} (StdDev: {detailed_params.get(std_k,np.nan):.2f}{unit}, N: {len(detailed_params[list_k])})")
            else: report_lines.append(f"- {disp_key}: Not reliably calculated or no valid intervals found.")
        
        report_lines.append("\n### C. Amplitudes (from Cleaned ECG) ###")
        if 'Amplitudes_mV' in detailed_params and detailed_params['Amplitudes_mV']:
            for amp_key in detailed_params['Amplitudes_mV']:
                if '_amp_mean' in amp_key: 
                    wave = amp_key.split('_')[0]
                    report_lines.append(f"- {wave} Peak Amp: Mean={detailed_params['Amplitudes_mV'][amp_key]:.3f}, Std={detailed_params['Amplitudes_mV'].get(f'{wave}_peak_amp_std',np.nan):.3f} (N={len(detailed_params['Amplitudes_mV'].get(f'{wave}_peak_amps',[]))}) units")
        else: report_lines.append("- Waveform amplitudes not calculated.")

        report_lines.append("\n### D. ST-Segment Analysis (Basic) ###")
        if 'ST_mean_deviation_mV' in detailed_params and 'ST_deviations_mV' in detailed_params and len(detailed_params['ST_deviations_mV']) > 0:
            report_lines.append(f"- ST-Segment Mean Deviation (J+60ms from PR baseline): {detailed_params['ST_mean_deviation_mV']:.3f} units")
            report_lines.append(f"- ST-Segment Std Deviation: {detailed_params['ST_std_deviation_mV']:.3f} (N: {len(detailed_params['ST_deviations_mV'])})")
        else: report_lines.append("- ST-Segment deviation: Not reliably calculated.")
        report_lines.append("- Note: T-wave/U-wave morphology requires visual assessment (not quantified beyond T-peak amplitude).")

        report_lines.append("\n### E. Axis Information ###")
        # QRS Axis Calculation (as implemented before)
        qrs_axis_str = "Not calculated."
        if num_all_leads >= 6:
            try:
                lead_I_idx = STANDARD_12_LEAD_NAMES.index("I"); lead_aVF_idx = STANDARD_12_LEAD_NAMES.index("aVF")
                if lead_I_idx < num_all_leads and lead_aVF_idx < num_all_leads:
                    sI, iI = nk.ecg_process(ecg_signal_all_leads[lead_I_idx,:],sampling_rate); cI=sI["ECG_Clean"].values
                    rI=get_valid_indices(iI,'ECG_R_Peaks'); qI=get_valid_indices(iI,'ECG_Q_Peaks'); sI_peaks=get_valid_indices(iI,'ECG_S_Peaks')
                    netI=(np.mean(cI[rI]) if rI else 0)+(np.mean(cI[qI]) if qI else 0)+(np.mean(cI[sI_peaks]) if sI_peaks else 0)
                    saVF,iaVF=nk.ecg_process(ecg_signal_all_leads[lead_aVF_idx,:],sampling_rate);caVF=saVF["ECG_Clean"].values
                    raVF=get_valid_indices(iaVF,'ECG_R_Peaks');qaVF=get_valid_indices(iaVF,'ECG_Q_Peaks');saVF_peaks=get_valid_indices(iaVF,'ECG_S_Peaks')
                    netaVF=(np.mean(caVF[raVF]) if raVF else 0)+(np.mean(caVF[qaVF]) if qaVF else 0)+(np.mean(caVF[saVF_peaks]) if saVF_peaks else 0)
                    if not(rI or qI or sI_peaks) and not(raVF or qaVF or saVF_peaks): qrs_axis_str = "Indeterminate (No QRS in Lead I & aVF)."
                    elif abs(netI)<1e-5 and abs(netaVF)<1e-5: qrs_axis_str = "Indeterminate (Net QRS in Lead I & aVF near zero)."
                    else: qrs_axis_str = f"{np.degrees(np.arctan2(netaVF, netI)):.1f} degrees"
                    report_lines.append(f"- QRS Axis (Lead I & aVF net deflections): {qrs_axis_str}")
                else: report_lines.append("- QRS Axis: Not attempted (Lead I or aVF index out of bounds).")
            except Exception as e_axis: report_lines.append(f"- QRS Axis Calc Error: {str(e_axis)}")
        else: report_lines.append("- QRS Axis: Not attempted (needs >= 6 leads).")
        report_lines.append("- P-wave axis and T-wave axis: Not implemented.")

        report_lines.append("\n### F. Heart Rate Variability (HRV) & Rate ###")
        if 'RR_mean_ms' in detailed_params:
            avg_hr = 60000 / detailed_params['RR_mean_ms'] if detailed_params['RR_mean_ms'] > 0 else 0
            report_lines.append(f"- Average Heart Rate: {avg_hr:.2f} bpm")
        else: report_lines.append("- Average Heart Rate: Not calculated.")
        
        if len(rpeaks_indices) > 5: # Min beats for reliable HRV
            try:
                # Use the main hrv() function
                hrv_summary = nk.hrv(rpeaks_indices, sampling_rate=sampling_rate, show=False)
                report_lines.append("  **HRV Metrics (from nk.hrv()):**")
                report_lines.append("  (Note: Some metrics, esp. frequency-domain & complex non-linear, are less reliable on short signals like this 10s strip.)")
                
                hrv_cols_time = [c for c in hrv_summary.columns if any(sub in c for sub in ['MeanNN','SDNN','RMSSD','pNN','CVNN','MedianNN','MadNN','MCVNN','IQRNN','SDRMSSD','Prc','MinNN','MaxNN','TINN','HTI','SDSD'])]
                hrv_cols_freq = [c for c in hrv_summary.columns if any(sub in c for sub in ['ULF','VLF','LF','HF','VHF','TP','LFHF','LFn','HFn','LnHF'])]
                hrv_cols_nonlin = [c for c in hrv_summary.columns if c not in hrv_cols_time and c not in hrv_cols_freq and "ECG" not in c and "RSP" not in c] # Crude catch-all

                if hrv_cols_time: report_lines.append("    Time-Domain:")
                for col in hrv_cols_time: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")
                
                if hrv_cols_freq: report_lines.append("    Frequency-Domain:")
                for col in hrv_cols_freq: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")
                
                if hrv_cols_nonlin: report_lines.append("    Non-Linear / Other:")
                for col in hrv_cols_nonlin: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")

                # RQA separately
                hrv_rqa_results = nk.hrv_rqa(rpeaks_indices, sampling_rate=sampling_rate, show=False)
                if hrv_rqa_results is not None and not hrv_rqa_results.empty:
                    report_lines.append("    Recurrence Quantification Analysis (RQA):")
                    for col in hrv_rqa_results.columns:
                        report_lines.append(f"      {col}: {hrv_rqa_results[col].iloc[0]:.4f}")
                
                # Plotting (Poincare from hrv_summary if available, or manual)
                if 'HRV_SD1' in hrv_summary.columns and 'HRV_SD2' in hrv_summary.columns and 'RR_intervals_ms' in detailed_params:
                    fig_poincare, ax_poincare = plt.subplots(figsize=(6,6))
                    nk.hrv_plot(hrv_summary, ax=ax_poincare, plot_type="poincare") # Try combined plot
                    ax_poincare.set_title(f"Poincaré Plot - {lead_name}") # May override internal title
                    save_plot(fig_poincare, f"{analysis_type.lower()}_1_poincare", lead_name_sanitized, tight_layout=False) # tight_layout false for hrv_plot
                    report_lines.append(f"  (Poincaré plot saved using hrv_plot)")

            except Exception as e_hrv: report_lines.append(f"  HRV Analysis failed/incomplete: {str(e_hrv)}")
        else: report_lines.append("  Not enough R-peaks for full HRV analysis.")
        report_lines.append(f"- Rhythm Type: Based on above parameters.")
        report_lines.append("- RSA (Respiratory Sinus Arrhythmia): Requires RSP signal, not calculated here.")


        report_lines.append("\n### G. Signal Quality ###")
        if 'ECG_Quality' in signals.columns:
            mean_quality = signals['ECG_Quality'].mean()
            report_lines.append(f"- Mean ECG Quality (NeuroKit): {mean_quality:.3f} (0-1, higher better relative to avg morphology)")
        else: report_lines.append("- ECG Quality metric not in signals DataFrame.")

        report_lines.append("\n### H. Cardiac Phase Information ###")
        phase_keys = ['ECG_Phase_Atrial','ECG_Phase_Completion_Atrial','ECG_Phase_Ventricular','ECG_Phase_Completion_Ventricular']
        plotted_phase = False; fig_phase, ax_phase_arr = plt.subplots(len(phase_keys),1,figsize=(15,2*len(phase_keys)),sharex=True)
        if not isinstance(ax_phase_arr,np.ndarray): ax_phase_arr = [ax_phase_arr] 
        time_axis = np.arange(len(signals))/sampling_rate; segment_end_sample = min(len(signals), int(5*sampling_rate)) 
        for i, key in enumerate(phase_keys):
            if key in signals.columns:
                report_lines.append(f"- Mean {key.replace('ECG_Phase_','')}: {signals[key].iloc[:segment_end_sample].mean():.3f}")
                ax_phase_arr[i].plot(time_axis[:segment_end_sample],signals[key].iloc[:segment_end_sample],label=key)
                ax_phase_arr[i].set_ylabel(key.replace('ECG_Phase_','').replace('_',' ')); ax_phase_arr[i].legend(loc='upper right'); plotted_phase = True
            else: ax_phase_arr[i].text(0.5,0.5,f'{key} N/A',ha='center',va='center',transform=ax_phase_arr[i].transAxes)
        if plotted_phase:
            ax_phase_arr[0].plot(time_axis[:segment_end_sample],signals['ECG_Clean'].iloc[:segment_end_sample].values*0.1+signals[phase_keys[0]].iloc[:segment_end_sample].mean(),label='ECG(scaled&offset)',color='gray',alpha=0.5) 
            ax_phase_arr[0].legend(loc='center right'); ax_phase_arr[-1].set_xlabel("Time (s)")
            fig_phase.suptitle(f"Cardiac Phase Signals - {lead_name}", y=0.99) # Adjusted y for suptitle
            save_plot(fig_phase, f"{analysis_type.lower()}_2_cardiac_phase", lead_name_sanitized)
            report_lines.append(f"  (Cardiac phase plot saved)")
        else: plt.close(fig_phase)

        report_lines.append("\n### I. Representative Heartbeats ###")
        try:
            epochs = nk.ecg_segment(signals['ECG_Clean'], rpeaks=rpeaks_indices, sampling_rate=sampling_rate, show=False)
            valid_epochs = {k: df for k, df in epochs.items() if df is not None and not df.empty and 'Signal' in df and not df['Signal'].isnull().all()}
            if valid_epochs:
                fig_beats, ax_beats = plt.subplots(figsize=(10,6))
                all_beat_signals_for_mean = []
                common_len = 0
                if valid_epochs:
                    lengths = [len(df["Signal"]) for df in valid_epochs.values() if df is not None and "Signal" in df] # Ensure df is not None
                    if lengths: common_len = int(np.median(lengths))

                for i, (epoch_idx, data) in enumerate(valid_epochs.items()):
                    if data is None or "Signal" not in data: continue
                    signal_data = data["Signal"].values
                    if common_len > 0 and len(signal_data) != common_len : 
                        signal_data = np.interp(np.linspace(0,1,common_len), np.linspace(0,1,len(signal_data)), signal_data)
                    if len(signal_data) == common_len: all_beat_signals_for_mean.append(signal_data)
                    if i < 10: ax_beats.plot(signal_data, label=f"Beat {epoch_idx}" if i < 3 else None, alpha=0.6)
                
                if all_beat_signals_for_mean:
                    mean_beat = np.mean(np.array(all_beat_signals_for_mean), axis=0)
                    ax_beats.plot(mean_beat, color='black', linewidth=2.5, label=f'Mean Beat (N={len(all_beat_signals_for_mean)})')
                ax_beats.set_title(f"Overlay of Segmented Heartbeats - {lead_name}"); ax_beats.set_xlabel("Samples from R-peak (approx)"); ax_beats.set_ylabel("Amplitude")
                ax_beats.legend(); save_plot(fig_beats, f"{analysis_type.lower()}_3_segmented_beats", lead_name_sanitized)
                report_lines.append(f"  (Segmented heartbeats plot saved)")
            else: report_lines.append("- Could not generate segmented heartbeats plot.")
        except Exception as e_segment: report_lines.append(f"- Error during heartbeat segmentation: {str(e_segment)}")

        if analysis_type == "NSR":
            report_lines.append("\n" + "="*30 + "\n### Clinical Summary Suggestion (NSR Focus) ###")
            report_lines.append("This detailed report provides extensive parameters for a Normal Sinus Rhythm assessment.")
            report_lines.append("Key NSR indicators to check from above sections:")
            report_lines.append("  - Rate: 60-100 bpm (Section F)")
            report_lines.append("  - Rhythm: Regular RR & PP intervals, low HRV_SDNN/RMSSD (Section B & F)")
            report_lines.append("  - P-waves: Present before each QRS, consistent morphology (Section A, C, I)")
            report_lines.append("  - P:QRS Ratio: Approx 1:1 (Section A - compare P_Peaks to R_Peaks count)")
            report_lines.append("  - PR Interval: 120-200ms, constant (Section B)")
            report_lines.append("  - QRS Duration: <120ms, typically <100ms (Section B)")
            report_lines.append("  - QRS Axis: Normal range (e.g., -30 to +90 degrees) (Section E)")
            report_lines.append("Please correlate with full clinical picture.")
            report_lines.append("="*30)

    except ValueError as ve: 
        report_lines.append(f"ANALYSIS STOPPED for {lead_name}: {str(ve)}")
        print(f"ANALYSIS STOPPED for {lead_name}: {str(ve)}")
    except Exception as e:
        error_msg = f"Critical Error during processing {lead_name}: {str(e)}"
        print(error_msg); report_lines.append(error_msg)
        import traceback; traceback.print_exc()

    report_filename = f"ultra_comprehensive_ecg_report_{lead_name_sanitized}.txt"
    with open(report_filename, "w") as f: f.write("\n".join(report_lines))
    
    print("\n\n" + "="*10 + " FINAL ULTRA COMPREHENSIVE REPORT " + "="*10)
    print("\n".join(report_lines))
    print(f"\nFull report saved to: {report_filename}")

if __name__ == "__main__":
    main_ultra_comprehensive_analysis()
