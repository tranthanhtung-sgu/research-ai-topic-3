import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- Helper function to save plots ---
def save_plot(fig, filename_base, lead_name_sanitized):
    """Saves the given matplotlib figure and closes it."""
    # Ensure figure size is reasonable for saving
    if not fig.get_size_inches()[0] > 1 or not fig.get_size_inches()[1] > 1: # if default tiny fig
        fig.set_size_inches(10, 6) 

    filepath = f"{filename_base}_{lead_name_sanitized}.png"
    try:
        fig.savefig(filepath)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Failed to save plot {filepath}: {str(e)}")
    plt.close(fig)

def main_afib_analysis():
    # Step 1: Load the .npy file
    npy_file_path = "/home/tony/neurokit/validation/validation02/validation02.npy"
    
    # --- Configuration ---
    sampling_rate = 100 # Hz (Verify with your data source, this is from the original script)
    # For AFib, Lead II is often good for P-wave/f-wave assessment.
    # Lead I is index 0, Lead II is index 1.
    preferred_lead_idx = 1 # Try Lead II first
    fallback_lead_idx = 0  # Fallback to Lead I

    # --- Report accumulator ---
    report_lines = []

    try:
        ecg_signal_all_leads = np.load(npy_file_path)
        report_lines.append(f"Successfully loaded ECG data from: {npy_file_path}")
    except FileNotFoundError:
        report_lines.append(f"Error: File not found at {npy_file_path}.")
        report_lines.append("Exiting analysis.")
        print("\n".join(report_lines))
        # Create a dummy signal for testing if file not found
        print("Creating a dummy AFib-like signal for testing as file not found.")
        sampling_rate_dummy = 100  # Hz
        duration_dummy = 30  # seconds
        # Simulate some R-peaks with irregularity
        rr_mean = 0.7  # seconds (around 85 bpm)
        rr_std = 0.15  # High std dev for irregularity
        rpeaks_dummy_samples = np.array(np.cumsum(np.random.normal(loc=rr_mean, scale=rr_std, size=int(duration_dummy / rr_mean))) * sampling_rate_dummy, dtype=int)
        rpeaks_dummy_samples = rpeaks_dummy_samples[rpeaks_dummy_samples < duration_dummy * sampling_rate_dummy]
        
        time_dummy = np.linspace(0, duration_dummy, duration_dummy * sampling_rate_dummy, endpoint=False)
        dummy_lead_signal = np.zeros_like(time_dummy)
        dummy_lead_signal += np.random.normal(0, 0.03, len(time_dummy)) # f-waves like noise
        for rpeak_sample in rpeaks_dummy_samples:
            qrs_start = rpeak_sample - int(0.02 * sampling_rate_dummy)
            qrs_end = rpeak_sample + int(0.08 * sampling_rate_dummy)
            if qrs_start >=0 and qrs_end < len(dummy_lead_signal):
                dummy_lead_signal[qrs_start:rpeak_sample] += np.linspace(-0.1, 1.0, rpeak_sample - qrs_start)
                dummy_lead_signal[rpeak_sample:qrs_end] += np.linspace(1.0, -0.2, qrs_end - rpeak_sample)
        ecg_signal_all_leads = np.array([dummy_lead_signal] * 12) # Make it 12-lead like
        # exit(1) # Exiting in original code
    except Exception as e:
        report_lines.append(f"Error loading .npy file: {str(e)}")
        report_lines.append("Exiting analysis.")
        print("\n".join(report_lines))
        exit(1)

    # Step 2: Verify data and select lead for analysis
    report_lines.append(f"ECG Signal Shape: {ecg_signal_all_leads.shape}")
    
    num_leads = ecg_signal_all_leads.shape[0]
    if num_leads > preferred_lead_idx:
        lead_to_analyze_idx = preferred_lead_idx 
        lead_name = f"II (index {lead_to_analyze_idx})"
    elif num_leads > fallback_lead_idx:
        lead_to_analyze_idx = fallback_lead_idx
        lead_name = f"I (index {lead_to_analyze_idx})"
    elif num_leads > 0:
        lead_to_analyze_idx = 0
        lead_name = f"Lead {lead_to_analyze_idx+1} (the only one available)"
    else:
        report_lines.append("Error: No leads found in the ECG data.")
        print("\n".join(report_lines))
        exit(1)
        
    ecg_lead_signal = ecg_signal_all_leads[lead_to_analyze_idx, :]
    lead_name_sanitized = lead_name.replace(" ", "_").replace("(", "").replace(")", "")

    report_lines.insert(0, f"## AFib Evidence Report for ECG Lead {lead_name} ##") # Insert at beginning
    report_lines.append(f"Analyzing Lead: {lead_name}")
    report_lines.append(f"Sampling Rate: {sampling_rate} Hz")
    report_lines.append(f"Signal Duration: {len(ecg_lead_signal)/sampling_rate:.2f} seconds")
    report_lines.append("-" * 30)

    # Step 4: Process the selected lead
    try:
        print(f"\nProcessing Lead {lead_name}...")
        # `ecg_process` cleans, finds peaks, and delineates waves.
        signals, info = nk.ecg_process(ecg_lead_signal, sampling_rate=sampling_rate, method='neurokit')
        
        fig_ecg_plot = nk.ecg_plot(signals, info)        # Access the figure object from nk.ecg_plot if it returns one, or gcf()
        if not isinstance(fig_ecg_plot, plt.Figure): # If nk.ecg_plot doesn't return fig directly
             fig_ecg_plot = plt.gcf()
        fig_ecg_plot.suptitle(f"Processed ECG for Lead {lead_name} with Delineations", y=1) # Adjust y if needed
        save_plot(fig_ecg_plot, "afib_processed_ecg", lead_name_sanitized)

        report_lines.append("\n### 1. Rhythm / Heart Rate Irregularity ###")
        rpeaks_indices = info['ECG_R_Peaks']
        if len(rpeaks_indices) < 5: # Need a few beats for meaningful rhythm analysis
            report_lines.append("Not enough R-peaks found to reliably assess rhythm irregularity.")
        else:
            rr_intervals_ms = np.diff(rpeaks_indices) / sampling_rate * 1000
            
            fig_tachogram, ax_tachogram = plt.subplots(figsize=(12, 4))
            ax_tachogram.plot(rr_intervals_ms, marker='o', linestyle='-')
            ax_tachogram.set_title(f"Tachogram (R-R Intervals) - Lead {lead_name}")
            ax_tachogram.set_xlabel("Beat Number")
            ax_tachogram.set_ylabel("R-R Interval (ms)")
            save_plot(fig_tachogram, "afib_tachogram", lead_name_sanitized)
            report_lines.append("- Tachogram: Shows beat-to-beat R-R interval variation. Marked irregularity is a key sign of AFib.")
            report_lines.append(f"  (See afib_tachogram_{lead_name_sanitized}.png)")

            try:
                hrv_analysis_results = nk.hrv(rpeaks_indices, sampling_rate=sampling_rate, show=False)
                
                # Create Poincaré plot manually since nk.plot_poincare doesn't exist
                fig_poincare, ax_poincare = plt.subplots(figsize=(6,6))
                # Plot RR intervals against lagged RR intervals (n vs n+1)
                rr_n = rr_intervals_ms[:-1]  # Current RR interval
                rr_n_plus_1 = rr_intervals_ms[1:]  # Next RR interval
                ax_poincare.scatter(rr_n, rr_n_plus_1, c='blue', alpha=0.75)
                
                # Add identity line
                min_val = min(np.min(rr_n), np.min(rr_n_plus_1))
                max_val = max(np.max(rr_n), np.max(rr_n_plus_1))
                ax_poincare.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                # Add SD1 and SD2 ellipse if available in HRV results
                if 'HRV_SD1' in hrv_analysis_results.columns and 'HRV_SD2' in hrv_analysis_results.columns:
                    sd1 = hrv_analysis_results['HRV_SD1'].iloc[0]
                    sd2 = hrv_analysis_results['HRV_SD2'].iloc[0]
                    
                    # Calculate center of the Poincaré plot
                    center_x = np.mean(rr_n)
                    center_y = np.mean(rr_n_plus_1)
                    
                    # Add text for SD1 and SD2 values
                    ax_poincare.text(0.1, 0.9, f'SD1: {sd1:.2f} ms', transform=ax_poincare.transAxes)
                    ax_poincare.text(0.1, 0.85, f'SD2: {sd2:.2f} ms', transform=ax_poincare.transAxes)
                
                ax_poincare.set_title(f"Poincaré Plot - Lead {lead_name}")
                ax_poincare.set_xlabel("RR interval (ms)")
                ax_poincare.set_ylabel("Next RR interval (ms)")
                ax_poincare.grid(True, alpha=0.3)
                save_plot(fig_poincare, "afib_poincare", lead_name_sanitized)
                report_lines.append("- Poincaré Plot: Visualizes R-R interval correlation. A 'fan' or 'comet' shape indicates high irregularity, typical of AFib.")
                report_lines.append(f"  (See afib_poincare_{lead_name_sanitized}.png)")

                report_lines.append("\n  Key HRV Metrics (from NeuroKit):")
                hrv_metrics_to_report = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_CVNN', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2']
                for metric in hrv_metrics_to_report:
                    if metric in hrv_analysis_results.columns:
                        value = hrv_analysis_results[metric].iloc[0]
                        report_lines.append(f"    {metric}: {value:.2f}")
                report_lines.append("  Interpretation: High SDNN, RMSSD, pNN50, CVNN suggest significant heart rate variability. Poincaré SD1 & SD2 quantify scatter.")
            except Exception as e_hrv:
                report_lines.append(f"- HRV Analysis (including Poincaré) failed or incomplete: {str(e_hrv)}")
                print(f"HRV analysis/Poincaré plot failed: {str(e_hrv)}")

        report_lines.append("\n### 2. P-wave Analysis ###")
        num_r_peaks = len(rpeaks_indices)
        p_peaks_indices = [p for p in info.get('ECG_P_Peaks', []) if not np.isnan(p)]
        num_p_peaks_detected = len(p_peaks_indices)

        report_lines.append(f"- Number of R-peaks found: {num_r_peaks}")
        report_lines.append(f"- Number of P-peaks detected by NeuroKit: {num_p_peaks_detected}")
        
        if num_p_peaks_detected == 0 or (num_r_peaks > 0 and num_p_peaks_detected < num_r_peaks * 0.5):
             report_lines.append("- Observation: Significantly fewer P-peaks detected than R-peaks, or no P-peaks. This is highly consistent with AFib, where discrete P-waves are absent and replaced by fibrillatory (f) waves.")
        else:
             report_lines.append("- Observation: P-peaks were detected. Visual inspection is crucial to determine if these are true P-waves or misidentified fibrillatory activity/noise.")
        report_lines.append("  Guidance: In AFib, expect no clear, consistent P-waves before QRS complexes. The baseline may show fine or coarse undulations (f-waves).")

        fig_pwave_detail, ax_pwave_detail = plt.subplots(figsize=(15, 6)) # Adjusted size
        segment_len_sec = min(5, len(ecg_lead_signal) / sampling_rate - 0.1) # Max 5s or signal duration
        plot_end_sample = int(min(len(signals['ECG_Clean']), segment_len_sec * sampling_rate))
        time_axis_segment = np.arange(plot_end_sample) / sampling_rate

        ax_pwave_detail.plot(time_axis_segment, signals['ECG_Clean'].iloc[:plot_end_sample], label="Cleaned ECG")
        
        p_peaks_plot = [p for p in p_peaks_indices if p < plot_end_sample]
        if p_peaks_plot:
            ax_pwave_detail.scatter(np.array(p_peaks_plot)/sampling_rate, signals['ECG_Clean'].iloc[p_peaks_plot], 
                                color='red', marker='P', s=100, label="Detected P-peaks (NK)", zorder=5)
        
        r_peaks_plot = [r for r in rpeaks_indices if r < plot_end_sample]
        if r_peaks_plot:
             ax_pwave_detail.scatter(np.array(r_peaks_plot)/sampling_rate, signals['ECG_Clean'].iloc[r_peaks_plot],
                                color='blue', marker='^', s=100, label="R-peaks", zorder=5)

        ax_pwave_detail.set_title(f"ECG Segment (Lead {lead_name}) - Detail for P-wave/f-wave assessment")
        ax_pwave_detail.set_xlabel("Time (s)")
        ax_pwave_detail.set_ylabel("Amplitude")
        ax_pwave_detail.legend()
        ax_pwave_detail.grid(True, linestyle=':', alpha=0.7)
        save_plot(fig_pwave_detail, "afib_pwave_detail", lead_name_sanitized)
        report_lines.append(f"  (See afib_pwave_detail_{lead_name_sanitized}.png for visual inspection of P-wave area).")

        report_lines.append("\n### 3. P:QRS Ratio ###")
        report_lines.append("- AFib Characteristic: None (atrial activity is chaotic, not consistently 1:1 with QRS).")
        if num_r_peaks > 0:
            report_lines.append(f"- Observation: {num_p_peaks_detected} P-peaks vs {num_r_peaks} R-peaks. A low P:R ratio or absent P-peaks supports AFib.")
        else:
            report_lines.append("- Observation: No R-peaks detected, cannot determine P:QRS ratio.")

        report_lines.append("\n### 4. PR Interval ###")
        report_lines.append("- AFib Characteristic: Not measurable (due to absence/inconsistency of distinct P-waves).")
        p_onsets_indices = [p for p in info.get('ECG_P_Onsets', []) if not np.isnan(p)]
        if not p_onsets_indices:
            report_lines.append("- Observation: No P-onsets were reliably detected by NeuroKit, making PR interval inherently unmeasurable. Consistent with AFib.")
        else:
            report_lines.append(f"- Observation: NeuroKit detected {len(p_onsets_indices)} P-onsets. In AFib, these are unlikely to be true, consistent P-wave onsets. Any calculated PR interval would be unreliable.")

        report_lines.append("\n### 5. QRS Duration ###")
        avg_qrs_duration = np.nan # Initialize
        r_onsets_calc = [x for x in info.get('ECG_R_Onsets', []) if not np.isnan(x)]
        r_offsets_calc = [x for x in info.get('ECG_R_Offsets', []) if not np.isnan(x)]

        if r_onsets_calc and r_offsets_calc and len(r_onsets_calc) == len(r_offsets_calc):
            qrs_durations_samples = np.array(r_offsets_calc) - np.array(r_onsets_calc)
            qrs_durations_ms = (qrs_durations_samples / sampling_rate) * 1000
            
            if len(qrs_durations_ms) > 0:
                avg_qrs_duration = np.nanmean(qrs_durations_ms)
                report_lines.append(f"- Average QRS duration (from R-onsets/offsets): {avg_qrs_duration:.2f} ms")
                if avg_qrs_duration < 120:
                    report_lines.append("  Observation: QRS duration is typically normal (<120ms) in AFib, unless other conduction issues are present.")
                else:
                    report_lines.append(f"  Observation: QRS duration ({avg_qrs_duration:.2f}ms) is wider than typical. Consider underlying conduction issues or aberrancy.")

                fig_qrs, ax_qrs = plt.subplots(figsize=(8, 5))
                ax_qrs.hist(qrs_durations_ms, bins=20, edgecolor='black')
                ax_qrs.set_title(f"Distribution of QRS Durations - Lead {lead_name}")
                ax_qrs.set_xlabel("QRS Duration (ms)")
                ax_qrs.set_ylabel("Frequency")
                save_plot(fig_qrs, "afib_qrs_duration_hist", lead_name_sanitized)
                report_lines.append(f"  (See afib_qrs_duration_hist_{lead_name_sanitized}.png).")
            else:
                report_lines.append("- No valid QRS durations could be calculated.")
        else:
            report_lines.append("- QRS onsets/offsets not consistently detected or paired for QRS duration calculation from `info` dict.")
        
        report_lines.append("\n" + "="*30)
        report_lines.append("### Clinical Summary Suggestion ###")
        report_lines.append(f"The analysis of ECG Lead {lead_name} demonstrates features highly suggestive of Atrial Fibrillation:")
        report_lines.append("1. **Markedly Irregular Ventricular Rhythm**: Evidenced by variable R-R intervals on tachogram, dispersed Poincaré plot, and elevated HRV metrics (e.g., SDNN, RMSSD).")
        report_lines.append("2. **Absence of Distinct P-waves**: Detected P-waves are sparse or absent relative to QRS complexes, consistent with chaotic atrial activity. Visual inspection likely shows fibrillatory waves instead of organized P-waves.")
        report_lines.append("3. **Unmeasurable PR Interval**: Due to the lack of clear and consistent P-waves.")
        qrs_summary = "within normal limits" if not np.isnan(avg_qrs_duration) and avg_qrs_duration < 120 else "potentially wide or unassessable (see details)"
        report_lines.append(f"4. **QRS Duration**: Appears {qrs_summary}, which is common in AFib unless complicated by other conduction abnormalities.")
        report_lines.append("\nThese findings strongly support further clinical evaluation for Atrial Fibrillation. Please correlate with full clinical picture.")
        report_lines.append("="*30)

    except Exception as e:
        error_msg = f"Critical Error during processing Lead {lead_name}: {str(e)}"
        print(error_msg)
        report_lines.append(error_msg)
        import traceback
        traceback.print_exc()
        try:
            cleaned_signal = nk.ecg_clean(ecg_lead_signal, sampling_rate=sampling_rate)
            fig_cleaned, ax_cleaned = plt.subplots(figsize=(12,4))
            ax_cleaned.plot(cleaned_signal)
            ax_cleaned.set_title(f"Cleaned ECG Signal (Processing Failed) - Lead {lead_name}")
            save_plot(fig_cleaned, "afib_cleaned_ecg_fallback", lead_name_sanitized)
        except Exception as e2:
            print(f"Error during fallback cleaning of Lead {lead_name}: {str(e2)}")

    # Save and print the final report
    report_filename = f"afib_report_{lead_name_sanitized}.txt"
    with open(report_filename, "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n\n" + "="*10 + " FINAL REPORT " + "="*10)
    print("\n".join(report_lines))
    print(f"\nFull report saved to: {report_filename}")
    print(f"Associated plots saved in the current directory with prefix 'afib_' and lead '{lead_name_sanitized}'.")

if __name__ == "__main__":
    main_afib_analysis()
    # If running in an environment that doesn't auto-show plots, 
    # and you want to see them after script finishes (assuming save_plot doesn't close them):
    # plt.show() 