import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# --- Helper function to save plots ---
def save_plot(fig, filename_base, lead_name_sanitized, tight_layout=True):
    """Saves the given matplotlib figure and closes it."""
    if not fig.get_size_inches()[0] > 1 or not fig.get_size_inches()[1] > 1: # Ensure reasonable default size
        fig.set_size_inches(10, 6)
    if tight_layout:
        try:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
        except Exception: # Catch errors if tight_layout fails for any reason
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
    if indices is None: # Handle cases where key exists but value is None
        return []
    # Ensure indices is iterable and filter
    if not hasattr(indices, '__iter__'):
        return []
    return [int(p) for p in indices if pd.notna(p) and isinstance(p, (int, float, np.number)) and p < 1e12] # Increased upper bound for safety

def calculate_intervals_and_amplitudes(signals_df, info_dict, rpeaks_indices, sampling_rate):
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
        results['RR_intervals_ms'] = rr_ms
        if len(rr_ms) > 0:
            results['RR_mean_ms'] = np.mean(rr_ms)
            results['RR_std_ms'] = np.std(rr_ms)


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
    # QRS onsets are often labeled as R_Onsets in NeuroKit's delineation
    qrs_onsets_candidates = get_valid_indices(info_dict, 'ECG_R_Onsets')
    if not qrs_onsets_candidates: # Fallback if R_Onsets are not there (older NeuroKit or different method)
         q_peaks = get_valid_indices(info_dict, 'ECG_Q_Peaks')
         if q_peaks:
             qrs_onsets_candidates = sorted(list(set(q_peaks))) # Use Q_peaks as proxy for QRS onset

    pr_intervals_ms_list = []
    if p_onsets and qrs_onsets_candidates:
        p_onsets_sorted = sorted(list(set(p_onsets)))
        qrs_onsets_sorted = sorted(list(set(qrs_onsets_candidates)))

        # More robust pairing: for each QRS onset, find the closest preceding P onset
        for q_on in qrs_onsets_sorted:
            relevant_p_onsets_before_q = [p_on for p_on in p_onsets_sorted if p_on < q_on]
            if relevant_p_onsets_before_q:
                # Choose the P_Onset closest to this QRS_Onset, but not *too* far.
                # This assumes P wave is reasonably close to QRS.
                closest_p_onset = max(relevant_p_onsets_before_q)
                # Additional check: ensure this p_onset hasn't been "claimed" by a previous QRS more appropriately
                # For simplicity here, we'll just use the max preceding one. More complex logic could check for R-P intervals.

                pr_duration_samples = q_on - closest_p_onset
                pr_duration_ms = (pr_duration_samples / sampling_rate) * 1000
                if 80 <= pr_duration_ms <= 350:  # Physiological range for PR
                    pr_intervals_ms_list.append(pr_duration_ms)
        if pr_intervals_ms_list:
            results['PR_intervals_ms'] = np.array(pr_intervals_ms_list)
            results['PR_mean_ms'] = np.mean(pr_intervals_ms_list)
            results['PR_std_ms'] = np.std(pr_intervals_ms_list)

    # --- QRS Durations ---
    r_onsets = get_valid_indices(info_dict, 'ECG_R_Onsets') # QRS Onset
    s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_S_Offsets') # S_Offsets are more common for QRS end
    if not s_offsets_or_r_offsets:
        s_offsets_or_r_offsets = get_valid_indices(info_dict, 'ECG_R_Offsets') # Fallback to R_Offsets

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
                    duration_samples = r_off_val - r_on_val
                    duration_ms = (duration_samples / sampling_rate) * 1000
                    if 40 <= duration_ms <= 200: # Physiological QRS duration
                        # Prefer the closest offset that forms a valid QRS
                        if duration_samples < min_duration_diff:
                             min_duration_diff = duration_samples
                             best_qrs_off_val = r_off_val
                             best_offset_idx = i
            if best_qrs_off_val != -1 and best_offset_idx != -1:
                qrs_durations_ms_list.append((best_qrs_off_val - r_on_val) / sampling_rate * 1000)
                used_offsets_indices[best_offset_idx] = True # Mark as used
        if qrs_durations_ms_list:
            results['QRS_durations_ms'] = np.array(qrs_durations_ms_list)
            results['QRS_mean_ms'] = np.mean(qrs_durations_ms_list)
            results['QRS_std_ms'] = np.std(qrs_durations_ms_list)

    # --- QT Intervals and QTc ---
    t_offsets = get_valid_indices(info_dict, 'ECG_T_Offsets')
    qt_intervals_ms_list = []; qtc_intervals_list = []
    # Use QRS onsets (r_onsets from above)
    if r_onsets and t_offsets and 'RR_intervals_ms' in results and len(results.get('RR_intervals_ms', [])) > 0:
        qrs_onsets_sorted = sorted(list(set(r_onsets))) # Already sorted if taken from above
        t_offsets_sorted = sorted(list(set(t_offsets)))
        rr_intervals_sec = results['RR_intervals_ms'] / 1000.0
        
        # Ensure RR intervals align with QRS onsets for QTc calculation
        # We need an RR interval for *each* QT interval. The RR preceding the QRS is typically used.
        
        beat_idx = 0 # Index for rr_intervals_sec
        used_t_offsets = [False] * len(t_offsets_sorted)

        for q_on_idx, q_on_val in enumerate(qrs_onsets_sorted):
            # Find corresponding T_Offset
            best_t_off_val = -1
            min_qt_duration = float('inf')
            best_t_off_idx = -1
            for i, t_off_val in enumerate(t_offsets_sorted):
                if not used_t_offsets[i] and t_off_val > q_on_val:
                    qt_duration_samples = t_off_val - q_on_val
                    qt_duration_ms_candidate = (qt_duration_samples / sampling_rate) * 1000
                    if 200 <= qt_duration_ms_candidate <= 700: # Physiological QT
                        if qt_duration_samples < min_qt_duration: # Find closest T_offset after QRS_onset
                            min_qt_duration = qt_duration_samples
                            best_t_off_val = t_off_val
                            best_t_off_idx = i
            
            if best_t_off_val != -1 and best_t_off_idx != -1:
                qt_duration_ms = (best_t_off_val - q_on_val) / sampling_rate * 1000
                qt_intervals_ms_list.append(qt_duration_ms)
                used_t_offsets[best_t_off_idx] = True

                # QTc calculation: Use RR interval ending at the R-peak associated with this QRS
                # Find the R-peak closest to this q_on_val
                r_peak_for_this_qrs = -1
                if rpeaks_indices:
                    r_peaks_after_qon = [r for r in rpeaks_indices if r >= q_on_val]
                    if r_peaks_after_qon:
                        r_peak_for_this_qrs = min(r_peaks_after_qon)
                
                rr_sec_for_this_beat = np.nan
                if r_peak_for_this_qrs != -1:
                    try:
                        # Find index of this R-peak in the original rpeaks_indices list
                        r_peak_list_idx = rpeaks_indices.index(r_peak_for_this_qrs)
                        if r_peak_list_idx > 0: # Need a preceding R-peak to have an RR interval
                           rr_sec_for_this_beat = (rpeaks_indices[r_peak_list_idx] - rpeaks_indices[r_peak_list_idx-1]) / sampling_rate
                        elif q_on_idx < len(rr_intervals_sec): # Fallback if direct R-peak matching is hard
                           rr_sec_for_this_beat = rr_intervals_sec[q_on_idx]


                    except (ValueError, IndexError):
                        # If r_peak_for_this_qrs not in list or out of bounds for rr_intervals_sec
                        # Try to use beat_idx as a fallback, but this is less robust
                        if beat_idx < len(rr_intervals_sec):
                            rr_sec_for_this_beat = rr_intervals_sec[beat_idx]


                if pd.notna(rr_sec_for_this_beat) and rr_sec_for_this_beat > 0.1: # Min sensible RR for QTc
                    qtc = qt_duration_ms / np.sqrt(rr_sec_for_this_beat) # Bazett's
                    qtc_intervals_list.append(qtc)
            
            beat_idx += 1 # Increment for fallback RR interval indexing

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
    # QRS onsets (r_onsets) and QRS offsets (s_offsets_or_r_offsets) from QRS duration calculation
    if r_onsets and s_offsets_or_r_offsets and p_onsets:
        j_points = sorted(list(set(s_offsets_or_r_offsets))) # J-point is typically end of QRS
        qrs_onsets_sorted = sorted(list(set(r_onsets)))
        p_onsets_sorted = sorted(list(set(p_onsets)))

        for j_point_val in j_points:
            # Find the QRS onset that corresponds to this J-point
            current_qrs_onsets_before_j = [qon for qon in qrs_onsets_sorted if qon < j_point_val]
            if not current_qrs_onsets_before_j: continue
            current_q_on = max(current_qrs_onsets_before_j)

            # Find the P onset for the PR segment baseline
            baseline_p_onsets_before_qrs = [pon for pon in p_onsets_sorted if pon < current_q_on]
            if not baseline_p_onsets_before_qrs: continue
            baseline_start_p_onset = max(baseline_p_onsets_before_qrs)

            # Ensure PR segment is distinct and before QRS onset
            if baseline_start_p_onset < current_q_on and (current_q_on - baseline_start_p_onset) > int(0.04 * sampling_rate):
                # Make sure PR segment doesn't overlap with a previous T-wave offset or P-wave offset
                # For simplicity, we use P-onset to QRS-onset as PR segment for baseline
                pr_segment_for_baseline = cleaned_ecg[baseline_start_p_onset:current_q_on]
                if len(pr_segment_for_baseline) > 0:
                    isoelectric_level = np.median(pr_segment_for_baseline) # Median is robust to P-wave itself

                    st_measurement_point_sample = j_point_val + int(0.06 * sampling_rate) # ST measurement at J+60ms
                    if st_measurement_point_sample < len(cleaned_ecg):
                        st_deviations_mv.append(cleaned_ecg[st_measurement_point_sample] - isoelectric_level)
        if st_deviations_mv:
            results['ST_deviations_mV'] = np.array(st_deviations_mv)
            results['ST_mean_deviation_mV'] = np.mean(st_deviations_mv)
            results['ST_std_deviation_mV'] = np.std(st_deviations_mv)
    return results

def plot_rr_interval_distribution(info, rpeaks_indices, sampling_rate, lead_name, lead_name_sanitized, 
                                 interpolate=False, detrend=None, interpolation_rate=100):
    """
    Create and save a plot showing the distribution of RR intervals using NeuroKit's intervals_process.
    
    Parameters
    ----------
    info : dict
        NeuroKit processing info dictionary with R-peak locations
    rpeaks_indices : list or array
        List of R-peak indices
    sampling_rate : int
        Sampling rate in Hz
    lead_name : str
        Name of the ECG lead being analyzed
    lead_name_sanitized : str
        Sanitized version of lead_name for file naming
    interpolate : bool, optional
        Whether to interpolate the interval signal. Default is False.
    detrend : str, optional
        Detrending method. Options include "polynomial", "tarvainen2002", "loess", "locreg".
        Default is None (no detrending).
    interpolation_rate : int, optional
        Sampling rate (Hz) of the interpolated interbeat intervals. Default is 100Hz.
    
    Returns
    -------
    str
        Information about what was plotted
    """
    try:
        if len(rpeaks_indices) < 2:
            return "Not enough R-peaks to calculate RR interval distribution"
        
        # Calculate RR intervals in milliseconds and timestamps in seconds
        rri = np.diff(rpeaks_indices) / sampling_rate * 1000  # Convert to milliseconds
        rri_time = np.array(rpeaks_indices[1:]) / sampling_rate  # Timestamps in seconds
        
        # Use NeuroKit's intervals_process for preprocessing
        processed_rri, processed_rri_time, _ = nk.intervals_process(
            intervals=rri,
            intervals_time=rri_time,
            interpolate=interpolate,
            interpolation_rate=interpolation_rate,
            detrend=detrend
        )
        
        # For the histogram, we'll use the original non-interpolated intervals
        # unless interpolate is specifically requested
        intervals_for_hist = processed_rri if interpolate else rri
        
        # Calculate statistics
        mean_rr = np.mean(intervals_for_hist)
        std_rr = np.std(intervals_for_hist)
        median_rr = np.median(intervals_for_hist)
        min_rr = np.min(intervals_for_hist)
        max_rr = np.max(intervals_for_hist)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine bin width and number of bins
        range_rr = max_rr - min_rr
        # Aim for about 15-20 bins, with reasonable round numbers
        bin_width = 25  # 25ms bin width is often used for RR intervals
        num_bins = int(np.ceil(range_rr / bin_width))
        num_bins = max(10, min(num_bins, 30))  # Ensure reasonable number of bins
        
        # Create histogram
        n, bins, patches = ax.hist(
            intervals_for_hist, 
            bins=num_bins,
            edgecolor='black',
            alpha=0.7,
            color='skyblue'
        )
        
        # Add vertical line for mean
        ax.axvline(x=mean_rr, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean RR: {mean_rr:.1f} ms')
        
        # Add text with statistics
        stats_text = (
            f"Mean: {mean_rr:.1f} ms\n"
            f"Median: {median_rr:.1f} ms\n"
            f"Std Dev: {std_rr:.1f} ms\n"
            f"Range: {min_rr:.1f} - {max_rr:.1f} ms\n"
            f"N beats: {len(intervals_for_hist)}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Preprocessing info in the title
        preproc_info = ""
        if interpolate:
            preproc_info += f" (Interpolated at {interpolation_rate}Hz"
            if detrend:
                preproc_info += f", Detrended: {detrend}"
            preproc_info += ")"
        elif detrend:
            preproc_info += f" (Detrended: {detrend})"
            
        # Set labels and title
        ax.set_xlabel('RR Interval [ms]')
        ax.set_ylabel('Frequency')
        ax.set_title(f'RR Interval Distribution - {lead_name}{preproc_info}')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Optional: If interpolate is True, add a subplot to show the processed time series
        if interpolate:
            # Create a new figure for the time series
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            
            # Plot original intervals
            ax_ts.plot(rri_time, rri, 'o-', alpha=0.5, label='Original RR intervals')
            
            # Plot processed intervals
            ax_ts.plot(processed_rri_time, processed_rri, 'r-', linewidth=2, 
                     label='Processed RR intervals')
            
            ax_ts.set_xlabel('Time (seconds)')
            ax_ts.set_ylabel('RR Interval (ms)')
            ax_ts.set_title(f'RR Interval Time Series - {lead_name}{preproc_info}')
            ax_ts.grid(True, alpha=0.3)
            ax_ts.legend()
            
            # Save the time series plot
            save_plot(fig_ts, f"NSR_rr_interval_timeseries", lead_name_sanitized)
        
        # Save the histogram plot
        save_plot(fig, f"NSR_rr_interval_distribution", lead_name_sanitized)
        
        # Return information about the processing
        result_msg = f"RR interval distribution created from {len(intervals_for_hist)} intervals"
        if interpolate:
            result_msg += f" (interpolated at {interpolation_rate}Hz"
            if detrend:
                result_msg += f", detrended using {detrend}"
            result_msg += ")"
        elif detrend:
            result_msg += f" (detrended using {detrend})"
        
        return result_msg
    
    except Exception as e:
        return f"Error creating RR interval distribution plot: {str(e)}"

def plot_ecg_segments(signals_df, rpeaks_indices, sampling_rate, lead_name, lead_name_sanitized):
    """
    Create and save a plot of segmented ECG heartbeats using NeuroKit's ecg_segment function.
    
    Parameters
    ----------
    signals_df : DataFrame
        NeuroKit processed signals dataframe
    rpeaks_indices : list or array
        List of R-peak indices
    sampling_rate : int
        Sampling rate in Hz
    lead_name : str
        Name of the ECG lead being analyzed
    lead_name_sanitized : str
        Sanitized version of lead_name for file naming
    
    Returns
    -------
    str
        Information about what was plotted
    """
    try:
        if len(rpeaks_indices) < 2:
            return "Not enough R-peaks to create segmented heartbeats plot"
            
        # Extract the cleaned ECG signal
        ecg_cleaned = signals_df['ECG_Clean'].values
        
        # Use NeuroKit's ecg_segment function to segment heartbeats
        # This returns a dict of DataFrames, one for each heartbeat
        segments = nk.ecg_segment(ecg_cleaned, rpeaks=rpeaks_indices, sampling_rate=sampling_rate, show=False)
        
        if not segments or len(segments) == 0:
            return "No segments created, ecg_segment() returned empty results"
        
        # Get the segment keys and ensure they're properly sorted
        segment_keys = sorted(list(segments.keys()))
        num_segments = len(segment_keys)
        
        if num_segments == 0:
            return "No valid segments found in the segmentation results"
            
        # Create a single figure for superimposed heartbeats
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
        
        # For coloring by segment number
        colors = plt.cm.viridis(np.linspace(0, 0.8, min(num_segments, 30))) # Max 30 distinct colors for overlay
        
        # Calculate time axis (centered around R-peak at time 0)
        beat_lengths = []
        for i, key in enumerate(segment_keys):
            if 'Signal' in segments[key].columns:
                beat_lengths.append(len(segments[key]['Signal'].values))
        
        if not beat_lengths:
            plt.close(fig) # Close the figure if no valid data
            return "No valid signal data found in segments"
            
        median_length = int(np.median(beat_lengths))
        if median_length <=0: # Handle case where median length is zero or negative
            plt.close(fig)
            return "Median beat length is zero or negative, cannot proceed."

        rpeak_idx = median_length // 2  # Assume R-peak is at the middle
        
        # Time in seconds relative to R-peak
        time_axis = np.linspace(-rpeak_idx/sampling_rate, (median_length-rpeak_idx-1)/sampling_rate, median_length)
        
        # Store resampled beats for average calculation
        resampled_beats = []
        
        # Plot individual beats (limit to first 30 for clarity)
        max_beats_to_plot_on_ax1 = min(30, num_segments)
        for i in range(max_beats_to_plot_on_ax1):
            if i >= len(segment_keys):
                break
                
            key = segment_keys[i]
            if 'Signal' not in segments[key].columns:
                continue
                
            beat = segments[key]['Signal'].values
            
            # Only include beats with reasonable length
            if len(beat) > 10:
                # Resample to median length for proper overlay
                resampled = np.interp(
                    np.linspace(0, 1, median_length),
                    np.linspace(0, 1, len(beat)),
                    beat
                )
                resampled_beats.append(resampled)
                
                # Plot with color gradient
                alpha = 0.5 if i < max_beats_to_plot_on_ax1 - 1 else 0.7  # Make last beat more visible
                ax1.plot(time_axis, resampled, color=colors[i % len(colors)], alpha=alpha, linewidth=1)
        
        # Calculate and plot the average beat if we have resampled beats
        avg_beat = None # Initialize avg_beat
        if resampled_beats:
            avg_beat_array = np.array(resampled_beats)
            if avg_beat_array.ndim == 2 and avg_beat_array.shape[0] > 0: # Ensure it's a 2D array with at least one beat
                 avg_beat = np.nanmean(avg_beat_array, axis=0) # Use nanmean for robustness
                 if np.all(np.isnan(avg_beat)): # if nanmean results in all nans
                     avg_beat = None
                 else:
                     ax1.plot(time_axis, avg_beat, color='red', linewidth=4.0, label='Average Beat')
        
        # Add vertical line at R-peak (time 0)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Labels and title
        ax1.set_xlabel('Time (s) relative to R-peak')
        ax1.set_ylabel('Amplitude')
        
        # Update title to show "Individual Average Heartbeat" and include average heart rate if available
        avg_heart_rate = ""
        if len(rpeaks_indices) > 1:
            # Calculate RR intervals in ms directly
            rr_intervals_ms = np.diff(rpeaks_indices) / sampling_rate * 1000
            if len(rr_intervals_ms) > 0:
                avg_rr_ms = np.mean(rr_intervals_ms)
                if avg_rr_ms > 0:
                    avg_hr = 60000 / avg_rr_ms  # Convert from ms to BPM
                    avg_heart_rate = f" (Average Heart Rate: {avg_hr:.1f} BPM)"
        
        ax1.set_title(f'Individual Average Heartbeat - {lead_name}{avg_heart_rate}')
        ax1.grid(True, alpha=0.3)
        if avg_beat is not None: # Check if avg_beat was successfully calculated and plotted
            ax1.legend()
        
        # Adjust layout and save the figure
        save_plot(fig, f"NSR_segmented_heartbeats", lead_name_sanitized)
        
        # Create a second figure with a heart rate variability plot (beat-to-beat intervals)
        if len(rpeaks_indices) > 2:
            fig_hrv, ax_hrv = plt.subplots(figsize=(10, 5))
            
            # Calculate R-R intervals in ms
            rr_intervals = np.diff(rpeaks_indices) / sampling_rate * 1000
            
            # Plot beat-to-beat intervals
            ax_hrv.plot(np.arange(1, len(rr_intervals) + 1), rr_intervals, 'o-', color='blue', alpha=0.7)
            
            # Add horizontal line for mean
            ax_hrv.axhline(y=np.mean(rr_intervals), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rr_intervals):.1f} ms')
            
            # Labels and title
            ax_hrv.set_xlabel('Beat Number')
            ax_hrv.set_ylabel('R-R Interval (ms)')
            ax_hrv.set_title(f'Beat-to-Beat Intervals - {lead_name}')
            ax_hrv.grid(True, alpha=0.3)
            ax_hrv.legend()
            
            # Save the figure
            save_plot(fig_hrv, f"NSR_beat_to_beat_intervals", lead_name_sanitized)
        
        return f"Created segmented heartbeats plot from {len(resampled_beats)} valid heartbeats"
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in plot_ecg_segments: {str(e)}\n{error_details}")
        # Ensure figure is closed if an error occurs before save_plot
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        if 'fig_hrv' in locals() and plt.fignum_exists(fig_hrv.number):
            plt.close(fig_hrv)
        return f"Error creating segmented heartbeats plot: {str(e)}"

# --- New function for Delineation Plots ---
def generate_delineation_plots(signals_df, info_dict, rpeaks_indices, sampling_rate, plot_prefix, lead_name_sanitized, report_lines):
    report_lines.append("\n### J. Detailed Waveform Delineation Plots (Zoomed) ###")
    if 'ECG_Clean' not in signals_df.columns:
        report_lines.append("- Skipping delineation plots: 'ECG_Clean' not found in signals DataFrame.")
        return

    ecg_cleaned = signals_df['ECG_Clean'].values
    if len(ecg_cleaned) == 0:
        report_lines.append("- Skipping delineation plots: Cleaned ECG signal is empty.")
        return

    N_CYCLES_FOR_ZOOM = 5  # Number of cardiac cycles to try to display in zoomed plots
    DEFAULT_ZOOM_SECONDS = 10

    # Determine zoom window
    if len(rpeaks_indices) >= N_CYCLES_FOR_ZOOM:
        end_rpeak_sample = rpeaks_indices[N_CYCLES_FOR_ZOOM - 1]
        max_sample_for_zoom = min(len(ecg_cleaned), end_rpeak_sample + int(sampling_rate * 1.0)) # 1s padding
    elif len(rpeaks_indices) > 0:
        end_rpeak_sample = rpeaks_indices[-1]
        max_sample_for_zoom = min(len(ecg_cleaned), end_rpeak_sample + int(sampling_rate * 1.0))
    else:
        max_sample_for_zoom = min(len(ecg_cleaned), int(DEFAULT_ZOOM_SECONDS * sampling_rate))

    if max_sample_for_zoom <= 0: # If signal is too short or no R-peaks
        max_sample_for_zoom = len(ecg_cleaned)

    ecg_zoom_segment = ecg_cleaned[:max_sample_for_zoom]
    if len(ecg_zoom_segment) == 0:
        report_lines.append("- Skipping delineation plots: Zoom segment is empty.")
        return

    plot_time_axis = np.arange(len(ecg_zoom_segment)) / sampling_rate

    # Helper to get events within the zoom window for plotting
    def get_events_in_zoom(all_event_indices):
        return [idx for idx in all_event_indices if idx < max_sample_for_zoom]
        
    # Custom function to plot events on a specific axis
    def plot_events_on_axis(ax, events, signal, color='red'):
        # First plot the signal on the axis
        if signal is not None:
            time = np.arange(len(signal)) / sampling_rate
            ax.plot(time, signal)
        
        # Then manually add vertical lines for events
        for event in events:
            if event < len(ecg_zoom_segment): # Ensure event is within the zoom segment's sample range
                event_time = event / sampling_rate
                if event_time <= plot_time_axis[-1]: # Ensure event_time is within plotted time axis
                    ax.axvline(x=event_time, color=color, linestyle='--')

    # Plot 1: Zoomed R-peaks
    rpeaks_zoomed = get_events_in_zoom(rpeaks_indices)
    if rpeaks_zoomed:
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        # Use our custom function to plot events
        plot_events_on_axis(ax1, rpeaks_zoomed, ecg_zoom_segment, color='red')
        ax1.set_title(f"{lead_name_sanitized} - Zoomed R-peaks")
        ax1.set_xlabel("Time (s)")
        save_plot(fig1, f"{plot_prefix}_delineate_zoomed_R", lead_name_sanitized)
        report_lines.append(f"- Zoomed R-peaks plot saved ({len(rpeaks_zoomed)} peaks shown).")
    else:
        report_lines.append("- Skipping Zoomed R-peaks plot: No R-peaks in zoom window.")


    # Plot 2: Key Wave Peaks (P, Q, R, S, T) Zoomed
    p_peaks_all = get_valid_indices(info_dict, 'ECG_P_Peaks')
    q_peaks_all = get_valid_indices(info_dict, 'ECG_Q_Peaks')
    s_peaks_all = get_valid_indices(info_dict, 'ECG_S_Peaks')
    t_peaks_all = get_valid_indices(info_dict, 'ECG_T_Peaks')

    events_for_all_peaks_plot = []
    event_labels_all_peaks = []
    event_colors_all_peaks = []

    collections = [
        (p_peaks_all, "P Peaks", "blue"), (q_peaks_all, "Q Peaks", "green"),
        (rpeaks_indices, "R Peaks", "red"), (s_peaks_all, "S Peaks", "purple"),
        (t_peaks_all, "T Peaks", "orange")
    ]
    for ev_all, label, color in collections:
        ev_zoomed = get_events_in_zoom(ev_all)
        if ev_zoomed:
            events_for_all_peaks_plot.append(ev_zoomed)
            event_labels_all_peaks.append(label)
            event_colors_all_peaks.append(color)

    if events_for_all_peaks_plot:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        # First plot the signal
        ax2.plot(plot_time_axis, ecg_zoom_segment)
        # Then add event markers for each type of peak
        for i, events_list in enumerate(events_for_all_peaks_plot):
            color = event_colors_all_peaks[i] if i < len(event_colors_all_peaks) else 'red'
            # Manually plot events
            for event_sample in events_list:
                if event_sample < len(ecg_zoom_segment): # Ensure event is within the zoom segment's sample range
                    event_time = event_sample / sampling_rate
                    if event_time <= plot_time_axis[-1]:  # Ensure event_time is within plotted time axis
                         ax2.axvline(x=event_time, color=color, linestyle='--')
            
        # Create manual legend for clarity
        handles = [plt.Line2D([0], [0], marker='|', color=c, linestyle='None', markersize=10, label=l)
                   for l, c in zip(event_labels_all_peaks, event_colors_all_peaks)]
        ax2.legend(handles=handles, loc='upper right')
        ax2.set_title(f"{lead_name_sanitized} - Zoomed P,Q,R,S,T Peaks")
        ax2.set_xlabel("Time (s)")
        save_plot(fig2, f"{plot_prefix}_delineate_zoomed_all_wave_peaks", lead_name_sanitized)
        report_lines.append("- Zoomed All Wave Peaks plot saved.")
    else:
        report_lines.append("- Skipping All Wave Peaks plot: No relevant peaks found in zoom window.")


    # Plot 3: P-Wave Boundaries Zoomed
    p_onsets_all = get_valid_indices(info_dict, 'ECG_P_Onsets')
    p_offsets_all = get_valid_indices(info_dict, 'ECG_P_Offsets')
    p_onsets_zoomed = get_events_in_zoom(p_onsets_all)
    p_offsets_zoomed = get_events_in_zoom(p_offsets_all)
    if p_onsets_zoomed or p_offsets_zoomed:
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        # First plot the signal
        ax3.plot(plot_time_axis, ecg_zoom_segment)
        # Then add P-onsets and P-offsets markers
        p_boundaries_events_lists = []
        p_boundaries_labels = []
        p_boundaries_colors = []
        if p_onsets_zoomed: p_boundaries_events_lists.append(p_onsets_zoomed); p_boundaries_labels.append("P Onsets"); p_boundaries_colors.append("cyan")
        if p_offsets_zoomed: p_boundaries_events_lists.append(p_offsets_zoomed); p_boundaries_labels.append("P Offsets"); p_boundaries_colors.append("magenta")
        
        if p_boundaries_events_lists:
            for i, events_list in enumerate(p_boundaries_events_lists):
                color = p_boundaries_colors[i] if i < len(p_boundaries_colors) else 'blue'
                for event_sample in events_list:
                    if event_sample < len(ecg_zoom_segment):
                        event_time = event_sample / sampling_rate
                        if event_time <= plot_time_axis[-1]:
                            ax3.axvline(x=event_time, color=color, linestyle='--')
                
            handles = [plt.Line2D([0], [0], marker='|', color=c, linestyle='None', markersize=10, label=l)
                       for l, c in zip(p_boundaries_labels, p_boundaries_colors)]
            ax3.legend(handles=handles, loc='upper right')
            ax3.set_title(f"{lead_name_sanitized} - Zoomed P-Wave Boundaries")
            ax3.set_xlabel("Time (s)")
            save_plot(fig3, f"{plot_prefix}_delineate_zoomed_P_boundaries", lead_name_sanitized)
            report_lines.append("- Zoomed P-Wave Boundaries plot saved.")
    else:
        report_lines.append("- Skipping P-Wave Boundaries plot: No P-onsets/offsets in zoom window.")

    # Plot 4: T-Wave Boundaries Zoomed
    t_onsets_all = get_valid_indices(info_dict, 'ECG_T_Onsets')
    t_offsets_all = get_valid_indices(info_dict, 'ECG_T_Offsets')
    t_onsets_zoomed = get_events_in_zoom(t_onsets_all)
    t_offsets_zoomed = get_events_in_zoom(t_offsets_all)
    if t_onsets_zoomed or t_offsets_zoomed:
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.plot(plot_time_axis, ecg_zoom_segment)
        t_boundaries_events_lists = []
        t_boundaries_labels = []
        t_boundaries_colors = []
        if t_onsets_zoomed: t_boundaries_events_lists.append(t_onsets_zoomed); t_boundaries_labels.append("T Onsets"); t_boundaries_colors.append("lime")
        if t_offsets_zoomed: t_boundaries_events_lists.append(t_offsets_zoomed); t_boundaries_labels.append("T Offsets"); t_boundaries_colors.append("gold")
        
        if t_boundaries_events_lists:
            for i, events_list in enumerate(t_boundaries_events_lists):
                color = t_boundaries_colors[i] if i < len(t_boundaries_colors) else 'orange'
                for event_sample in events_list:
                    if event_sample < len(ecg_zoom_segment):
                        event_time = event_sample / sampling_rate
                        if event_time <= plot_time_axis[-1]:
                             ax4.axvline(x=event_time, color=color, linestyle='--')
                
            handles = [plt.Line2D([0], [0], marker='|', color=c, linestyle='None', markersize=10, label=l)
                       for l, c in zip(t_boundaries_labels, t_boundaries_colors)]
            ax4.legend(handles=handles, loc='upper right')
            ax4.set_title(f"{lead_name_sanitized} - Zoomed T-Wave Boundaries")
            ax4.set_xlabel("Time (s)")
            save_plot(fig4, f"{plot_prefix}_delineate_zoomed_T_boundaries", lead_name_sanitized)
            report_lines.append("- Zoomed T-Wave Boundaries plot saved.")
    else:
        report_lines.append("- Skipping T-Wave Boundaries plot: No T-onsets/offsets in zoom window.")

    # Plot 5: QRS (R-Wave) Boundaries Zoomed
    r_onsets_all = get_valid_indices(info_dict, 'ECG_R_Onsets')
    r_offsets_all = get_valid_indices(info_dict, 'ECG_R_Offsets')
    r_onsets_zoomed = get_events_in_zoom(r_onsets_all)
    r_offsets_zoomed = get_events_in_zoom(r_offsets_all)
    if r_onsets_zoomed or r_offsets_zoomed:
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.plot(plot_time_axis, ecg_zoom_segment)
        r_boundaries_events_lists = []
        r_boundaries_labels = []
        r_boundaries_colors = []
        if r_onsets_zoomed: r_boundaries_events_lists.append(r_onsets_zoomed); r_boundaries_labels.append("R Onsets (QRS Start)"); r_boundaries_colors.append("darkgreen")
        if r_offsets_zoomed: r_boundaries_events_lists.append(r_offsets_zoomed); r_boundaries_labels.append("R Offsets (QRS End)"); r_boundaries_colors.append("maroon")
        
        if r_boundaries_events_lists:
            for i, events_list in enumerate(r_boundaries_events_lists):
                color = r_boundaries_colors[i] if i < len(r_boundaries_colors) else 'green'
                for event_sample in events_list:
                     if event_sample < len(ecg_zoom_segment):
                        event_time = event_sample / sampling_rate
                        if event_time <= plot_time_axis[-1]:
                            ax5.axvline(x=event_time, color=color, linestyle='--')
                
            handles = [plt.Line2D([0], [0], marker='|', color=c, linestyle='None', markersize=10, label=l)
                       for l, c in zip(r_boundaries_labels, r_boundaries_colors)]
            ax5.legend(handles=handles, loc='upper right')
            ax5.set_title(f"{lead_name_sanitized} - Zoomed QRS (R-Wave) Boundaries")
            ax5.set_xlabel("Time (s)")
            save_plot(fig5, f"{plot_prefix}_delineate_zoomed_R_boundaries", lead_name_sanitized)
            report_lines.append("- Zoomed QRS (R-Wave) Boundaries plot saved.")
    else:
        report_lines.append("- Skipping QRS Boundaries plot: No R-onsets/offsets in zoom window.")


# --- Main Analysis Function ---
def main_ultra_comprehensive_analysis():
    npy_file_path = "/home/tony/neurokit/validation/validation01/validation01.npy" # Replace with your .npy file path
    analysis_type = "NSR"; sampling_rate = 100; preferred_lead_idx = 1; fallback_lead_idx = 0
    report_lines = []
    try:
        ecg_signal_all_leads = np.load(npy_file_path)
        report_lines.append(f"Successfully loaded ECG data from: {npy_file_path}")
    except Exception as e:
        report_lines.append(f"File load error: {str(e)} - Using dummy data."); print(report_lines[-1])
        ecg_signal_all_leads = np.array([nk.ecg_simulate(duration=30, sampling_rate=100, heart_rate=75, random_state=42)]*12) # Longer dummy data for HRV

    report_lines.append(f"ECG Signal Shape: {ecg_signal_all_leads.shape}")
    num_all_leads = ecg_signal_all_leads.shape[0] if ecg_signal_all_leads.ndim > 1 else 1
    if ecg_signal_all_leads.ndim == 1: # Handle single lead in .npy
        ecg_signal_all_leads = ecg_signal_all_leads.reshape(1, -1)
        num_all_leads = 1
        
    STANDARD_12_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    lead_to_analyze_idx = preferred_lead_idx if num_all_leads > preferred_lead_idx else fallback_lead_idx
    if num_all_leads == 0 or lead_to_analyze_idx >= num_all_leads : lead_to_analyze_idx = 0
    if num_all_leads == 0: report_lines.append("Error: No leads found."); print("\n".join(report_lines)); exit(1)
    
    lead_name = f"Lead {STANDARD_12_LEAD_NAMES[lead_to_analyze_idx]}" if num_all_leads >= lead_to_analyze_idx + 1 and lead_to_analyze_idx < len(STANDARD_12_LEAD_NAMES) else f"Lead {lead_to_analyze_idx + 1}"
    lead_name += f" (index {lead_to_analyze_idx})"
    ecg_lead_signal = ecg_signal_all_leads[lead_to_analyze_idx, :]
    lead_name_sanitized = lead_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+","")

    report_lines.insert(0, f"## Ultra Comprehensive ECG Analysis for {lead_name} ({analysis_type} focus) ##")
    report_lines.append(f"Analyzing Lead: {lead_name}\nSampling Rate: {sampling_rate} Hz\nSignal Duration: {len(ecg_lead_signal)/sampling_rate:.2f} seconds\n" + "-"*30)

    try:
        print(f"\nProcessing {lead_name} for comprehensive analysis...")
        # Use a robust peak detection method for ecg_process
        signals, info = nk.ecg_process(ecg_lead_signal, sampling_rate=sampling_rate, method='neurokit')
        rpeaks_indices = get_valid_indices(info, 'ECG_R_Peaks')
        if not rpeaks_indices: raise ValueError("No R-peaks detected in the selected lead.")

        fig0 = nk.ecg_plot(signals, info); # Remove sampling_rate parameter
        if not isinstance(fig0, plt.Figure) and plt.get_fignums(): fig0 = plt.gcf() # Get current figure if ecg_plot doesn't return one directly
        elif not isinstance(fig0, plt.Figure) : fig0, ax_temp = plt.subplots() # Create one if none and no return
        
        if isinstance(fig0, plt.Figure): # Ensure fig0 is a Figure object
             fig0.suptitle(f"Processed ECG Overview - {lead_name}", y=1.02) # Adjusted y
             save_plot(fig0, f"{analysis_type.lower()}_0_processed_ecg_overview", lead_name_sanitized)
        else:
            report_lines.append("- Failed to generate or save overview ECG plot.")

        # Generate and save the RR interval distribution plot
        interpolate = False  # Set to True to enable interpolation
        detrend = None  # Options: None, "polynomial", "tarvainen2002", "loess", "locreg"
        interpolation_rate = 100  # Hz, only used if interpolate=True
        
        rr_dist_result = plot_rr_interval_distribution(
            info, 
            rpeaks_indices,
            sampling_rate, 
            lead_name, 
            lead_name_sanitized,
            interpolate=interpolate,
            detrend=detrend,
            interpolation_rate=interpolation_rate
        )
        
        report_lines.append("\n### RR Interval Distribution ###")
        report_lines.append(f"- {rr_dist_result}")
        report_lines.append(f"  (See NSR_rr_interval_distribution_{lead_name_sanitized}.png)")
        report_lines.append("- The histogram shows the distribution of beat-to-beat intervals. In normal sinus rhythm, expect a narrow, bell-shaped distribution.")
        
        if interpolate:
            report_lines.append(f"  (See also NSR_rr_interval_timeseries_{lead_name_sanitized}.png for the time series visualization)")

        # Generate and save ECG segments plot
        ecg_segments_result = plot_ecg_segments(signals, rpeaks_indices, sampling_rate, lead_name, lead_name_sanitized)
        report_lines.append("\n### ECG Segmented Heartbeats ###")
        report_lines.append(f"- {ecg_segments_result}")
        report_lines.append(f"  (See NSR_segmented_heartbeats_{lead_name_sanitized}.png for superimposed and stacked heartbeats)")
        report_lines.append("- Superimposed view shows beat-to-beat consistency/variability with average beat in red")
        report_lines.append("- Stacked view helps visualize individual beat morphology and progression")
        report_lines.append(f"  (See also NSR_beat_to_beat_intervals_{lead_name_sanitized}.png for visualization of heart rate variability)")

        detailed_params = calculate_intervals_and_amplitudes(signals, info, rpeaks_indices, sampling_rate)

        report_lines.append("\n### A. Fiducial Points Detection ###")
        fiducial_keys = ['ECG_P_Onsets','ECG_P_Peaks','ECG_P_Offsets',
                         'ECG_Q_Peaks', # Q-peaks
                         'ECG_R_Onsets','ECG_R_Peaks','ECG_R_Offsets', # R related
                         'ECG_S_Peaks', # S-peaks
                         'ECG_T_Onsets','ECG_T_Peaks','ECG_T_Offsets'] # T related
        for key in fiducial_keys: report_lines.append(f"- Number of {key.replace('ECG_', '')} detected: {len(get_valid_indices(info, key))}")

        report_lines.append("\n### B. Intervals and Durations ###")
        interval_report_keys = {
            'RR Mean (ms)':('RR_mean_ms','RR_std_ms','RR_intervals_ms'),
            'PP Mean (ms)':('PP_mean_ms','PP_std_ms','PP_intervals_ms'),
            'PR Mean (ms)':('PR_mean_ms','PR_std_ms','PR_intervals_ms'),
            'QRS Mean (ms)':('QRS_mean_ms','QRS_std_ms','QRS_durations_ms'),
            'QT Mean (ms)':('QT_mean_ms','QT_std_ms','QT_intervals_ms'),
            'QTc Mean (Bazett)':('QTc_mean','QTc_std','QTc_intervals') # Specify Bazett if used
        }
        for disp_key, (mean_k, std_k, list_k) in interval_report_keys.items():
            mean_val = detailed_params.get(mean_k)
            std_val = detailed_params.get(std_k)
            list_val = detailed_params.get(list_k, [])
            if mean_val is not None and len(list_val) > 0:
                unit = "ms" if "ms" in disp_key else ""
                std_dev_str = f"{std_val:.2f}{unit}" if std_val is not None else "N/A"
                report_lines.append(f"- {disp_key}: {mean_val:.2f}{unit} (StdDev: {std_dev_str}, N: {len(list_val)})")
            else: report_lines.append(f"- {disp_key}: Not reliably calculated or no valid intervals found.")
        
        report_lines.append("\n### C. Amplitudes (from Cleaned ECG) ###")
        if 'Amplitudes_mV' in detailed_params and detailed_params['Amplitudes_mV']:
            for amp_key_mean in detailed_params['Amplitudes_mV']:
                if '_amp_mean' in amp_key_mean:
                    wave = amp_key_mean.split('_')[0]
                    mean_amp = detailed_params['Amplitudes_mV'][amp_key_mean]
                    std_amp_key = f'{wave}_peak_amp_std'
                    std_amp = detailed_params['Amplitudes_mV'].get(std_amp_key, np.nan)
                    amp_values_key = f'{wave}_peak_amps'
                    num_amps = len(detailed_params['Amplitudes_mV'].get(amp_values_key,[]))
                    std_amp_str = f"{std_amp:.3f}" if pd.notna(std_amp) else "N/A"
                    report_lines.append(f"- {wave} Peak Amp: Mean={mean_amp:.3f}, Std={std_amp_str} (N={num_amps}) arbitrary units")
        else: report_lines.append("- Waveform amplitudes not calculated.")

        report_lines.append("\n### D. ST-Segment Analysis (Basic) ###")
        st_mean_dev = detailed_params.get('ST_mean_deviation_mV')
        st_std_dev = detailed_params.get('ST_std_deviation_mV')
        st_values = detailed_params.get('ST_deviations_mV', [])
        if st_mean_dev is not None and len(st_values) > 0:
            st_std_str = f"{st_std_dev:.3f}" if st_std_dev is not None else "N/A"
            report_lines.append(f"- ST-Segment Mean Deviation (J+60ms from PR baseline): {st_mean_dev:.3f} arbitrary units")
            report_lines.append(f"- ST-Segment Std Deviation: {st_std_str} (N: {len(st_values)})")
        else: report_lines.append("- ST-Segment deviation: Not reliably calculated.")
        report_lines.append("- Note: T-wave/U-wave morphology requires visual assessment (not quantified beyond T-peak amplitude).")

        report_lines.append("\n### E. Axis Information ###")
        qrs_axis_str = "Not calculated."
        if num_all_leads >= 6: # Need at least Lead I and aVF
            try:
                # Ensure STANDARD_12_LEAD_NAMES matches the actual lead order if it's truly 12-lead
                # For this example, assuming direct indexing if STANDARD_12_LEAD_NAMES is used
                lead_I_idx = STANDARD_12_LEAD_NAMES.index("I") if "I" in STANDARD_12_LEAD_NAMES else 0
                lead_aVF_idx = STANDARD_12_LEAD_NAMES.index("aVF") if "aVF" in STANDARD_12_LEAD_NAMES else 5 # Common position

                if lead_I_idx < num_all_leads and lead_aVF_idx < num_all_leads:
                    sig_I_proc, info_I = nk.ecg_process(ecg_signal_all_leads[lead_I_idx,:], sampling_rate)
                    cleaned_I = sig_I_proc["ECG_Clean"].values
                    r_I = get_valid_indices(info_I,'ECG_R_Peaks'); q_I = get_valid_indices(info_I,'ECG_Q_Peaks'); s_I_p = get_valid_indices(info_I,'ECG_S_Peaks')
                    
                    net_I_amp = 0
                    if r_I: net_I_amp += np.sum(cleaned_I[r_I])
                    if q_I: net_I_amp += np.sum(cleaned_I[q_I]) # Q is negative
                    if s_I_p: net_I_amp += np.sum(cleaned_I[s_I_p]) # S is negative
                    # More accurate: R - abs(Q) - abs(S) or sum of peak values if they are signed
                    # For simplicity using median of peak values
                    median_r_I = np.median(cleaned_I[r_I]) if r_I else 0
                    median_q_I = np.median(cleaned_I[q_I]) if q_I else 0
                    median_s_I = np.median(cleaned_I[s_I_p]) if s_I_p else 0
                    net_I = median_r_I + median_q_I + median_s_I # Q and S should be negative if true peaks

                    sig_aVF_proc, info_aVF = nk.ecg_process(ecg_signal_all_leads[lead_aVF_idx,:], sampling_rate)
                    cleaned_aVF = sig_aVF_proc["ECG_Clean"].values
                    r_aVF = get_valid_indices(info_aVF,'ECG_R_Peaks'); q_aVF = get_valid_indices(info_aVF,'ECG_Q_Peaks'); s_aVF_p = get_valid_indices(info_aVF,'ECG_S_Peaks')
                    median_r_aVF = np.median(cleaned_aVF[r_aVF]) if r_aVF else 0
                    median_q_aVF = np.median(cleaned_aVF[q_aVF]) if q_aVF else 0
                    median_s_aVF = np.median(cleaned_aVF[s_aVF_p]) if s_aVF_p else 0
                    net_aVF = median_r_aVF + median_q_aVF + median_s_aVF
                    
                    if not (r_I or q_I or s_I_p) and not (r_aVF or q_aVF or s_aVF_p): qrs_axis_str = "Indeterminate (No QRS in Lead I & aVF)."
                    elif abs(net_I)<1e-5 and abs(net_aVF)<1e-5: qrs_axis_str = "Indeterminate (Net QRS in Lead I & aVF near zero)."
                    else: qrs_axis_str = f"{np.degrees(np.arctan2(net_aVF, net_I)):.1f} degrees"
                    report_lines.append(f"- QRS Axis (Median peak deflections in Lead I & aVF): {qrs_axis_str}")
                else: report_lines.append("- QRS Axis: Not attempted (Lead I or aVF index out of bounds).")
            except Exception as e_axis: report_lines.append(f"- QRS Axis Calc Error: {str(e_axis)}")
        else: report_lines.append("- QRS Axis: Not attempted (needs >= 6 leads, or Leads I and aVF).")
        report_lines.append("- P-wave axis and T-wave axis: Not implemented in this script.")

        report_lines.append("\n### F. Heart Rate Variability (HRV) & Rate ###")
        avg_hr_from_rr = detailed_params.get('RR_mean_ms')
        if avg_hr_from_rr and avg_hr_from_rr > 0:
            avg_hr = 60000 / avg_hr_from_rr
            report_lines.append(f"- Average Heart Rate (from RR intervals): {avg_hr:.2f} bpm")
        else: report_lines.append("- Average Heart Rate: Not calculated from RR intervals.")
        
        if len(rpeaks_indices) > 10: # Min beats for more reliable HRV, esp. freq domain
            try:
                hrv_summary = nk.hrv(rpeaks_indices, sampling_rate=sampling_rate, show=False)
                if not hrv_summary.empty:
                    report_lines.append("  **HRV Metrics (from nk.hrv()):**")
                    report_lines.append("  (Note: Frequency-domain & complex non-linear metrics need longer, stable recordings.)")
                    
                    hrv_cols_time = [c for c in hrv_summary.columns if any(sub.upper() in c.upper() for sub in ['MeanNN','SDNN','RMSSD','pNN','CVNN','MedianNN','MadNN','MCVNN','IQRNN','SDRMSSD','Prc','MinNN','MaxNN','TINN','HTI','SDSD']) and "ECG" not in c and "RSP" not in c]
                    hrv_cols_freq = [c for c in hrv_summary.columns if any(sub.upper() in c.upper() for sub in ['ULF','VLF','LF','HF','VHF','TP','LFHF','LFn','HFn','LnHF']) and "ECG" not in c and "RSP" not in c]
                    hrv_cols_nonlin = [c for c in hrv_summary.columns if c not in hrv_cols_time and c not in hrv_cols_freq and "ECG" not in c and "RSP" not in c and "RSA" not in c and "HRV" in c.upper()]


                    if hrv_cols_time: report_lines.append("    Time-Domain:")
                    for col in hrv_cols_time: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")
                    
                    if hrv_cols_freq: report_lines.append("    Frequency-Domain:")
                    for col in hrv_cols_freq: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")
                    
                    if hrv_cols_nonlin: report_lines.append("    Non-Linear / Other (HRV specific):")
                    for col in hrv_cols_nonlin: report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.3f}")

                    # RQA from hrv_nonlinear if available (hrv() might include some)
                    if any("RQA" in c.upper() for c in hrv_summary.columns):
                         report_lines.append("    Recurrence Quantification Analysis (RQA - from hrv summary):")
                         for col in [c for c in hrv_summary.columns if "RQA" in c.upper()]:
                             report_lines.append(f"      {col}: {hrv_summary[col].iloc[0]:.4f}")
                    else: # Try separately if not in hrv() output
                        try:
                            hrv_rqa_results = nk.hrv_rqa(rpeaks_indices, sampling_rate=sampling_rate, show=False)
                            if hrv_rqa_results is not None and not hrv_rqa_results.empty:
                                report_lines.append("    Recurrence Quantification Analysis (RQA - separate call):")
                                for col_rqa in hrv_rqa_results.columns:
                                    report_lines.append(f"      {col_rqa}: {hrv_rqa_results[col_rqa].iloc[0]:.4f}")
                        except Exception as e_rqa_sep:
                            report_lines.append(f"    Separate RQA calculation failed: {str(e_rqa_sep)}")


                    if 'HRV_SD1' in hrv_summary.columns and 'HRV_SD2' in hrv_summary.columns and 'RR_intervals_ms' in detailed_params and detailed_params['RR_intervals_ms'] is not None and len(detailed_params['RR_intervals_ms']) > 1:
                        # Attempt to create Poincare plot
                        fig_poincare_attempt, ax_poincare_attempt = plt.subplots(figsize=(6,6))
                        try:
                            # Check if hrv_plot exists in neurokit2
                            if hasattr(nk, 'hrv_plot'):
                                nk.hrv_plot(hrv_summary, ax=ax_poincare_attempt, plot_type="poincare")
                                ax_poincare_attempt.set_title(f"Poincaré Plot - {lead_name}")
                                save_plot(fig_poincare_attempt, f"{analysis_type.lower()}_1_poincare", lead_name_sanitized, tight_layout=False)
                                report_lines.append(f"  (Poincaré plot saved using hrv_plot)")
                            else:
                                # Manual Poincaré plot as fallback
                                rr_intervals = detailed_params['RR_intervals_ms']
                                rr_n = rr_intervals[:-1]  # RR(n)
                                rr_n1 = rr_intervals[1:]  # RR(n+1)
                                
                                # Plot points
                                ax_poincare_attempt.scatter(rr_n, rr_n1, alpha=0.75, c='blue')
                                
                                # Get SD1 and SD2 from HRV
                                sd1 = hrv_summary['HRV_SD1'].iloc[0]
                                sd2 = hrv_summary['HRV_SD2'].iloc[0]
                                
                                # Identity line
                                min_rr_val = min(min(rr_n), min(rr_n1)) if len(rr_n)>0 and len(rr_n1)>0 else 0
                                max_rr_val = max(max(rr_n), max(rr_n1)) if len(rr_n)>0 and len(rr_n1)>0 else 1
                                ax_poincare_attempt.plot([min_rr_val, max_rr_val], [min_rr_val, max_rr_val], 'k--', alpha=0.5)
                                
                                # Mean RR
                                mean_rr_val = np.mean(rr_intervals)
                                ax_poincare_attempt.plot(mean_rr_val, mean_rr_val, 'ro')
                                
                                # Set labels and title
                                ax_poincare_attempt.set_xlabel('RR(n) (ms)')
                                ax_poincare_attempt.set_ylabel('RR(n+1) (ms)')
                                ax_poincare_attempt.set_title(f"Poincaré Plot - {lead_name}\nSD1: {sd1:.2f}, SD2: {sd2:.2f}")
                                
                                save_plot(fig_poincare_attempt, f"{analysis_type.lower()}_1_poincare", lead_name_sanitized, tight_layout=False)
                                report_lines.append(f"  (Poincaré plot created manually as fallback)")
                        except Exception as e_hrv_plot:
                            if plt.fignum_exists(fig_poincare_attempt.number): plt.close(fig_poincare_attempt) # Close if plotting failed
                            report_lines.append(f"  (Poincaré plot creation failed: {str(e_hrv_plot)})")
                else:
                    report_lines.append("  HRV summary from nk.hrv() was empty.")

            except Exception as e_hrv: report_lines.append(f"  HRV Analysis failed/incomplete: {str(e_hrv)}")
        else: report_lines.append("  Not enough R-peaks for full HRV analysis (need > 10).")
        report_lines.append(f"- Rhythm Type: (Requires interpretation of overall parameters).")
        report_lines.append("- RSA (Respiratory Sinus Arrhythmia): Requires RSP signal, not calculated here.")

        report_lines.append("\n### G. Signal Quality ###")
        if 'ECG_Quality' in signals.columns:
            mean_quality = signals['ECG_Quality'].mean()
            report_lines.append(f"- Mean ECG Quality (NeuroKit's method): {mean_quality:.3f} (0-1, relative to avg morphology)")
        else: report_lines.append("- ECG Quality metric not in signals DataFrame.")

        report_lines.append("\n### H. Cardiac Phase Information ###")
        phase_keys = ['ECG_Phase_Atrial','ECG_Phase_Completion_Atrial','ECG_Phase_Ventricular','ECG_Phase_Completion_Ventricular']
        plotted_phase = False
        # Determine number of rows for subplot based on available keys
        num_phase_plots = sum(1 for key in phase_keys if key in signals.columns)
        if num_phase_plots > 0:
            fig_phase, ax_phase_arr = plt.subplots(num_phase_plots, 1, figsize=(15, 2 * num_phase_plots), sharex=True, squeeze=False) # squeeze=False ensures ax_phase_arr is always 2D
            ax_idx = 0
            time_axis_full = np.arange(len(signals)) / sampling_rate
            segment_end_sample = min(len(signals), int(10 * sampling_rate)) # Show up to 10s for phase

            for key in phase_keys:
                if key in signals.columns:
                    series_to_plot = signals[key].iloc[:segment_end_sample]
                    report_lines.append(f"- Mean {key.replace('ECG_Phase_','')}: {series_to_plot.mean():.3f} (over first {segment_end_sample/sampling_rate:.1f}s)")
                    current_ax = ax_phase_arr[ax_idx, 0]
                    current_ax.plot(time_axis_full[:segment_end_sample], series_to_plot, label=key)
                    current_ax.set_ylabel(key.replace('ECG_Phase_','').replace('_',' '))
                    
                    if ax_idx == 0: # Add scaled ECG to the first phase plot for context
                        ecg_clean_segment = signals['ECG_Clean'].iloc[:segment_end_sample].values
                        if len(ecg_clean_segment) > 0 and np.any(np.isfinite(ecg_clean_segment)) and np.any(np.isfinite(series_to_plot)):
                             # Ensure target range for rescale is valid
                             target_min = series_to_plot.min()
                             target_max = series_to_plot.max()
                             if np.isfinite(target_min) and np.isfinite(target_max) and target_min < target_max:
                                 current_ax.plot(time_axis_full[:segment_end_sample],
                                              nk.rescale(ecg_clean_segment, to=[target_min, target_max]),
                                              label='ECG (scaled)',color='gray',alpha=0.5)
                             else: # Fallback if target range is invalid (e.g. phase is flat)
                                 current_ax.plot(time_axis_full[:segment_end_sample], ecg_clean_segment, label='ECG (raw - phase flat)', color='lightgray', alpha=0.4)
                        current_ax.legend(loc='center right') # Adjust legend position
                    else:
                        current_ax.legend(loc='upper right') # Default for other plots
                    plotted_phase = True
                    ax_idx += 1
            
            if plotted_phase:
                ax_phase_arr[-1, 0].set_xlabel("Time (s)")
                fig_phase.suptitle(f"Cardiac Phase Signals - {lead_name}", y=0.99) # y slightly adjusted if titles overlap
                save_plot(fig_phase, f"{analysis_type.lower()}_2_cardiac_phase", lead_name_sanitized)
                report_lines.append(f"  (Cardiac phase plot saved for first {segment_end_sample/sampling_rate:.1f}s)")
            else:
                if 'fig_phase' in locals() and plt.fignum_exists(fig_phase.number): plt.close(fig_phase) # Close if no phase plots were made
        else:
            report_lines.append("- No cardiac phase columns found in signals DataFrame.")


        report_lines.append("\n### I. Representative Heartbeats ###")
        try:
            # Ensure 'ECG_Clean' is used for segmentation
            epochs = nk.ecg_segment(signals['ECG_Clean'], rpeaks=rpeaks_indices, sampling_rate=sampling_rate, show=False)
            valid_epochs_list = [df['Signal'].values for k, df in epochs.items() if df is not None and 'Signal' in df and not df['Signal'].isnull().all()]
            
            if valid_epochs_list:
                fig_beats, ax_beats = plt.subplots(figsize=(10,6))
                # Determine common length by padding/truncating to median length for averaging
                lengths = [len(s) for s in valid_epochs_list if s is not None]
                if not lengths: raise ValueError("No valid epoch lengths.")
                common_len = int(np.median(lengths))
                if common_len <=0: raise ValueError("Median epoch length is zero or negative.")
                
                time_axis_beats = np.arange(common_len)/sampling_rate # Time axis for segmented beats
                
                all_beat_signals_for_mean = []
                for i, signal_data in enumerate(valid_epochs_list):
                    if signal_data is None or len(signal_data) == 0: continue # Skip empty signals
                    if len(signal_data) != common_len:
                        signal_data_interp = np.interp(np.linspace(0, 1, common_len), np.linspace(0, 1, len(signal_data)), signal_data)
                    else:
                        signal_data_interp = signal_data
                    all_beat_signals_for_mean.append(signal_data_interp)
                    if i < 10: # Plot first 10 beats
                        ax_beats.plot(time_axis_beats, signal_data_interp, alpha=0.5, label=f"Beat {i+1}" if i < 3 else None)
                
                if all_beat_signals_for_mean:
                    mean_beat_array = np.array(all_beat_signals_for_mean)
                    if mean_beat_array.ndim == 2 and mean_beat_array.shape[0] > 0:
                        mean_beat = np.nanmean(mean_beat_array, axis=0)
                        if not np.all(np.isnan(mean_beat)):
                             ax_beats.plot(time_axis_beats, mean_beat, color='black', linewidth=2.5, label=f'Mean Beat (N={len(all_beat_signals_for_mean)})')
                
                ax_beats.set_title(f"Overlay of Segmented Heartbeats - {lead_name}"); ax_beats.set_xlabel("Time from R-peak (s)"); ax_beats.set_ylabel("Amplitude (cleaned units)")
                ax_beats.legend(); save_plot(fig_beats, f"{analysis_type.lower()}_3_segmented_beats", lead_name_sanitized)
                report_lines.append(f"  (Segmented heartbeats plot saved, N={len(all_beat_signals_for_mean)})")
            else: report_lines.append("- Could not generate segmented heartbeats plot (no valid epochs).")
        except Exception as e_segment: 
            report_lines.append(f"- Error during heartbeat segmentation for representative plot: {str(e_segment)}")
            if 'fig_beats' in locals() and plt.fignum_exists(fig_beats.number): plt.close(fig_beats)


        # Call the new delineation plots function
        generate_delineation_plots(signals, info, rpeaks_indices, sampling_rate, 
                                   f"{analysis_type.lower()}_4", lead_name_sanitized, report_lines)


        if analysis_type == "NSR":
            report_lines.append("\n" + "="*30 + "\n### Clinical Summary Suggestion (NSR Focus) ###")
            report_lines.append("This detailed report provides extensive parameters for a Normal Sinus Rhythm assessment.")
            report_lines.append("Key NSR indicators to check from above sections:")
            report_lines.append("  - Rate: Approx 60-100 bpm (Section F)")
            report_lines.append("  - Rhythm: Check regularity of RR & PP intervals, HRV_SDNN/RMSSD (Section B & F)")
            report_lines.append("  - P-waves: Present before each QRS, consistent morphology (Section A, C, I, J)")
            report_lines.append("  - P:QRS Ratio: Approx 1:1 (Section A - compare P_Peaks to R_Peaks count)")
            report_lines.append("  - PR Interval: Approx 120-200ms, constant (Section B)")
            report_lines.append("  - QRS Duration: Approx <120ms (Section B)")
            report_lines.append("  - QRS Axis: Normal range (e.g., -30 to +90 degrees) (Section E)")
            report_lines.append("  - QT/QTc: Within normal limits (Section B)")
            report_lines.append("Please correlate with full clinical picture. This is not a diagnostic tool.")
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