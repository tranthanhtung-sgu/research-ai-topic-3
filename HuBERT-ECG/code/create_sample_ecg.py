#!/usr/bin/env python3
"""
Script to create a sample ECG file for testing
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def create_synthetic_ecg(samples=5000, num_leads=12, save_path="sample_ecg.npy"):
    """
    Create a synthetic ECG signal and save it as a .npy file
    
    Args:
        samples: Number of samples
        num_leads: Number of leads
        save_path: Path to save the .npy file
    """
    print(f"Creating synthetic ECG with {num_leads} leads and {samples} samples...")
    
    # Create time array
    t = np.linspace(0, 10, samples)
    
    # Create ECG data array
    ecg_data = np.zeros((num_leads, samples))
    
    # Generate synthetic ECG for each lead
    for i in range(num_leads):
        # Base signal (QRS complex simulation)
        signal = np.sin(2 * np.pi * 1.2 * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, samples)
        
        # Add some baseline wander
        baseline = 0.2 * np.sin(2 * np.pi * 0.05 * t)
        
        # Combine
        ecg_data[i] = signal + noise + baseline
    
    # Save as .npy file
    np.save(save_path, ecg_data)
    print(f"Saved synthetic ECG to {save_path}")
    
    # Plot the first lead
    plt.figure(figsize=(10, 4))
    plt.plot(t, ecg_data[0])
    plt.title("Synthetic ECG (Lead I)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Save plot
    plot_path = save_path.replace(".npy", ".png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    return ecg_data

def create_batch_ecg(num_samples=5, samples=5000, num_leads=12, save_path="batch_ecg.npy"):
    """
    Create a batch of synthetic ECG signals and save it as a .npy file
    
    Args:
        num_samples: Number of ECG samples in the batch
        samples: Number of samples per ECG
        num_leads: Number of leads
        save_path: Path to save the .npy file
    """
    print(f"Creating batch of {num_samples} synthetic ECGs...")
    
    # Create ECG data array
    batch_data = np.zeros((num_samples, num_leads, samples))
    
    # Generate synthetic ECG for each sample
    for j in range(num_samples):
        # Create time array
        t = np.linspace(0, 10, samples)
        
        # Generate synthetic ECG for each lead
        for i in range(num_leads):
            # Base signal (QRS complex simulation)
            signal = np.sin(2 * np.pi * 1.2 * t)
            
            # Add some noise
            noise = np.random.normal(0, 0.1, samples)
            
            # Add some baseline wander
            baseline = 0.2 * np.sin(2 * np.pi * 0.05 * t)
            
            # Combine
            batch_data[j, i] = signal + noise + baseline
    
    # Save as .npy file
    np.save(save_path, batch_data)
    print(f"Saved batch of synthetic ECGs to {save_path}")
    
    return batch_data

if __name__ == "__main__":
    # Create directory for sample data if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Create a single ECG
    create_synthetic_ecg(save_path="sample_data/single_ecg.npy")
    
    # Create a batch of ECGs
    create_batch_ecg(num_samples=5, save_path="sample_data/batch_ecg.npy")
    
    print("Done!") 