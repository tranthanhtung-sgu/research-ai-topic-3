#!/usr/bin/env python3
"""
Test script for the ECG demo backend
"""
import os
import sys
import numpy as np
from backend import inference, eda, report

def main():
    """
    Test the backend functionality
    """
    print("Testing ECG demo backend...")
    
    # Create a simple synthetic ECG signal
    print("Creating synthetic ECG signal...")
    samples = 5000
    t = np.linspace(0, 10, samples)
    
    # Create 12 leads of synthetic ECG
    ecg_data = np.zeros((12, samples))
    for i in range(12):
        # Base signal (QRS complex simulation)
        signal = np.sin(2 * np.pi * 1.2 * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, samples)
        
        # Add some baseline wander
        baseline = 0.2 * np.sin(2 * np.pi * 0.05 * t)
        
        # Combine
        ecg_data[i] = signal + noise + baseline
    
    # Test inference
    print("\nTesting inference module...")
    try:
        condition, fs, detailed_results = inference.predict(ecg_data, "test_sample")
        print(f"Predicted condition: {condition}")
        print(f"Sampling rate: {fs}")
        print("Detailed results:")
        print(detailed_results)
    except Exception as e:
        print(f"Error in inference: {e}")
        # Use default values for testing the rest of the pipeline
        condition = "NORMAL (Confidence: 0.95)"
        fs = 500
        detailed_results = "Detailed results not available"
    
    # Test visualization
    print("\nTesting visualization module...")
    try:
        figures = eda.make_figs(ecg_data, fs)
        print(f"Generated {len(figures)} figures")
    except Exception as e:
        print(f"Error in visualization: {e}")
        # Create a dummy figure for testing report generation
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("Dummy ECG")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        figures = [Image.open(buf)]
        plt.close(fig)
    
    # Test report generation
    print("\nTesting report generation...")
    try:
        # Use placeholder API key
        api_key = "iusehdfiuhsdiufhidsiashdu"
        report_text = report.write(condition, figures, api_key)
        print(f"Generated report with {len(report_text)} characters")
        print("Report preview:")
        print(report_text[:200] + "...")
    except Exception as e:
        print(f"Error in report generation: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 