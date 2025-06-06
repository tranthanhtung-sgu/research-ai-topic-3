#!/usr/bin/env python3
"""
Test script to validate ECG samples in the validation04 folder
using the predict function from validate_six_conditions.py
"""
import os
import sys
import numpy as np
import glob
import json
from pathlib import Path

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the predict function from validate_six_conditions
from validate_six_conditions import predict

def test_validation04_folder(validation_folder="./validation/validation04", 
                            output_file="./validation04_results.json",
                            model_path=None):
    """
    Test all samples in the validation04 folder
    
    Args:
        validation_folder: Path to the validation04 folder
        output_file: Path to save the results
        model_path: Path to the model file (if None, will use default)
    """
    # Ensure validation folder exists
    if not os.path.exists(validation_folder):
        print(f"Validation folder {validation_folder} does not exist")
        return
    
    # Get all .npy files in the folder
    npy_files = glob.glob(os.path.join(validation_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {validation_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files in {validation_folder}")
    
    results = {}
    
    for npy_file in npy_files:
        sample_name = os.path.basename(npy_file).split('.')[0]
        print(f"\nProcessing {sample_name}...")
        
        try:
            # Load the ECG data
            ecg_data = np.load(npy_file)
            
            # Print diagnostic information about the data
            print(f"Data shape: {ecg_data.shape}")
            print(f"Data type: {ecg_data.dtype}")
            print(f"Data min/max: {ecg_data.min():.4f}/{ecg_data.max():.4f}")
            print(f"Data mean/std: {ecg_data.mean():.4f}/{ecg_data.std():.4f}")
            
            # Use the predict function with the specified model path
            prediction, sampling_rate, detailed_results = predict(ecg_data, sample_name, model_path)
            
            # Store results
            results[sample_name] = {
                "prediction": prediction,
                "sampling_rate": sampling_rate,
                "detailed_results": detailed_results
            }
            
            # Print the prediction
            print(f"Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results if we have any
    if results:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        print("\nSummary of predictions:")
        for sample_name, result in results.items():
            print(f"  {sample_name}: {result['prediction']}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test validation04 folder using predict function")
    parser.add_argument("--validation_folder", type=str, default="./validation/validation04",
                        help="Path to the validation04 folder")
    parser.add_argument("--output_file", type=str, default="./validation04_results.json",
                        help="Output file for validation results")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model file (if not specified, will use default)")
    args = parser.parse_args()
    
    # Check if model path exists if specified
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        sys.exit(1)
    
    test_validation04_folder(args.validation_folder, args.output_file, args.model_path) 