#!/usr/bin/env python3
"""
Script to validate ECG samples using the six conditions HuBERT-ECG model
with enhanced preprocessing including heartbeat outlier removal
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob
import json

# Add the code directory to the path
sys.path.append('/home/tony/neurokit/HuBERT-ECG/code')

from hubert_ecg_classification import HuBERTForECGClassification
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from preprocessing import ECGPreprocessor, create_preprocessor_config
from ecg_outlier_removal import heartbeat_outlier_removal

def load_six_conditions_model(model_path, device='cpu'):
    """
    Load the six conditions model
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Get the conditions list from the checkpoint
    conditions = checkpoint.get('conditions')
    
    if conditions is None:
        # Default six conditions
        conditions = [
            'NORMAL',
            'Atrial Fibrillation',
            'Atrial Flutter',
            'Left bundle branch block',
            'Right bundle branch block',
            '1st-degree AV block'
        ]
    
    print(f"Model trained on conditions: {conditions}")
    
    # Check if this is the anti-overfitting model
    is_anti_overfitting = 'anti_overfitting' in model_path
    
    # Create a config based on model type
    if is_anti_overfitting:
        print("Using reduced complexity model configuration")
        config = HuBERTECGConfig(
            hidden_size=384,  # Reduced from 512
            num_hidden_layers=6,  # Reduced from 8
            num_attention_heads=6,  # Reduced from 8
            intermediate_size=1536,  # Reduced from 2048
            hidden_act="gelu",
            hidden_dropout_prob=0.2,  # Increased from 0.1
            attention_probs_dropout_prob=0.2,  # Increased from 0.1
            layer_norm_eps=1e-12,
            feature_extractor_type="default",
            feature_extractor_channels=384,  # Reduced from 512
            feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],
            feature_extractor_strides=[2, 2, 2, 2, 2]
        )
    else:
        print("Using standard model configuration")
        config = HuBERTECGConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            feature_extractor_type="default",
            feature_extractor_channels=512,
            feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],  # Smaller kernel sizes
            feature_extractor_strides=[2, 2, 2, 2, 2]  # More conservative strides
        )
    
    # Create the base model
    hubert_ecg = HuBERTECG(config)
    
    # Create the classification model with appropriate parameters
    if is_anti_overfitting:
        model = HuBERTForECGClassification(
            hubert_ecg_model=hubert_ecg,
            num_labels=len(conditions),
            classifier_hidden_size=384,  # Reduced from 512
            dropout=0.3,  # Increased from 0.1
            pooling_strategy='mean'
        )
    else:
        model = HuBERTForECGClassification(
            hubert_ecg_model=hubert_ecg,
            num_labels=len(conditions),
            classifier_hidden_size=512,
            dropout=0.1,
            pooling_strategy='mean'
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model, conditions

def preprocess_ecg(ecg_data, target_length=5000):
    """
    Preprocess ECG data for the model with enhanced outlier removal
    """
    # Create preprocessor with heartbeat outlier removal enabled
    config = create_preprocessor_config('mimic_physionet')
    config['remove_heartbeat_outliers'] = True
    preprocessor = ECGPreprocessor(**config)
    
    print("Using enhanced preprocessing with heartbeat outlier removal")
    
    # Get original shape
    original_shape = ecg_data.shape
    
    # Make sure it's float32
    ecg_data = ecg_data.astype(np.float32)
    
    # Apply heartbeat outlier removal directly if data is not resampled
    if original_shape[1] == target_length:
        try:
            # First apply heartbeat outlier removal
            ecg_data = heartbeat_outlier_removal(ecg_data)
            print("Applied heartbeat outlier removal preprocessing")
        except Exception as e:
            print(f"Warning: Error in heartbeat outlier removal: {e}")
    
    # Resample if needed to match our target length
    if original_shape[1] != target_length:
        # Simple linear interpolation
        from scipy.interpolate import interp1d
        x_original = np.linspace(0, 1, original_shape[1])
        x_target = np.linspace(0, 1, target_length)
        
        resampled_data = np.zeros((original_shape[0], target_length), dtype=np.float32)
        for i in range(original_shape[0]):
            f = interp1d(x_original, ecg_data[i, :])
            resampled_data[i, :] = f(x_target)
        
        ecg_data = resampled_data
        
        # Apply heartbeat outlier removal after resampling
        try:
            ecg_data = heartbeat_outlier_removal(ecg_data)
            print("Applied heartbeat outlier removal after resampling")
        except Exception as e:
            print(f"Warning: Error in heartbeat outlier removal after resampling: {e}")
    
    # Preprocess signal
    processed_signal, attention_mask = preprocessor.preprocess_signal(ecg_data)
    
    return processed_signal, attention_mask

def predict(ecg_data: np.ndarray, sample_name="Sample", model_path: str = None):
    """
    Predict ECG condition using HuBERT-ECG model.
    
    Args:
        ecg_data: ECG data as numpy array. Expected (leads, samples) or compatible.
        sample_name: Name of the sample for display.
        model_path: Optional path to the model file. If None, tries default locations.
        
    Returns:
        Tuple: (top_prediction_string, sampling_rate (fixed at 500), detailed_results_table)
    """
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, conditions = load_six_conditions_model(model_path or "./outputs/six_conditions_model_anti_overfitting_best/best_model_accuracy.pt", device)
        
        # Ensure data is in the right format (12 leads, samples)
        if ecg_data.ndim == 1:
            # Single lead data, reshape to (1, samples)
            ecg_data = ecg_data.reshape(1, -1)
        elif ecg_data.ndim == 2:
            # Check if we need to transpose
            if ecg_data.shape[0] > ecg_data.shape[1]:
                # Likely (samples, leads), transpose to (leads, samples)
                ecg_data = ecg_data.T
        
        # Handle if fewer than 12 leads are provided
        if ecg_data.shape[0] < 12:
            # Pad with zeros for missing leads
            padding = np.zeros((12 - ecg_data.shape[0], ecg_data.shape[1]))
            ecg_data = np.vstack([ecg_data, padding])
        elif ecg_data.shape[0] > 12:
            # Take first 12 leads
            ecg_data = ecg_data[:12]
        
        # Ensure data is float32
        ecg_data = ecg_data.astype(np.float32)
        
        # Run prediction
        prediction = predict_sample(model, conditions, ecg_data, device)
        
        if prediction is None:
            return "Error in prediction", 500, "Error: Failed to make prediction"
        
        # Format results
        sorted_predictions = sorted(prediction.items(), key=lambda x: x[1]['probability'], reverse=True)
        top_condition = sorted_predictions[0]
        condition_name = top_condition[0]
        probability = top_condition[1]['probability']
        
        # Format the result
        result = f"{condition_name} (Confidence: {probability:.2f})"
        
        # Create detailed results table
        output_lines = []
        output_lines.append(f"\nPrediction Results for {sample_name}:")
        output_lines.append("-" * 50)
        output_lines.append(f"{'Condition':<30} {'Probability':<15} {'Prediction'}")
        output_lines.append("-" * 50)
        
        for condition, result_data in sorted_predictions:
            prob = result_data['probability']
            pred = result_data['prediction']
            output_lines.append(f"{condition:<30} {prob:.4f}         {'✓' if pred else '✗'}")
        
        detailed_results = "\n".join(output_lines)
        
        # Print detailed results to console
        print(detailed_results)
        
        # Return the predicted condition, sampling rate, and detailed results
        return result, 500, detailed_results
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500, f"Error: {str(e)}"

def predict_sample(model, conditions, ecg_data, device='cpu'):
    """
    Predict the condition for an ECG sample
    """
    # Preprocess the ECG data
    processed_signal, attention_mask = preprocess_ecg(ecg_data)
    
    # Convert to tensors and add batch dimension
    signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0).to(device)
    attention_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        try:
            outputs = model(signal_tensor, attention_mask_tensor)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)
            
            # Get predictions
            predictions = (probs > 0.5).float()
            
            # Convert to numpy
            probs_np = probs.cpu().numpy()[0]
            preds_np = predictions.cpu().numpy()[0]
            
            # Create result dictionary
            results = {}
            for i, condition in enumerate(conditions):
                results[condition] = {
                    'probability': float(probs_np[i]),
                    'prediction': bool(preds_np[i] > 0.4)
                }
            
            return results
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

def validate_folder(model, conditions, validation_folder, device='cpu'):
    """
    Validate all samples in a folder
    """
    # Get all .npy files in the folder
    npy_files = glob.glob(os.path.join(validation_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {validation_folder}")
        return {}
    
    results = {}
    
    for npy_file in npy_files:
        sample_name = os.path.basename(npy_file).split('.')[0]
        print(f"Processing {sample_name}...")
        
        # Load the ECG data
        ecg_data = np.load(npy_file)
        
        # Predict
        prediction = predict_sample(model, conditions, ecg_data, device)
        
        if prediction:
            results[sample_name] = prediction
            
            # Print results
            print(f"\nPrediction Results for {sample_name}:")
            print("-" * 50)
            print(f"{'Condition':<30} {'Probability':<15} {'Prediction'}")
            print("-" * 50)
            
            # Sort by probability
            sorted_predictions = sorted(prediction.items(), key=lambda x: x[1]['probability'], reverse=True)
            
            for condition, result in sorted_predictions:
                prob = result['probability']
                pred = result['prediction']
                print(f"{condition:<30} {prob:.4f}         {'✓' if pred else '✗'}")
    
    return results

def validate_single_file(model, conditions, file_path, device='cpu'):
    """
    Validate a single ECG file
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return {}
    
    if not file_path.endswith('.npy'):
        print(f"File {file_path} is not a .npy file")
        return {}
    
    sample_name = os.path.basename(file_path).split('.')[0]
    print(f"Processing {sample_name}...")
    
    # Load the ECG data
    ecg_data = np.load(file_path)
    
    # Print diagnostic information about the data
    print(f"Data shape: {ecg_data.shape}")
    print(f"Data type: {ecg_data.dtype}")
    print(f"Data min/max: {ecg_data.min():.4f}/{ecg_data.max():.4f}")
    print(f"Data mean/std: {ecg_data.mean():.4f}/{ecg_data.std():.4f}")
    
    # Predict
    prediction = predict_sample(model, conditions, ecg_data, device)
    
    results = {}
    if prediction:
        results[sample_name] = prediction
        
        # Print results
        print(f"\nPrediction Results for {sample_name}:")
        print("-" * 50)
        print(f"{'Condition':<30} {'Probability':<15} {'Prediction'}")
        print("-" * 50)
        
        # Sort by probability
        sorted_predictions = sorted(prediction.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        for condition, result in sorted_predictions:
            prob = result['probability']
            pred = result['prediction']
            print(f"{condition:<30} {prob:.4f}         {'✓' if pred else '✗'}")
    
    return results

def generate_summary(all_results):
    """
    Generate a summary of the validation results
    """
    summary = {
        'total_samples': 0,
        'predictions_by_condition': {},
        'top_predictions': []
    }
    
    # Count total samples
    for folder, folder_results in all_results.items():
        summary['total_samples'] += len(folder_results)
        
        # Count predictions by condition
        for sample, sample_results in folder_results.items():
            # Find top prediction
            top_condition = None
            top_probability = 0.0
            
            for condition, result in sample_results.items():
                if result['prediction']:
                    if condition not in summary['predictions_by_condition']:
                        summary['predictions_by_condition'][condition] = 0
                    summary['predictions_by_condition'][condition] += 1
                
                if result['probability'] > top_probability:
                    top_probability = result['probability']
                    top_condition = condition
            
            if top_condition:
                summary['top_predictions'].append({
                    'folder': folder,
                    'sample': sample,
                    'condition': top_condition,
                    'probability': top_probability
                })
    
    return summary

def main():
    """
    Main function
    """
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Validate ECG samples using six conditions HuBERT-ECG model")
    parser.add_argument("--model_path", type=str, default="./outputs/six_conditions_model_anti_overfitting_best/best_model_accuracy.pt", 
                        help="Path to the trained model")
    
    # Add validation path options
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument("--validation_dir", type=str, 
                        help="Directory containing validation folders (default: ./validation)")
    validation_group.add_argument("--single_folder", type=str, 
                        help="Path to a single folder containing .npy files to validate")
    validation_group.add_argument("--single_file", type=str, 
                        help="Path to a single .npy file to validate")
    
    parser.add_argument("--output_file", type=str, default="./six_conditions_validation_results.json",
                        help="Output file for validation results")
    parser.add_argument("--summary_file", type=str, default="./six_conditions_validation_summary.txt",
                        help="Output file for validation summary")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    args = parser.parse_args()
    
    # Set default validation directory if no path options are provided
    if not args.validation_dir and not args.single_folder and not args.single_file:
        args.validation_dir = "./validation"
    
    # Load model
    model, conditions = load_six_conditions_model(args.model_path, args.device)
    
    all_results = {}
    
    # Process based on input type
    if args.single_file:
        # Process a single file
        file_path = os.path.abspath(args.single_file)
        print(f"Processing single file: {file_path}")
        
        results = validate_single_file(model, conditions, file_path, args.device)
        if results:
            all_results["single_file"] = results
    
    elif args.single_folder:
        # Process a single folder
        folder_path = os.path.abspath(args.single_folder)
        print(f"Processing single folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return
        
        folder_name = os.path.basename(folder_path)
        results = validate_folder(model, conditions, folder_path, args.device)
        if results:
            all_results[folder_name] = results
    
    else:
        # Process validation directory with multiple folders
        validation_dir = os.path.abspath(args.validation_dir)
        print(f"Processing validation directory: {validation_dir}")
        
        if not os.path.exists(validation_dir):
            print(f"Validation directory {validation_dir} does not exist")
            return
        
        # Get all validation folders
        validation_folders = [f for f in os.listdir(validation_dir) 
                             if os.path.isdir(os.path.join(validation_dir, f)) and f.startswith("validation")]
        
        if not validation_folders:
            # If no validation folders found, try to process the directory itself
            print(f"No validation folders found in {validation_dir}, trying to process directory directly")
            results = validate_folder(model, conditions, validation_dir, args.device)
            if results:
                all_results[os.path.basename(validation_dir)] = results
        else:
            # Sort folders
            validation_folders.sort()
            
            # Process each folder
            for folder in validation_folders:
                folder_path = os.path.join(validation_dir, folder)
                print(f"\nProcessing folder: {folder}")
                
                results = validate_folder(model, conditions, folder_path, args.device)
                if results:
                    all_results[folder] = results
    
    # Save results if we have any
    if all_results:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nResults saved to {args.output_file}")
        
        # Generate and save summary
        summary = generate_summary(all_results)
        
        with open(args.summary_file, 'w') as f:
            f.write("Validation Summary\n")
            f.write("=================\n\n")
            f.write(f"Total samples: {summary['total_samples']}\n\n")
            
            f.write("Predictions by condition:\n")
            for condition, count in sorted(summary['predictions_by_condition'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / summary['total_samples']) * 100
                f.write(f"  {condition}: {count} ({percentage:.1f}%)\n")
            
            f.write("\nTop predictions by sample:\n")
            for pred in summary['top_predictions']:
                f.write(f"  {pred['folder']}/{pred['sample']}: {pred['condition']} ({pred['probability']:.4f})\n")
        
        print(f"Summary saved to {args.summary_file}")
    else:
        print("No results were generated. Please check your input paths.")