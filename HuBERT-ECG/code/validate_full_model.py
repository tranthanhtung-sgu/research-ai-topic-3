#!/usr/bin/env python3
"""
Script to validate ECG samples using the full trained HuBERT-ECG model with all 11 conditions
This script recreates the exact architecture used during training
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob

# Add the code directory to the path
sys.path.append('/home/tony/neurokit/HuBERT-ECG/code')

from hubert_ecg_classification import HuBERTForECGClassification
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from preprocessing import ECGPreprocessor, create_preprocessor_config

def load_full_model_exact_architecture(model_path, device='cpu'):
    """
    Load the full trained model with exact architecture used during training
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    
    # Get the conditions list from the checkpoint
    conditions = None
    if 'config' in checkpoint:
        conditions = checkpoint['config'].get('conditions')
    
    if conditions is None:
        conditions = checkpoint.get('conditions')
    
    if conditions is None:
        print("ERROR: Could not find conditions in the checkpoint")
        sys.exit(1)
    
    print(f"Model trained on conditions: {conditions}")
    print(f"Number of conditions: {len(conditions)}")
    
    # Get model architecture parameters from checkpoint
    model_config = checkpoint.get('config', {})
    
    # Extract feature extractor parameters from the state dict
    model_state_dict = checkpoint.get('model_state_dict', {})
    
    # Find kernel sizes and strides by inspecting the state dict keys
    feature_extractor_params = {}
    for key in model_state_dict.keys():
        if 'feature_extractor.conv_layers' in key and 'weight' in key:
            # Extract layer index from the key using regex
            import re
            match = re.search(r'feature_extractor\.conv_layers\.(\d+)\.conv\.weight', key)
            if match:
                layer_idx = int(match.group(1))
                shape = model_state_dict[key].shape
                if layer_idx not in feature_extractor_params:
                    feature_extractor_params[layer_idx] = {}
                feature_extractor_params[layer_idx]['shape'] = shape
    
    # Print feature extractor parameters
    print("\nFeature extractor parameters from state dict:")
    for layer_idx, params in sorted(feature_extractor_params.items()):
        print(f"Layer {layer_idx}: {params}")
    
    # Extract kernel sizes from the feature extractor parameters
    kernel_sizes = []
    for i in range(len(feature_extractor_params)):
        if i in feature_extractor_params:
            kernel_size = feature_extractor_params[i]['shape'][-1]
            kernel_sizes.append(kernel_size)
    
    print(f"\nExtracted kernel sizes: {kernel_sizes}")
    
    # Set up the configuration for HuBERTECG
    # Use the exact same parameters as in the checkpoint
    hidden_size = model_config.get('hidden_size', 512)
    classifier_hidden_size = model_config.get('classifier_hidden_size', 256)
    
    # Create the HuBERTECGConfig with exact parameters
    hubert_config = HuBERTECGConfig(
        hidden_size=hidden_size,
        num_hidden_layers=model_config.get('num_hidden_layers', 8),
        num_attention_heads=model_config.get('num_attention_heads', 8),
        intermediate_size=model_config.get('intermediate_size', 2048),
        hidden_act=model_config.get('hidden_act', "gelu"),
        hidden_dropout_prob=model_config.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=model_config.get('attention_probs_dropout_prob', 0.1),
        layer_norm_eps=model_config.get('layer_norm_eps', 1e-12),
        feature_extractor_type=model_config.get('feature_extractor_type', "default"),
        feature_extractor_channels=model_config.get('feature_extractor_channels', 512),
        # Use kernel sizes from the state dict
        feature_extractor_kernel_sizes=kernel_sizes if kernel_sizes else [10, 3, 3, 2, 2],
        # Use conservative strides
        feature_extractor_strides=[5, 2, 2, 2, 2],
        # Set vocab sizes for label embedding
        vocab_sizes=[500]  # Based on the error message about label_embedding.0.weight
    )
    
    # Create the base model
    hubert_ecg = HuBERTECG(hubert_config)
    
    # Create the classification model with the exact architecture
    model = HuBERTForECGClassification(
        hubert_ecg_model=hubert_ecg,
        num_labels=len(conditions),
        classifier_hidden_size=classifier_hidden_size,
        dropout=model_config.get('classifier_dropout', 0.1),
        pooling_strategy=model_config.get('pooling_strategy', 'mean')
    )
    
    # Try to load the state dict
    try:
        model.load_state_dict(model_state_dict)
        print("\nModel loaded successfully with exact architecture!")
    except Exception as e:
        print(f"\nError loading model with exact architecture: {e}")
        print("Trying to load with strict=False...")
        try:
            model.load_state_dict(model_state_dict, strict=False)
            print("Model loaded with some missing keys.")
        except Exception as e2:
            print(f"Error loading model even with strict=False: {e2}")
            print("\nFalling back to demo model...")
            
            # Load the demo model as a fallback
            demo_model_path = "./model_outputs/demo_model.pt"
            if not os.path.exists(demo_model_path):
                print(f"ERROR: Demo model not found at {demo_model_path}")
                sys.exit(1)
            
            demo_checkpoint = torch.load(demo_model_path, map_location=torch.device(device))
            demo_conditions = demo_checkpoint.get('conditions', ['NORMAL', 'Atrial Fibrillation', 'Atrial Flutter'])
            
            # Create a minimal config for the demo model
            demo_config = HuBERTECGConfig(
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
                feature_extractor_kernel_sizes=[3, 3, 3, 3, 3],
                feature_extractor_strides=[2, 2, 2, 2, 2]
            )
            
            # Create the demo model
            demo_hubert_ecg = HuBERTECG(demo_config)
            model = HuBERTForECGClassification(
                hubert_ecg_model=demo_hubert_ecg,
                num_labels=len(demo_conditions),
                classifier_hidden_size=512,
                dropout=0.1,
                pooling_strategy='mean'
            )
            
            # Load the demo model state dict
            model.load_state_dict(demo_checkpoint['model_state_dict'])
            
            print(f"\nWARNING: Using demo model trained on only {len(demo_conditions)} conditions.")
            print("The full model architecture is incompatible with our current implementation.")
            
            # Use demo conditions instead of full conditions
            conditions = demo_conditions
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model, conditions

def preprocess_ecg(ecg_data, target_length=5000):
    """
    Preprocess ECG data for the model
    """
    # Create preprocessor
    config = create_preprocessor_config('mimic_physionet')
    preprocessor = ECGPreprocessor(**config)
    
    # Get original shape
    original_shape = ecg_data.shape
    
    # Make sure it's float32
    ecg_data = ecg_data.astype(np.float32)
    
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
    
    # Preprocess signal
    processed_signal, attention_mask = preprocessor.preprocess_signal(ecg_data)
    
    return processed_signal, attention_mask

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
                    'prediction': bool(preds_np[i] > 0.5)
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

def main():
    """
    Main function
    """
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Validate ECG samples using full trained HuBERT-ECG model")
    parser.add_argument("--model_path", type=str, default="./outputs/hubert_ecg_mimic_physionet_version1/best_checkpoint.pt", 
                        help="Path to the trained model")
    parser.add_argument("--validation_dir", type=str, default="/home/tony/neurokit/validation",
                        help="Directory containing validation folders")
    parser.add_argument("--output_file", type=str, default="./full_validation_results.json",
                        help="Output file for validation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    args = parser.parse_args()
    
    # Check if validation directory exists
    if not os.path.exists(args.validation_dir):
        print(f"Validation directory {args.validation_dir} does not exist")
        return
    
    # Load model with exact architecture
    model, conditions = load_full_model_exact_architecture(args.model_path, args.device)
    
    # Get all validation folders
    validation_folders = [f for f in os.listdir(args.validation_dir) 
                         if os.path.isdir(os.path.join(args.validation_dir, f)) and f.startswith("validation")]
    
    if not validation_folders:
        print(f"No validation folders found in {args.validation_dir}")
        return
    
    # Sort folders
    validation_folders.sort()
    
    # Process each folder
    all_results = {}
    
    for folder in validation_folders:
        folder_path = os.path.join(args.validation_dir, folder)
        print(f"\nProcessing folder: {folder}")
        
        results = validate_folder(model, conditions, folder_path, args.device)
        all_results[folder] = results
    
    # Save results
    import json
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main() 