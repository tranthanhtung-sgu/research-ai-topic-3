#!/usr/bin/env python3
"""
Script to load the saved HuBERT-ECG model and run inference on a sample
"""
import os
import torch
import numpy as np
from pathlib import Path
from custom_dataset import MIMICPhysionetDataset
from hubert_ecg_classification import HuBERTForECGClassification, ModelOutput
from hubert_ecg import HuBERTECG, HuBERTECGConfig

def load_model(model_path):
    """
    Load a saved HuBERT-ECG model
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get the conditions list
    conditions = checkpoint.get('conditions', ['NORMAL', 
                                               'Atrial Fibrillation', 
                                               'Atrial Flutter',
                                               '' 
                                               ])
    
    # Create a minimal config
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
    
    # Create the classification model
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
    
    return model, conditions

def load_sample(dataset_paths, conditions):
    """
    Load a single sample from the dataset
    """
    # Create a tiny dataset
    dataset = MIMICPhysionetDataset(
        dataset_paths=dataset_paths,
        conditions=conditions,
        split='test',  # Use test split to avoid any data leakage
        random_crop=False,
        augment=False,
        target_length=5000,  # Use same size as training
        seed=42
    )
    
    # Get a sample
    idx = 0
    signal, attention_mask, labels = dataset[idx]
    
    # Get condition name(s)
    true_conditions = []
    for i, val in enumerate(labels):
        if val > 0:
            true_conditions.append(conditions[i])
    
    print(f"Sample loaded - Labels: {true_conditions}")
    
    return signal, attention_mask, labels

def run_inference(model, signal, attention_mask, conditions):
    """
    Run inference on a single sample
    """
    # Prepare inputs (add batch dimension)
    signal = signal.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Move to same device as model
    device = next(model.parameters()).device
    signal = signal.to(device)
    attention_mask = attention_mask.to(device)
    
    print(f"Running inference on signal with shape: {signal.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            outputs = model(signal, attention_mask)
            
        # Get logits and apply sigmoid for probabilities
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.sigmoid(logits)
        
        # Get predictions
        predictions = (probs > 0.5).float()
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"{'Condition':<30} {'Probability':<15} {'Prediction'}")
        print("-" * 50)
        
        for i, condition in enumerate(conditions):
            prob = probs[0, i].item()
            pred = predictions[0, i].item()
            print(f"{condition:<30} {prob:.4f}         {'✓' if pred > 0.5 else '✗'}")
        
        return probs, predictions
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """
    Main function
    """
    print("\n=== HuBERT-ECG Model Inference ===")
    
    # Parameters
    model_path = './test_model_output/test_model.pt'
    dataset_paths = [
        '/home/tony/neurokit/MIMIC IV Selected',
        '/home/tony/neurokit/Physionet 2021 Selected'
    ]
    
    # 1. Load model
    model, conditions = load_model(model_path)
    
    # 2. Load a sample
    signal, attention_mask, labels = load_sample(dataset_paths, conditions)
    
    # 3. Run inference
    probs, preds = run_inference(model, signal, attention_mask, conditions)
    
    if probs is not None:
        print("\n✅ Inference completed successfully!")
    else:
        print("\n❌ Inference failed!")

if __name__ == "__main__":
    main() 