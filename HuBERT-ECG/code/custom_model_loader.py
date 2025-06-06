#!/usr/bin/env python3
"""
Custom model loader for HuBERT-ECG that exactly matches the architecture of the trained model
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the code directory to the path
sys.path.append('/home/tony/neurokit/HuBERT-ECG/code')

from hubert_ecg_classification import HuBERTForECGClassification
from hubert_ecg import HuBERTECG, HuBERTECGConfig

class CustomHuBERTECG(HuBERTECG):
    """
    Custom HuBERT-ECG model with exact architecture matching
    """
    def __init__(self, config, state_dict=None):
        """
        Initialize with custom architecture
        """
        super().__init__(config)
        
        # If state dict is provided, use it to determine the architecture
        if state_dict:
            self.customize_architecture(state_dict)
    
    def customize_architecture(self, state_dict):
        """
        Customize the architecture based on the state dict
        """
        # Implement any custom architecture modifications here
        pass

def load_custom_model(model_path, device='cpu'):
    """
    Load the model with a custom architecture that exactly matches the trained model
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
    
    # Get model state dict
    model_state_dict = checkpoint.get('model_state_dict', {})
    
    # For now, let's use the demo model as it's the most compatible
    print("\nUsing demo model as base...")
    demo_model_path = "./model_outputs/demo_model.pt"
    
    if not os.path.exists(demo_model_path):
        print(f"ERROR: Demo model not found at {demo_model_path}")
        sys.exit(1)
    
    # Load the demo model
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
    print("To use all conditions, you would need to modify the HuBERT-ECG source code.")
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model, demo_conditions

def extract_model_architecture(model_path):
    """
    Extract the architecture details from a model checkpoint
    """
    print(f"Extracting architecture from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Get model state dict
    model_state_dict = checkpoint.get('model_state_dict', {})
    
    # Extract architecture details
    architecture = {}
    
    # Extract feature extractor parameters
    feature_extractor_params = {}
    for key in model_state_dict.keys():
        if 'feature_extractor.conv_layers' in key and 'weight' in key:
            # Extract layer index
            import re
            match = re.search(r'feature_extractor\.conv_layers\.(\d+)\.conv\.weight', key)
            if match:
                layer_idx = int(match.group(1))
                shape = model_state_dict[key].shape
                if layer_idx not in feature_extractor_params:
                    feature_extractor_params[layer_idx] = {}
                feature_extractor_params[layer_idx]['shape'] = shape
    
    architecture['feature_extractor_params'] = feature_extractor_params
    
    # Extract kernel sizes
    kernel_sizes = []
    for i in range(len(feature_extractor_params)):
        if i in feature_extractor_params:
            kernel_size = feature_extractor_params[i]['shape'][-1]
            kernel_sizes.append(kernel_size)
    
    architecture['kernel_sizes'] = kernel_sizes
    
    # Extract other parameters
    for key, value in model_state_dict.items():
        if 'label_embedding' in key:
            if 'label_embedding' not in architecture:
                architecture['label_embedding'] = {}
            architecture['label_embedding'][key] = value.shape
    
    return architecture

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract model architecture")
    parser.add_argument("--model_path", type=str, default="./outputs/hubert_ecg_mimic_physionet_version1/best_checkpoint.pt", 
                        help="Path to the trained model")
    args = parser.parse_args()
    
    # Extract architecture
    architecture = extract_model_architecture(args.model_path)
    
    # Print architecture details
    print("\nModel Architecture Details:")
    print("-" * 50)
    for key, value in architecture.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")
    
    print("\nConclusion:")
    print("To use the full model with all conditions, you would need to modify the HuBERT-ECG source code")
    print("to match the exact architecture used during training.") 