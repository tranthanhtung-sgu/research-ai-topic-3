#!/usr/bin/env python3
"""
Test script to verify ECGFounder fine-tuning setup
This script tests data loading, model loading, and basic functionality
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    try:
        from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
        from ecg_data_loader import ECGDataLoader
        from train_6_conditions import ECGTrainer
        import wfdb
        import scipy.io
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if pre-trained models can be loaded"""
    print("\n🧪 Testing model loading...")
    
    # Import the functions needed for this test
    try:
        from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
    except ImportError as e:
        print(f"❌ Cannot import model functions: {e}")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 12-lead model
    try:
        model_path = './checkpoint/12_lead_ECGFounder.pth'
        if not os.path.exists(model_path):
            print(f"❌ 12-lead model not found at {model_path}")
            return False
            
        model = ft_12lead_ECGFounder(device, model_path, n_classes=6, linear_prob=False)
        print(f"✅ 12-lead model loaded successfully")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 12, 5000).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"❌ 12-lead model loading failed: {e}")
        return False
    
    # Test 1-lead model  
    try:
        model_path = './checkpoint/1_lead_ECGFounder.pth'
        if not os.path.exists(model_path):
            print(f"❌ 1-lead model not found at {model_path}")
            return False
            
        model = ft_1lead_ECGFounder(device, model_path, n_classes=6, linear_prob=False)
        print(f"✅ 1-lead model loaded successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 1, 5000).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"❌ 1-lead model loading failed: {e}")
        return False
    
    return True

def test_data_paths():
    """Test if dataset paths exist and have expected structure"""
    print("\n🧪 Testing dataset paths...")
    
    mimic_path = '../MIMIC IV Selected'
    physionet_path = '../Physionet 2021 Selected'
    
    # Check MIMIC IV
    if not os.path.exists(mimic_path):
        print(f"❌ MIMIC IV dataset not found at {mimic_path}")
        return False
    
    # Check Physionet
    if not os.path.exists(physionet_path):
        print(f"❌ Physionet dataset not found at {physionet_path}")
        return False
    
    print(f"✅ Dataset paths exist")
    
    # Check condition folders
    expected_conditions = [
        'NORMAL', 'Atrial Fibrillation', 'Atrial Flutter', 
        '1st-degree AV block', 'Right bundle branch block', 'Left bundle branch block'
    ]
    
    mimic_folders = os.listdir(mimic_path)
    physionet_folders = os.listdir(physionet_path)
    
    print("   MIMIC IV folders:", [f for f in mimic_folders if os.path.isdir(os.path.join(mimic_path, f))])
    print("   Physionet folders:", [f for f in physionet_folders if os.path.isdir(os.path.join(physionet_path, f))])
    
    return True

def test_data_loading():
    """Test basic data loading functionality"""
    print("\n🧪 Testing data loading...")
    
    try:
        from ecg_data_loader import ECGDataLoader
        
        mimic_path = '../MIMIC IV Selected'
        physionet_path = '../Physionet 2021 Selected'
        
        if not (os.path.exists(mimic_path) and os.path.exists(physionet_path)):
            print("⚠️  Dataset paths not found, skipping data loading test")
            return True
        
        data_loader = ECGDataLoader(mimic_path, physionet_path)
        print("✅ Data loader initialized")
        
        # Test loading a small amount of data
        print("   Testing data loading with max 10 samples per condition...")
        train_loader, val_loader = data_loader.create_dataloaders(
            batch_size=4, max_samples_per_condition=10
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Test getting a batch
        for batch_data, batch_labels in train_loader:
            print(f"   Sample batch shape: {batch_data.shape}")
            print(f"   Sample labels: {batch_labels}")
            break
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    return True

def test_device():
    """Test device configuration"""
    print("\n🧪 Testing device configuration...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = torch.device('cuda')
    else:
        print("Using CPU (training will be slow)")
        device = torch.device('cpu')
    
    print(f"✅ Device configured: {device}")
    return True

def main():
    """Run all tests"""
    print("🚀 ECGFounder Fine-tuning Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Device", test_device),
        ("Model Loading", test_model_loading),
        ("Dataset Paths", test_data_paths),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Run: ./run_training.sh --help")
        print("2. Or: python train_6_conditions.py --help")
        print("3. Or: jupyter notebook fine_tune_ecg_6_conditions.ipynb")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Download model checkpoints (see README_FINETUNING.md)")
        print("- Check dataset paths")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 