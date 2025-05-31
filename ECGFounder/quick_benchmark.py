#!/usr/bin/env python3
"""
Quick benchmark script - automatically finds and evaluates the latest trained model
on both validation and test sets
"""

import os
import glob
# Set matplotlib backend to non-interactive to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from benchmark_model import ECGBenchmark
from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
from ecg_data_loader import ECGDataLoader
import torch
from datetime import datetime

def find_latest_model(results_dir='./results'):
    """Find the latest trained model"""
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find all model directories
    model_dirs = glob.glob(os.path.join(results_dir, "ecg_6conditions_*"))
    if not model_dirs:
        print(f"❌ No trained models found in {results_dir}")
        return None
    
    # Find the latest one
    latest_dir = max(model_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return None
    
    print(f"✅ Found latest model: {model_path}")
    return model_path, latest_dir

def quick_benchmark():
    """Run quick benchmark on latest model - evaluates on both validation and test sets"""
    print("🚀 Quick ECG Model Benchmark")
    print("="*50)
    
    # Find latest model
    result = find_latest_model()
    if result is None:
        print("Please train a model first using: ./run_training.sh")
        return
    
    model_path, model_dir = result
    
    # Determine model type from directory name
    model_type = '12lead' if '12lead' in os.path.basename(model_dir) else '1lead'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    
    # Load trained model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model architecture
    if model_type == '12lead':
        original_model_path = './checkpoint/12_lead_ECGFounder.pth'
        model = ft_12lead_ECGFounder(device, original_model_path, n_classes=6, linear_prob=False)
    else:
        original_model_path = './checkpoint/1_lead_ECGFounder.pth'
        model = ft_1lead_ECGFounder(device, original_model_path, n_classes=6, linear_prob=False)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint.get('class_names', ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB'])
    best_val_acc = checkpoint.get('best_val_acc', 'N/A')
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Best validation accuracy during training: {best_val_acc}")
    
    # Load test data
    print("Loading data...")
    data_loader = ECGDataLoader('../MIMIC IV Selected', '../Physionet 2021 Selected')
    
    # Create data loaders (smaller sample for quick benchmark)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        batch_size=32,
        max_samples_per_condition=200  # Smaller for quick evaluation
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create benchmark directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = f"./benchmark_results/quick_benchmark_{model_type}_{timestamp}"
    
    # Create separate directories for validation and test results
    val_dir = os.path.join(benchmark_dir, "validation")
    test_dir = os.path.join(benchmark_dir, "test")
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("Evaluating on validation set...")
    benchmark_val = ECGBenchmark(model, device, class_names, val_dir)
    val_results, val_df = benchmark_val.run_full_benchmark(val_loader)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    benchmark_test = ECGBenchmark(model, device, class_names, test_dir)
    test_results, test_df = benchmark_test.run_full_benchmark(test_loader)
    
    # Print summary comparison
    print("\n" + "🎯 BENCHMARK SUMMARY COMPARISON")
    print("="*70)
    print(f"📁 Results saved to: {benchmark_dir}")
    print(f"📊 Training Best Val Accuracy: {best_val_acc}")
    print("\n📋 Performance Comparison:")
    print(f"                     Validation          Test")
    print(f"                     ----------          ----")
    print(f"Overall Accuracy:    {val_results['overall_accuracy']:.3f}              {test_results['overall_accuracy']:.3f}")
    print(f"Macro Precision:     {val_results['macro_averages']['precision']:.3f}              {test_results['macro_averages']['precision']:.3f}")
    print(f"Macro Recall:        {val_results['macro_averages']['recall']:.3f}              {test_results['macro_averages']['recall']:.3f}")
    print(f"Macro F1-Score:      {val_results['macro_averages']['f1_score']:.3f}              {test_results['macro_averages']['f1_score']:.3f}")
    
    print("\n📋 Per-Condition F1-Scores:")
    for condition in class_names:
        val_f1 = val_results['per_class_metrics'][condition]['f1_score']
        test_f1 = test_results['per_class_metrics'][condition]['f1_score']
        print(f"  {condition:6s}:        {val_f1:.3f}              {test_f1:.3f}")
    
    print(f"\n📁 Detailed results: {benchmark_dir}/")
    print("   - validation/")
    print("     - performance_table.csv")
    print("     - confusion_matrix.png") 
    print("     - per_class_metrics.png")
    print("     - roc_curves.png")
    print("     - detailed_results.json")
    print("   - test/")
    print("     - performance_table.csv")
    print("     - confusion_matrix.png") 
    print("     - per_class_metrics.png")
    print("     - roc_curves.png")
    print("     - detailed_results.json")

if __name__ == "__main__":
    quick_benchmark() 