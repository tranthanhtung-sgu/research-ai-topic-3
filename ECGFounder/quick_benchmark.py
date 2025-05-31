#!/usr/bin/env python3
"""
Quick benchmark script - automatically finds and evaluates the latest trained model
"""

import os
import glob
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
    """Run quick benchmark on latest model"""
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
    print("Loading test data...")
    data_loader = ECGDataLoader('../MIMIC IV Selected', '../Physionet 2021 Selected')
    
    # Create test data loader (smaller sample for quick benchmark)
    _, test_loader = data_loader.create_dataloaders(
        batch_size=32,
        max_samples_per_condition=200  # Smaller for quick evaluation
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Create benchmark directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./benchmark_results/quick_benchmark_{model_type}_{timestamp}"
    
    # Initialize benchmark
    benchmark = ECGBenchmark(model, device, class_names, save_dir)
    
    # Run benchmark
    print("\n" + "="*50)
    detailed_results, results_df = benchmark.run_full_benchmark(test_loader)
    
    # Print summary
    print("\n" + "🎯 QUICK BENCHMARK SUMMARY")
    print("="*50)
    print(f"📁 Results saved to: {save_dir}")
    print(f"📊 Overall Test Accuracy: {detailed_results['overall_accuracy']:.3f}")
    print(f"📈 Training Val Accuracy: {best_val_acc}")
    
    print("\n📋 Per-Condition Performance:")
    for condition, metrics in detailed_results['per_class_metrics'].items():
        print(f"  {condition:6s}: F1={metrics['f1_score']:.3f}, AUC={metrics['auc']:.3f}")
    
    print(f"\n📁 Detailed results: {save_dir}/")
    print("   - performance_table.csv")
    print("   - confusion_matrix.png") 
    print("   - per_class_metrics.png")
    print("   - roc_curves.png")
    print("   - detailed_results.json")

if __name__ == "__main__":
    quick_benchmark() 