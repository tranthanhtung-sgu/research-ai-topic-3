#!/usr/bin/env python3
"""
Benchmark script for evaluating trained ECGFounder models on 6 ECG conditions
This script loads a trained model and provides comprehensive evaluation metrics
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
from ecg_data_loader import ECGDataLoader

class ECGBenchmark:
    """Comprehensive benchmarking class for ECG classification models"""
    
    def __init__(self, model, device, class_names, save_dir='./benchmark_results'):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        print("🔍 Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics"""
        print("📊 Calculating metrics...")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.n_classes)
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # AUC scores (one-vs-rest)
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        if self.n_classes == 2:
            auc_scores = [roc_auc_score(y_true_bin, y_prob[:, 1])]
        else:
            auc_scores = []
            for i in range(self.n_classes):
                try:
                    auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    auc_scores.append(auc)
                except ValueError:
                    auc_scores.append(0.0)  # Handle case where class is not present
        
        return {
            'overall_accuracy': overall_accuracy,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'auc': auc_scores
            },
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            }
        }
    
    def create_results_table(self, metrics):
        """Create a detailed results table"""
        print("📋 Creating results table...")
        
        # Create per-class results DataFrame
        results_data = []
        for i, class_name in enumerate(self.class_names):
            results_data.append({
                'Condition': class_name,
                'Precision': f"{metrics['per_class']['precision'][i]:.3f}",
                'Recall': f"{metrics['per_class']['recall'][i]:.3f}",
                'F1-Score': f"{metrics['per_class']['f1'][i]:.3f}",
                'AUC': f"{metrics['per_class']['auc'][i]:.3f}",
                'Support': int(metrics['per_class']['support'][i])
            })
        
        # Add summary rows
        results_data.append({
            'Condition': 'Macro Avg',
            'Precision': f"{metrics['macro_avg']['precision']:.3f}",
            'Recall': f"{metrics['macro_avg']['recall']:.3f}",
            'F1-Score': f"{metrics['macro_avg']['f1']:.3f}",
            'AUC': f"{np.mean(metrics['per_class']['auc']):.3f}",
            'Support': int(np.sum(metrics['per_class']['support']))
        })
        
        results_data.append({
            'Condition': 'Weighted Avg',
            'Precision': f"{metrics['weighted_avg']['precision']:.3f}",
            'Recall': f"{metrics['weighted_avg']['recall']:.3f}",
            'F1-Score': f"{metrics['weighted_avg']['f1']:.3f}",
            'AUC': '-',
            'Support': int(np.sum(metrics['per_class']['support']))
        })
        
        df = pd.DataFrame(results_data)
        
        # Save table
        table_path = os.path.join(self.save_dir, 'performance_table.csv')
        df.to_csv(table_path, index=False)
        
        # Print table
        print("\n" + "="*80)
        print("📊 PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
        print("\nPer-Condition Performance:")
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Create and save confusion matrix plot"""
        print("🎨 Creating confusion matrix plot...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, cm_percent
    
    def plot_per_class_metrics(self, metrics):
        """Create per-class metrics bar plot"""
        print("📊 Creating per-class metrics plot...")
        
        # Prepare data
        conditions = self.class_names
        precision = metrics['per_class']['precision']
        recall = metrics['per_class']['recall']
        f1 = metrics['per_class']['f1']
        auc = metrics['per_class']['auc']
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(conditions))
        width = 0.6
        
        # Precision
        bars1 = ax1.bar(x, precision, width, color='skyblue', alpha=0.8)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision by Condition', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=45)
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Recall
        bars2 = ax2.bar(x, recall, width, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Recall by Condition', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(conditions, rotation=45)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # F1-Score
        bars3 = ax3.bar(x, f1, width, color='lightgreen', alpha=0.8)
        ax3.set_ylabel('F1-Score', fontsize=12)
        ax3.set_title('F1-Score by Condition', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(conditions, rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # AUC
        bars4 = ax4.bar(x, auc, width, color='gold', alpha=0.8)
        ax4.set_ylabel('AUC', fontsize=12)
        ax4.set_title('AUC by Condition', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(conditions, rotation=45)
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        metrics_path = os.path.join(self.save_dir, 'per_class_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_prob):
        """Create ROC curves for each class"""
        print("📈 Creating ROC curves...")
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i in range(self.n_classes):
            if np.sum(y_true_bin[:, i]) > 0:  # Check if class is present
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                
                axes[i].plot(fpr, tpr, linewidth=2, 
                           label=f'ROC Curve (AUC = {auc:.3f})')
                axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'ROC Curve - {self.class_names[i]}')
                axes[i].legend(loc="lower right")
                axes[i].grid(alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, 'No samples\nfor this class', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'ROC Curve - {self.class_names[i]}')
        
        plt.tight_layout()
        
        # Save plot
        roc_path = os.path.join(self.save_dir, 'roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_results(self, metrics, y_true, y_pred):
        """Save detailed results to JSON"""
        print("💾 Saving detailed results...")
        
        # Create detailed results dictionary
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': float(metrics['overall_accuracy']),
            'class_names': self.class_names,
            'per_class_metrics': {},
            'macro_averages': {
                'precision': float(metrics['macro_avg']['precision']),
                'recall': float(metrics['macro_avg']['recall']),
                'f1_score': float(metrics['macro_avg']['f1'])
            },
            'weighted_averages': {
                'precision': float(metrics['weighted_avg']['precision']),
                'recall': float(metrics['weighted_avg']['recall']),
                'f1_score': float(metrics['weighted_avg']['f1'])
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            detailed_results['per_class_metrics'][class_name] = {
                'precision': float(metrics['per_class']['precision'][i]),
                'recall': float(metrics['per_class']['recall'][i]),
                'f1_score': float(metrics['per_class']['f1'][i]),
                'auc': float(metrics['per_class']['auc'][i]),
                'support': int(metrics['per_class']['support'][i])
            }
        
        # Save to JSON
        results_path = os.path.join(self.save_dir, 'detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return detailed_results
    
    def run_full_benchmark(self, test_loader):
        """Run complete benchmarking pipeline"""
        print("🚀 Starting comprehensive benchmark...")
        print("="*60)
        
        # Evaluate model
        y_true, y_pred, y_prob = self.evaluate_model(test_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Create results table
        results_df = self.create_results_table(metrics)
        
        # Create visualizations
        cm, cm_percent = self.plot_confusion_matrix(y_true, y_pred)
        self.plot_per_class_metrics(metrics)
        self.plot_roc_curves(y_true, y_prob)
        
        # Save detailed results
        detailed_results = self.save_detailed_results(metrics, y_true, y_pred)
        
        print(f"\n✅ Benchmark completed!")
        print(f"📁 Results saved to: {self.save_dir}")
        print(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.3f}")
        
        return detailed_results, results_df

def main():
    parser = argparse.ArgumentParser(description='Benchmark trained ECGFounder model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['12lead', '1lead'], default='12lead',
                      help='Model type')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint',
                      help='Path to original model checkpoints')
    parser.add_argument('--mimic_path', type=str, default='../MIMIC IV Selected',
                      help='Path to MIMIC IV Selected dataset')
    parser.add_argument('--physionet_path', type=str, default='../Physionet 2021 Selected',
                      help='Path to Physionet 2021 Selected dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=1000,
                      help='Maximum samples per condition for testing')
    parser.add_argument('--save_dir', type=str, default='./benchmark_results',
                      help='Directory to save benchmark results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"benchmark_{args.model_type}_{timestamp}")
    
    # Load trained model
    print(f"Loading trained model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model architecture
    if args.model_type == '12lead':
        original_model_path = os.path.join(args.checkpoint_path, '12_lead_ECGFounder.pth')
        model = ft_12lead_ECGFounder(device, original_model_path, n_classes=6, linear_prob=False)
    else:
        original_model_path = os.path.join(args.checkpoint_path, '1_lead_ECGFounder.pth')
        model = ft_1lead_ECGFounder(device, original_model_path, n_classes=6, linear_prob=False)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint.get('class_names', ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB'])
    
    print(f"Model loaded successfully!")
    print(f"Best validation accuracy during training: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # Initialize data loader
    print("Loading test data...")
    data_loader = ECGDataLoader(args.mimic_path, args.physionet_path)
    
    # Create test data loader
    _, test_loader = data_loader.create_dataloaders(
        batch_size=args.batch_size,
        max_samples_per_condition=args.max_samples
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize benchmark
    benchmark = ECGBenchmark(model, device, class_names, save_dir)
    
    # Run benchmark
    detailed_results, results_df = benchmark.run_full_benchmark(test_loader)
    
    print(f"\n🎯 Benchmark Summary:")
    print(f"   📁 Results directory: {save_dir}")
    print(f"   📊 Performance table: {save_dir}/performance_table.csv")
    print(f"   🎨 Confusion matrix: {save_dir}/confusion_matrix.png")
    print(f"   📈 Per-class metrics: {save_dir}/per_class_metrics.png")
    print(f"   📉 ROC curves: {save_dir}/roc_curves.png")
    print(f"   💾 Detailed results: {save_dir}/detailed_results.json")

if __name__ == "__main__":
    main() 