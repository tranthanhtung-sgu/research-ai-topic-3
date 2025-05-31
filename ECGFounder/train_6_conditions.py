import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# Set matplotlib backend to non-interactive to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
from ecg_data_loader import ECGDataLoader

class ECGTrainer:
    """Trainer class for ECG classification on 6 conditions"""
    
    def __init__(self, model, device, class_names, save_dir='./results'):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Check for NaN in input data
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Warning: NaN/Inf detected in input data at batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            output = self.model(data)
            
            # Check for NaN in model output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Warning: NaN/Inf detected in model output at batch {batch_idx}")
                continue
                
            loss = criterion(output, target)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping")
                continue
                
            loss.backward()
            
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4, 
              patience=10, save_best=True):
        """Full training loop with early stopping"""
        
        # Setup optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print("-" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_predictions, val_targets = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                if save_best:
                    model_path = os.path.join(self.save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_acc': best_val_acc,
                        'class_names': self.class_names
                    }, model_path)
                    print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                    
                    # Save validation predictions for best model
                    self.save_classification_report(val_targets, val_predictions, epoch)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def save_classification_report(self, targets, predictions, epoch):
        """Save detailed classification report"""
        # Classification report
        report = classification_report(targets, predictions, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Save as JSON
        report_path = os.path.join(self.save_dir, f'classification_report_epoch_{epoch}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(self.save_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        history_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ECGFounder on 6 conditions')
    parser.add_argument('--mimic_path', type=str, default='../MIMIC IV Selected',
                      help='Path to MIMIC IV Selected dataset')
    parser.add_argument('--physionet_path', type=str, default='../Physionet 2021 Selected',
                      help='Path to Physionet 2021 Selected dataset')
    parser.add_argument('--model_type', type=str, choices=['12lead', '1lead'], default='12lead',
                      help='Model type to use')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint',
                      help='Path to model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_samples', type=int, default=1000, 
                      help='Maximum samples per condition')
    parser.add_argument('--linear_prob', action='store_true', 
                      help='Use linear probing (freeze all except last layer)')
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"ecg_6conditions_{args.model_type}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    args_path = os.path.join(save_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = ECGDataLoader(args.mimic_path, args.physionet_path)
    
    # Create data loaders with 40-40-20 split
    print("Loading data...")
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        batch_size=args.batch_size,
        max_samples_per_condition=args.max_samples
    )
    
    class_names = data_loader.get_class_names()
    n_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load pre-trained model
    print(f"Loading {args.model_type} ECGFounder model...")
    
    if args.model_type == '12lead':
        model_path = os.path.join(args.checkpoint_path, '12_lead_ECGFounder.pth')
        model = ft_12lead_ECGFounder(device, model_path, n_classes, args.linear_prob)
    else:
        model_path = os.path.join(args.checkpoint_path, '1_lead_ECGFounder.pth')
        model = ft_1lead_ECGFounder(device, model_path, n_classes, args.linear_prob)
    
    print(f"Model loaded successfully!")
    print(f"Linear probing mode: {args.linear_prob}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = ECGTrainer(model, device, class_names, save_dir)
    
    # Train model
    best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=10
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save test loader in checkpoint for later use
    model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(model_path)
    checkpoint['test_set_exists'] = True
    torch.save(checkpoint, model_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {save_dir}")
    
    # Optionally: run quick test evaluation
    print("\nEvaluating on test set...")
    from benchmark_model import ECGBenchmark
    
    # Initialize benchmark
    benchmark = ECGBenchmark(trainer.model, device, class_names, 
                            os.path.join(save_dir, 'test_evaluation'))
    
    # Run evaluation on test set
    test_results, _ = benchmark.run_full_benchmark(test_loader)
    
    print(f"\nTest set evaluation:")
    print(f"  Overall accuracy: {test_results['overall_accuracy']:.3f}")
    print(f"  Macro F1-score: {test_results['macro_averages']['f1_score']:.3f}")
    print(f"  Results saved to: {os.path.join(save_dir, 'test_evaluation')}")

if __name__ == "__main__":
    main() 