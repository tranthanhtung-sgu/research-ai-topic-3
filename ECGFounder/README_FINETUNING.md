# ECGFounder Fine-tuning on 6 ECG Conditions

This repository contains code to fine-tune the [ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder) model on 6 specific ECG conditions using MIMIC IV and Physionet 2021 datasets.

## 🎯 Target Conditions

The model is fine-tuned to classify the following 6 ECG conditions:

1. **NORM** - Normal Sinus Rhythm
2. **AFIB** - Atrial Fibrillation  
3. **AFLT** - Atrial Flutter
4. **1dAVb** - 1st Degree AV Block
5. **RBBB** - Right Bundle Branch Block
6. **LBBB** - Left Bundle Branch Block

## 📁 Project Structure

```
ECGFounder/
├── checkpoint/                     # Pre-trained model checkpoints
│   ├── 12_lead_ECGFounder.pth     # 12-lead ECGFounder model
│   └── 1_lead_ECGFounder.pth      # 1-lead ECGFounder model
├── results/                        # Training results and outputs
├── ecg_data_loader.py             # Data loading utilities
├── train_6_conditions.py          # Main training script
├── fine_tune_ecg_6_conditions.ipynb  # Jupyter notebook for interactive training
├── run_training.sh                # Shell script for easy training
├── finetune_model.py              # Model loading functions
├── net1d.py                       # Model architecture
└── requirements.txt               # Dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n ECGFounder python=3.10
conda activate ECGFounder

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

```bash
# Create checkpoint directory
mkdir -p checkpoint

# Download 12-lead model (370MB)
wget https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth -P checkpoint/

# Download 1-lead model (370MB) 
wget https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/1_lead_ECGFounder.pth -P checkpoint/
```

### 3. Dataset Preparation

Ensure your datasets are organized as follows:

```
../MIMIC IV Selected/
├── NORMAL/
├── Atrial Fibrillation/
├── Atrial Flutter/
├── 1st-degree AV block/
├── Right bundle branch block/
└── Left bundle branch block/

../Physionet 2021 Selected/
├── NORMAL/
├── Atrial Fibrillation/
├── Atrial Flutter/
├── 1st Degree AV Block/
├── Right bundle branch block/
└── Left bundle branch block/
```

### 4. Run Training

#### Option A: Using Shell Script (Recommended)

```bash
# Basic training with default parameters
./run_training.sh

# Custom configuration
./run_training.sh --model_type 12lead --epochs 30 --batch_size 16 --max_samples 500

# Linear probing (freeze all layers except classifier)
./run_training.sh --linear_prob --epochs 20
```

#### Option B: Direct Python Script

```bash
python train_6_conditions.py \
    --model_type 12lead \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --max_samples 1000
```

#### Option C: Jupyter Notebook

```bash
jupyter notebook fine_tune_ecg_6_conditions.ipynb
```

## ⚙️ Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model_type` | ECG model type | `12lead` | `12lead`, `1lead` |
| `--batch_size` | Training batch size | `32` | Any integer |
| `--epochs` | Number of training epochs | `50` | Any integer |
| `--lr` | Learning rate | `0.001` | Any float |
| `--weight_decay` | L2 regularization | `1e-4` | Any float |
| `--max_samples` | Max samples per condition | `1000` | Any integer |
| `--linear_prob` | Linear probing mode | `False` | Flag |
| `--mimic_path` | MIMIC IV dataset path | `../MIMIC IV Selected` | Path |
| `--physionet_path` | Physionet dataset path | `../Physionet 2021 Selected` | Path |

## 📊 Expected Results

### Performance Metrics

The fine-tuned model typically achieves:
- **Validation Accuracy**: 85-95% (depending on data quality and size)
- **Training Time**: 30-60 minutes (with GPU)
- **Model Size**: ~370MB (pre-trained) + classifier weights

### Output Files

After training, the following files are generated in the `results/` directory:

```
results/ecg_6conditions_12lead_YYYYMMDD_HHMMSS/
├── best_model.pth                    # Best model checkpoint
├── training_history.png              # Loss and accuracy plots
├── confusion_matrix_epoch_X.png      # Confusion matrix
├── classification_report_epoch_X.json # Detailed metrics
├── args.json                         # Training configuration
└── ecg_classifier_production.pth     # Production-ready model
```

## 🧠 Model Architecture

The ECGFounder uses a 1D ResNet-style architecture with:
- **Input**: 12-lead ECG signals (12 × 5000 samples at 500Hz)
- **Backbone**: Pre-trained feature extractor
- **Classifier**: Fully connected layer for 6-class classification
- **Parameters**: ~40M total, ~6K trainable (classifier only in linear probing)

## 💡 Training Strategies

### Full Fine-tuning (Default)
- Trains all model parameters
- Best for large datasets (>1000 samples per class)
- Longer training time but better performance

### Linear Probing (`--linear_prob`)
- Freezes backbone, only trains classifier
- Good for smaller datasets or quick experimentation
- Faster training, prevents overfitting

### Recommendations
- Start with linear probing for initial experiments
- Use full fine-tuning for production models
- Adjust learning rate based on validation performance
- Use early stopping to prevent overfitting

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   ./run_training.sh --batch_size 16
   ```

2. **Data Loading Errors**
   ```bash
   # Check dataset paths
   ls "../MIMIC IV Selected/"
   ls "../Physionet 2021 Selected/"
   ```

3. **Model Download Issues**
   ```bash
   # Manual download
   wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 \
   https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth \
   -P checkpoint/
   ```

### Performance Tips

- **GPU Usage**: Training is much faster with GPU (30 minutes vs 4+ hours)
- **Data Balance**: Ensure similar sample counts across classes
- **Memory**: 8GB+ RAM recommended for large datasets
- **Storage**: Ensure sufficient disk space for model checkpoints

## 📈 Advanced Usage

### Custom Data Augmentation

```python
# Add to ecg_data_loader.py
def augment_ecg(ecg_data):
    # Add noise
    noise = np.random.normal(0, 0.01, ecg_data.shape)
    return ecg_data + noise
```

### Cross-Validation

```python
# Modify train_6_conditions.py for k-fold CV
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
    # Train model on fold
    pass
```

### Hyperparameter Tuning

```bash
# Grid search example
for lr in 0.01 0.001 0.0001; do
    for bs in 16 32 64; do
        ./run_training.sh --lr $lr --batch_size $bs --epochs 20
    done
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{li2024electrocardiogram,
  title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains},
  author={Li, Jun and Aguirre, Aaron and Moura, Junior and Liu, Che and Zhong, Lanhai and Sun, Chenxi and Clifford, Gari and Westover, Brandon and Hong, Shenda},
  journal={arXiv preprint arXiv:2410.04133},
  year={2024}
}
```

## 📞 Support

- **Issues**: Create GitHub issues for bugs or feature requests
- **Questions**: Use GitHub Discussions for general questions
- **Documentation**: Check the original [ECGFounder repository](https://github.com/PKUDigitalHealth/ECGFounder)

## 📜 License

This project is licensed under the MIT License - see the original ECGFounder repository for details. 