# HuBERT-ECG Custom Training Pipeline

This repository contains a custom training pipeline for fine-tuning the HuBERT-ECG model on MIMIC IV and PhysioNet 2021 ECG datasets.

## Overview

The HuBERT-ECG model is a self-supervised learning model for ECG signals. This pipeline allows you to fine-tune it for specific cardiac conditions classification tasks.

## Files

- `custom_dataset.py`: Dataset loader for WFDB format ECG files
- `preprocessing.py`: Preprocessing pipeline for ECG signals
- `hubert_ecg_classification.py`: Classification wrapper for HuBERT-ECG
- `custom_finetune.py`: Fine-tuning script with various training options
- `example_usage.py`: Demo script showing how to use the pipeline
- `test_training.py`: A minimal training test to verify functionality
- `model_inference.py`: Script to test saved models on new data

## Quick Start

To test the pipeline:

```bash
python example_usage.py --test-only
```

To train a model:

```bash
python example_usage.py --train-only
```

To run both tests and training:

```bash
python example_usage.py
```

## Features

- **Multi-dataset Support**: Handles both MIMIC IV and PhysioNet 2021 datasets
- **Data Augmentation**: Includes time shifting, amplitude scaling, and noise addition
- **Imbalanced Data Handling**: Class weights and specialized loss functions (e.g., Focal Loss)
- **Flexible Training**: Options for freezing layers, learning rate schedules, and more
- **Automatic Handling**: Handles different sequence lengths and tensor dimensions

## Model Configuration

The model configuration in this repository uses a smaller, faster version of HuBERT-ECG with:

- 8 transformer layers
- 512 hidden dimension
- 8 attention heads
- Smaller convolutional kernel sizes for faster processing

## Sequence Length Handling

This implementation can handle different ECG sequence lengths, automatically adapting to the model's requirements by:

1. Using larger initial sequences (5000 samples)
2. Applying appropriate striding in the feature extractor
3. Implementing robust error handling for different input dimensions

## Testing and Inference

The repository includes scripts for both testing during development and inference on new data. The model saves checkpoints that can be loaded for inference.

## Performance

The model is designed to work with imbalanced datasets, which is common in ECG classification tasks. It uses:

- Binary cross-entropy loss
- Optional focal loss for highly imbalanced classes
- Class weights based on sample frequency

## Customization

You can customize the model by modifying:

- Model size and architecture in `HuBERTECGConfig`
- Preprocessing parameters in `MIMICPhysionetDataset`
- Training parameters in the training scripts

## References

- Original HuBERT-ECG model: [Edoardo-BS/HuBERT-ECG-SSL-Pretrained](https://huggingface.co/Edoardo-BS/HuBERT-ECG-SSL-Pretrained-small)
- MIMIC IV dataset: [PhysioNet MIMIC IV](https://physionet.org/content/mimic-iv/2.2/)
- PhysioNet 2021 dataset: [PhysioNet Challenge 2021](https://physionetchallenges.org/2021/)

## 📁 Files Overview

### Core Components

1. **`custom_dataset.py`**: Custom dataset loader for WFDB format ECG data
   - Handles MIMIC IV and Physionet datasets
   - Multi-label classification support
   - Automatic train/val/test splits
   - Class balancing

2. **`preprocessing.py`**: Comprehensive ECG preprocessing pipeline
   - Signal filtering and normalization
   - Resampling and temporal alignment
   - Artifact removal
   - Data augmentation

3. **`hubert_ecg_classification.py`**: Classification wrapper for HuBERT-ECG
   - Multiple pooling strategies
   - Flexible classification head
   - Advanced loss functions (Focal Loss, Label Smoothing)

4. **`custom_finetune.py`**: Complete fine-tuning script
   - Layer-wise learning rates
   - Gradient accumulation
   - Early stopping
   - Comprehensive logging

5. **`example_usage.py`**: Example script showing how to use everything

## 🔧 Installation

### Dependencies
```bash
pip install torch transformers
pip install numpy pandas scipy scikit-learn
pip install wfdb loguru tqdm
pip install wandb  # Optional, for experiment tracking
```

### Dataset Structure
Your datasets should be organized like this:
```
MIMIC IV Selected/
├── NORMAL/
│   ├── 12345.dat
│   ├── 12345.hea
│   └── ...
├── AF/
│   ├── 67890.dat
│   ├── 67890.hea
│   └── ...
└── ...

Physionet 2021 Selected/
├── NORMAL/
│   ├── JS00001.dat
│   ├── JS00001.hea
│   └── ...
└── ...
```

## 🎛️ Configuration Options

### Dataset Configuration
- **`dataset_paths`**: List of paths to your dataset folders
- **`conditions`**: List of condition names to include
- **`target_length`**: Target signal length in samples (2500 = 5 seconds at 500Hz)
- **`random_crop`**: Use random cropping during training
- **`augment`**: Apply data augmentation

### Model Configuration
- **`model_size`**: HuBERT-ECG model size ('small', 'base', 'large')
- **`classifier_hidden_size`**: Hidden size for classification head
- **`pooling_strategy`**: How to pool sequence representations ('mean', 'max', 'attention')

### Training Configuration
- **`epochs`**: Number of training epochs
- **`batch_size`**: Batch size for training
- **`lr`**: Learning rate
- **`layer_wise_lr`**: Use different learning rates for different layers
- **`freeze_feature_extractor`**: Freeze the CNN feature extractor
- **`freeze_transformer_layers`**: Number of transformer layers to freeze

### Advanced Options
- **`use_class_weights`**: Use class weights for imbalanced datasets
- **`accumulation_steps`**: Gradient accumulation steps
- **`warmup_ratio`**: Learning rate warmup ratio
- **`patience`**: Early stopping patience

## 📊 Expected Results

With the provided configuration, you can expect:
- **Training time**: 2-4 hours (depending on dataset size and GPU)
- **Memory usage**: 4-8GB GPU memory (with batch_size=16)
- **Performance**: AUROC 85-95% depending on conditions and dataset quality

## 🔍 Monitoring Training

### Logs
Training logs are saved to:
- Console output with progress bars
- `training.log` file
- Wandb dashboard (if enabled)

### Checkpoints
Model checkpoints are saved to:
- `outputs/latest_checkpoint.pt`: Latest model
- `outputs/best_checkpoint.pt`: Best validation model
- `outputs/test_results.json`: Final test results

### Metrics
The pipeline tracks:
- **AUROC**: Area under ROC curve (macro average)
- **F1-score**: F1 score (macro average)
- **Precision**: Precision (macro average)
- **Recall**: Recall (macro average)

## 🚨 Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `batch_size` or use `accumulation_steps`
2. **Dataset loading errors**: Check file paths and WFDB format
3. **Slow training**: Enable `freeze_feature_extractor` or `freeze_transformer_layers`
4. **Poor performance**: Try different learning rates or enable `use_class_weights`

### Performance Tips

1. **For small datasets**: Use `freeze_feature_extractor=True`
2. **For imbalanced datasets**: Use `use_class_weights=True`
3. **For faster training**: Use `freeze_transformer_layers=4`
4. **For better performance**: Use `layer_wise_lr=True`

## 📈 Advanced Usage

### Custom Loss Functions
```python
from hubert_ecg_classification import FocalLoss, LabelSmoothingBCELoss

# Use Focal Loss for imbalanced datasets
criterion = FocalLoss(alpha=1.0, gamma=2.0)

# Use Label Smoothing for regularization
criterion = LabelSmoothingBCELoss(smoothing=0.1)
```

### Custom Preprocessing
```python
from preprocessing import ECGPreprocessor

# Create custom preprocessor
preprocessor = ECGPreprocessor(
    target_fs=250,  # Different sampling rate
    target_length=1250,  # 5 seconds at 250Hz
    normalize_method='robust',  # Robust normalization
    filter_config={
        'lowcut': 0.5,
        'highcut': 30.0,  # Lower high-freq cutoff
        'order': 6
    }
)
```

### Multi-GPU Training
```python
# Wrap model for data parallel training
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## 🤝 Contributing

To add support for new datasets:
1. Extend `MIMICPhysionetDataset` class
2. Add dataset-specific preprocessing in `preprocessing.py`
3. Update condition mappings as needed

## 📚 References

- [HuBERT-ECG Paper](https://arxiv.org/abs/2409.06974)
- [Original HuBERT-ECG Code](https://github.com/Edoar-do/HuBERT-ECG)
- [HuggingFace Models](https://huggingface.co/Edoardo-BS/HuBERT-ECG-SSL-Pretrained)

## 📄 License

This code is provided under the same license as the original HuBERT-ECG project (CC BY-NC 4.0).

### Handling Dataset Variations

The dataset loader automatically handles slight variations in folder names between MIMIC IV and Physionet datasets:

```python
# In custom_dataset.py
condition_variants = {
    '1st-degree AV block': ['1st-degree AV block', '1st Degree AV Block'],
    '2nd-degree AV block': ['2nd-degree AV block', '2nd Degree AV Block'],
    # Add other variants if needed
}
```

If you encounter other folder name variations, simply add them to the `condition_variants` dictionary. 