# ECGFounder Model Benchmarking Guide

This guide demonstrates how to benchmark your trained ECGFounder models on 6 ECG conditions and reproduce comprehensive performance analysis.

## 🎯 Overview

The benchmarking system provides detailed performance evaluation for ECGFounder models trained on 6 ECG conditions:
- **NORM** - Normal Sinus Rhythm
- **AFIB** - Atrial Fibrillation  
- **AFLT** - Atrial Flutter
- **1dAVb** - 1st Degree AV Block
- **RBBB** - Right Bundle Branch Block
- **LBBB** - Left Bundle Branch Block

## 📊 What You Get

### Performance Metrics
- **Overall Accuracy** - Global model performance
- **Per-Condition Metrics** - Precision, Recall, F1-Score, AUC for each condition
- **Confusion Matrix** - Visual representation of classification performance
- **ROC Curves** - Receiver Operating Characteristic curves for each condition

### Output Files
```
benchmark_results/
├── performance_table.csv          # Detailed metrics table
├── confusion_matrix.png           # Visual confusion matrix (counts & percentages)
├── per_class_metrics.png          # Bar charts for all metrics
├── roc_curves.png                 # ROC curves for each condition
└── detailed_results.json          # Complete results in JSON format
```

## 🚀 Quick Start

### Prerequisites
1. **Trained Model**: You need a trained ECGFounder model (see main README for training)
2. **Test Data**: MIMIC IV and Physionet 2021 datasets organized by condition
3. **Environment**: ECGFounder conda environment activated

### Option 1: Quick Benchmark (Recommended)
```bash
# Automatically finds and benchmarks your latest trained model
python quick_benchmark.py
```

### Option 2: Manual Benchmark
```bash
# Benchmark a specific model
python benchmark_model.py --model_path ./results/ecg_6conditions_12lead_YYYYMMDD_HHMMSS/best_model.pth
```

## 📋 Step-by-Step Reproduction Guide

### Step 1: Environment Setup
```bash
# Activate the ECGFounder environment
conda activate ECGFounder

# Navigate to ECGFounder directory
cd ECGFounder/
```

### Step 2: Train a Model (if you haven't already)
```bash
# Quick training with small dataset for testing
./run_training.sh --lr 0.0001 --epochs 10 --max_samples 100

# Or full training
./run_training.sh --epochs 50 --max_samples 1000
```

### Step 3: Run Benchmark
```bash
# Quick benchmark (uses latest model automatically)
python quick_benchmark.py
```

### Step 4: View Results
The benchmark will generate timestamped results in `./benchmark_results/`. Check the console output for the exact path.

## 🔧 Advanced Usage

### Custom Benchmark Configuration
```bash
python benchmark_model.py \
    --model_path ./results/your_model/best_model.pth \
    --model_type 12lead \
    --max_samples 500 \
    --batch_size 16 \
    --save_dir ./custom_benchmark_results
```

### Benchmark Multiple Models
```bash
# Benchmark all trained models
for model_dir in ./results/ecg_6conditions_*/; do
    if [ -f "$model_dir/best_model.pth" ]; then
        echo "Benchmarking: $model_dir"
        python benchmark_model.py --model_path "$model_dir/best_model.pth"
    fi
done
```

### Compare Different Configurations
```bash
# Benchmark 12-lead vs 1-lead models
python benchmark_model.py --model_path ./results/model_12lead/best_model.pth --model_type 12lead
python benchmark_model.py --model_path ./results/model_1lead/best_model.pth --model_type 1lead
```

## 📈 Example Results

### Console Output
```
🚀 Starting comprehensive benchmark...
============================================================
🔍 Evaluating model...
Evaluating: 100%|██████████| 38/38 [00:02<00:00, 15.23it/s]
📊 Calculating metrics...
📋 Creating results table...

================================================================================
📊 PERFORMANCE SUMMARY
================================================================================
Overall Accuracy: 0.892

Per-Condition Performance:
  Condition  Precision  Recall  F1-Score    AUC  Support
       NORM      0.945   0.912     0.928  0.987      200
       AFIB      0.876   0.891     0.883  0.952      200
       AFLT      0.823   0.845     0.834  0.934      200
      1dAVb      0.901   0.887     0.894  0.965      200
       RBBB      0.889   0.923     0.906  0.971      200
       LBBB      0.912   0.898     0.905  0.978      200
  Macro Avg      0.891   0.893     0.892  0.964     1200
Weighted Avg      0.891   0.893     0.892      -     1200
================================================================================

✅ Benchmark completed!
📁 Results saved to: ./benchmark_results/benchmark_12lead_20241201_143022
📊 Overall Accuracy: 0.892
```

### Performance Table (CSV)
| Condition | Precision | Recall | F1-Score | AUC   | Support |
|-----------|-----------|--------|----------|-------|---------|
| NORM      | 0.945     | 0.912  | 0.928    | 0.987 | 200     |
| AFIB      | 0.876     | 0.891  | 0.883    | 0.952 | 200     |
| AFLT      | 0.823     | 0.845  | 0.834    | 0.934 | 200     |
| 1dAVb     | 0.901     | 0.887  | 0.894    | 0.965 | 200     |
| RBBB      | 0.889     | 0.923  | 0.906    | 0.971 | 200     |
| LBBB      | 0.912     | 0.898  | 0.905    | 0.978 | 200     |

### Visual Outputs
1. **Confusion Matrix**: Shows true vs predicted labels with counts and percentages
2. **Per-Class Metrics**: Bar charts for Precision, Recall, F1-Score, and AUC
3. **ROC Curves**: Individual ROC curves for each of the 6 conditions

## 🔍 Interpreting Results

### Key Metrics to Focus On

1. **Overall Accuracy**: Should be >85% for good performance
2. **Per-Class F1-Score**: Balanced measure of precision and recall
3. **AUC**: Area Under Curve - higher is better (>0.9 is excellent)
4. **Confusion Matrix**: Look for strong diagonal (correct predictions)

### Performance Expectations

| Metric | Excellent | Good | Needs Improvement |
|--------|-----------|------|-------------------|
| Overall Accuracy | >90% | 80-90% | <80% |
| Per-Class F1 | >0.9 | 0.8-0.9 | <0.8 |
| AUC | >0.95 | 0.85-0.95 | <0.85 |

### Common Issues and Solutions

#### Low Performance on Specific Conditions
```bash
# Check class distribution in your data
python -c "
from ecg_data_loader import ECGDataLoader
loader = ECGDataLoader('../MIMIC IV Selected', '../Physionet 2021 Selected')
loader.load_all_data(max_samples_per_condition=1000)
"
```

#### Imbalanced Performance
- **Solution**: Use class weights or balanced sampling
- **Check**: Confusion matrix for systematic misclassifications

#### Low Overall Accuracy
- **Solutions**: 
  - Increase training epochs
  - Reduce learning rate
  - Use data augmentation
  - Check for data quality issues

## 🔄 Reproducibility Checklist

### Before Benchmarking
- [ ] ECGFounder environment activated
- [ ] Model successfully trained (no NaN losses)
- [ ] Test datasets available and organized
- [ ] Sufficient disk space for results (~100MB per benchmark)

### During Benchmarking
- [ ] Monitor GPU memory usage
- [ ] Check for any error messages
- [ ] Verify test data loading correctly

### After Benchmarking
- [ ] Review performance table
- [ ] Examine confusion matrix for patterns
- [ ] Compare with expected performance ranges
- [ ] Save results for future comparison

## 📊 Benchmark Configuration Options

### Command Line Arguments
```bash
python benchmark_model.py --help
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to trained model | Required |
| `--model_type` | 12lead or 1lead | 12lead |
| `--batch_size` | Evaluation batch size | 32 |
| `--max_samples` | Max samples per condition | 1000 |
| `--save_dir` | Results directory | ./benchmark_results |

### Quick Benchmark vs Full Benchmark

| Feature | Quick Benchmark | Full Benchmark |
|---------|----------------|----------------|
| Model Detection | Automatic | Manual path |
| Sample Size | 200 per condition | Configurable |
| Speed | ~2-3 minutes | ~5-10 minutes |
| Use Case | Quick validation | Comprehensive analysis |

## 🚨 Troubleshooting

### Common Errors

#### "No trained models found"
```bash
# Check if you have trained models
ls -la ./results/
# If empty, train a model first
./run_training.sh --epochs 5 --max_samples 50
```

#### "CUDA out of memory"
```bash
# Reduce batch size
python benchmark_model.py --model_path your_model.pth --batch_size 16
```

#### "Dataset not found"
```bash
# Check dataset paths
ls -la "../MIMIC IV Selected/"
ls -la "../Physionet 2021 Selected/"
```

#### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Performance Issues

#### Slow benchmarking
- Reduce `--max_samples` for faster evaluation
- Use smaller `--batch_size` if memory constrained
- Ensure GPU is being used (check console output)

#### Inconsistent results
- Set random seeds for reproducibility
- Use same test data split
- Ensure model is in eval mode

## 📚 Additional Resources

### Files in This Repository
- `benchmark_model.py` - Full benchmarking script
- `quick_benchmark.py` - Automated quick benchmark
- `ecg_data_loader.py` - Data loading utilities
- `train_6_conditions.py` - Training script
- `README_FINETUNING.md` - Training guide

### External Resources
- [ECGFounder Paper](https://arxiv.org/abs/2410.04133)
- [Original Repository](https://github.com/PKUDigitalHealth/ECGFounder)
- [Hugging Face Model](https://huggingface.co/PKUDigitalHealth/ECGFounder)

## 🤝 Contributing

To improve the benchmarking system:

1. **Add new metrics**: Modify `ECGBenchmark.calculate_metrics()`
2. **New visualizations**: Add methods to `ECGBenchmark` class
3. **Export formats**: Extend `save_detailed_results()` method
4. **Comparison tools**: Create scripts to compare multiple models

### Example: Adding Sensitivity/Specificity
```python
# In benchmark_model.py, add to calculate_metrics():
from sklearn.metrics import confusion_matrix

def calculate_sensitivity_specificity(y_true, y_pred, class_idx):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm) == 2 else cm[class_idx, class_idx], ...
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity
```

## 📄 Citation

If you use this benchmarking system in your research:

```bibtex
@article{li2024electrocardiogram,
  title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains},
  author={Li, Jun and Aguirre, Aaron and Moura, Junior and Liu, Che and Zhong, Lanhai and Sun, Chenxi and Clifford, Gari and Westover, Brandon and Hong, Shenda},
  journal={arXiv preprint arXiv:2410.04133},
  year={2024}
}
```

---

**Happy Benchmarking! 🚀**

For questions or issues, please check the troubleshooting section or create an issue in the repository. 