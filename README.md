# 12-Lead ECG Classification Benchmark Results

## Overview
This directory contains the benchmark results for a 12-lead ECG classification model evaluated on May 31, 2025. The model was tested on 6 cardiac conditions with 200 samples per class.

## Performance Metrics

| Condition | Precision | Recall | F1-Score | AUC | Support |
|-----------|-----------|--------|----------|-----|---------|
| NORM      | 0.677     | 0.640  | 0.658    | 0.916| 200     |
| AFIB      | 0.641     | 0.590  | 0.615    | 0.923| 200     |
| AFLT      | 0.698     | 0.660  | 0.679    | 0.924| 200     |
| 1dAVb     | 0.789     | 0.825  | 0.807    | 0.971| 200     |
| RBBB      | 0.808     | 0.885  | 0.845    | 0.976| 200     |
| LBBB      | 0.729     | 0.765  | 0.746    | 0.962| 200     |
| **Macro Avg** | **0.724** | **0.727** | **0.725** | **0.945** | **1200** |
| **Weighted Avg** | **0.724** | **0.728** | **0.725** | **-** | **1200** |

- **Overall Accuracy**: 72.75%
- **Best performing class**: RBBB (F1-Score: 0.845, AUC: 0.976)
- **Most challenging class**: AFIB (F1-Score: 0.615, AUC: 0.923)

## Visualizations

### ROC Curves
![ROC Curves](roc_curves.png)

The ROC curves demonstrate the trade-off between sensitivity and specificity for each class. All classes show good discrimination ability with AUC values ranging from 0.916 to 0.976.

### Per-Class Metrics
![Per-Class Metrics](per_class_metrics.png)

This visualization compares precision, recall, and F1-score across all classes.

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

The confusion matrix shows the distribution of predictions for each true class. Notable observations:
- RBBB has the highest correct classification rate (177/200)
- AFIB has the lowest correct classification rate (118/200)
- Some confusion exists between AFIB and AFLT classes

## Abbreviations
- NORM: Normal ECG
- AFIB: Atrial Fibrillation
- AFLT: Atrial Flutter
- 1dAVb: First-degree Atrioventricular Block
- RBBB: Right Bundle Branch Block
- LBBB: Left Bundle Branch Block

## Files
- `detailed_results.json`: Complete benchmark metrics and raw data
- `performance_table.csv`: Summary performance metrics
- `roc_curves.png`: ROC curves for all classes
- `per_class_metrics.png`: Visual comparison of precision, recall, and F1-score
- `confusion_matrix.png`: Visualization of prediction errors 