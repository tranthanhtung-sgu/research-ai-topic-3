# ECG Diagnosis App

This is a Streamlit-based application for ECG diagnosis using the HuBERT-ECG model. The app allows users to upload ECG data in `.npy` format and get predictions for six cardiac conditions, along with visualizations and a plain-language report.

## Features

- Upload ECG data in `.npy` format
- Process single or batch ECG recordings
- Visualize ECG signals with Neurokit2
- Generate AI-based diagnosis for six cardiac conditions
- Create plain-language reports using OpenAI GPT-4 Vision (optional)
- Export results as PDF reports

## Supported Conditions

The model can detect the following cardiac conditions:
- Normal ECG
- Atrial Fibrillation
- Atrial Flutter
- Left bundle branch block
- Right bundle branch block
- 1st-degree AV block

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Make sure the HuBERT-ECG model is available in one of the following locations:
   - `outputs/six_conditions_model_anti_overfitting/best_model_accuracy.pt`
   - `outputs/six_conditions_model_anti_overfitting/best_model_loss.pt`
   - `model_outputs/*.pt`

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Upload a `.npy` file containing ECG data
2. (Optional) Check the "Process as batch" option if your file contains multiple ECG recordings
3. Enter your OpenAI API key (optional - if not provided, default reports will be generated)
4. Click "Start" to process the ECG data
5. View the results and download the PDF report

## Input Format

The application accepts `.npy` files with the following formats:
- Single lead: 1D array of shape `(samples,)`
- Multi-lead: 2D array of shape `(leads, samples)` or `(samples, leads)`
- Batch: 3D array of shape `(batch, leads, samples)` or `(batch, samples, leads)`

## Sample Data

You can generate sample ECG data for testing using the provided script:
```
python create_sample_ecg.py
```

This will create:
- `sample_data/single_ecg.npy`: A single 12-lead ECG recording
- `sample_data/batch_ecg.npy`: A batch of 5 ECG recordings

## Backend Components

The application is composed of three main backend modules:

1. **inference.py**: Handles model loading and ECG prediction
2. **eda.py**: Creates ECG visualizations using Matplotlib and Neurokit2
3. **report.py**: Generates plain-language reports using OpenAI GPT-4 Vision

## Testing

You can test the backend functionality using the provided test script:
```
python test_app.py
``` 