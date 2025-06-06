# Code explanation

## Dumping
`dumping.py` contains the code and entry points to compute and dump feature descriptors of raw ECG fragments. These descriptors include:
- time-frequency feautures
- 39 MFCC coefficients
- time-frequency features + 13 MFCC coefficients
- latent representations extracted from $i^{th}$ encoding layer, $i = 0, 1, 2..., 11$

## Clustering
After dumping ECG feature descriptors, one can proceed with the offline clustering step, that is, clustering the feature descriptor and fit a K-means clustering model. 
`clustering.py` implements such a step, saves the resulting model, which is necessary to produce labels to use in the pre-training, and provides evaluation functions to quantify the clustering quality

## Dataset
The `dataset.py` file contains the ECGDataset implementation, responsible of iterating over a csv file representing an ECG dataset (normally train/val/test sets) and provinding the data loader with ECGs, ECG feature descriptors, and ECG up/downstream labels.

## HuBERT-ECG
The architecture of HuBERT-ECG one sees during pre-training is provided in the `hubert_ecg.py` file, while the archicture one sees during fine-tuning or training from scratch is provided in the `hubert_ecg_classification.py` file.
The difference consists in projection & look-up embedding matrices present in the former architecture that are replaced by the classification head present in the latter one.

## Pre-training
`pretrain.py` contains the code to pre-train HuBERT-ECG in a self-supervised manner. `python pretrain.py --help` is highly suggested.

## Fine-tuning
`finetune.py` contains the code to fine-tune and train from scratch HuBERT-ECG in a supervised manner. `python finetune.py --help` is highly suggested as well as a look at `finetune.sh`

## Testing/Evaluation
`test.py` contains the code to evaluate fine-tuned or fully trained HuBERT-ECG instances on test data. `python test.py --help` is highly suggested as well as a look at `test.sh`

## Utils
`utils.py` contains utility functions.

# HuBERT-ECG

HuBERT-ECG is a deep learning model for ECG classification based on the HuBERT architecture.

## ECG Diagnosis App

We've integrated a Streamlit-based application for ECG diagnosis using the HuBERT-ECG model. The app allows users to upload ECG data in `.npy` format and get predictions for six cardiac conditions, along with visualizations and a plain-language report.

### Running the App

```bash
cd /home/tony/neurokit/HuBERT-ECG/code
streamlit run app.py
```

### Features

- Upload ECG data in `.npy` format
- Process single or batch ECG recordings
- Visualize ECG signals with Neurokit2
- Generate AI-based diagnosis for six cardiac conditions
- Create plain-language reports using OpenAI GPT-4 Vision (optional)
- Export results as PDF reports

### Sample Data

You can generate sample ECG data for testing using the provided script:
```bash
python create_sample_ecg.py
```

This will create:
- `sample_data/single_ecg.npy`: A single 12-lead ECG recording
- `sample_data/batch_ecg.npy`: A batch of 5 ECG recordings

For more details, see [README_APP.md](README_APP.md).
