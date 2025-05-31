import torch
import numpy as np
from transformers import AutoModel
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Define output directories
OUTPUT_BASE_DIR = Path("hubert_ecg_results")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)
PREDICTIONS_CSV_PATH = OUTPUT_BASE_DIR / "validation_predictions.csv"
CONFIDENCE_SCORES_CSV_PATH = OUTPUT_BASE_DIR / "validation_confidence_scores.csv"

def load_validation_data(validation_dir):
    """Load all validation data from the directory."""
    validation_files = []
    for i in range(1, 7):  # validation01 through validation06
        dir_name = f"validation{i:02d}"
        npy_file = Path(validation_dir) / dir_name / f"{dir_name}.npy"
        if npy_file.exists():
            validation_files.append(str(npy_file))
    return validation_files

def preprocess_ecg(ecg_data):
    """Preprocess ECG data for the model.
    The model expects input of shape (batch_size, sequence_length).
    We'll process each channel separately and then combine them."""
    # Ensure data is in the correct shape (channels, time_points)
    if len(ecg_data.shape) == 2:
        if ecg_data.shape[0] != 12:  # If channels are in the second dimension
            ecg_data = ecg_data.T
    
    # Process each channel
    processed_channels = []
    for channel in range(12):
        # Get the channel data
        channel_data = ecg_data[channel]
        
        # Normalize the channel
        channel_data = (channel_data - channel_data.mean()) / (channel_data.std() + 1e-8)
        
        # Add batch dimension
        channel_data = np.expand_dims(channel_data, 0)
        processed_channels.append(channel_data)
    
    # Stack all channels
    processed_data = np.vstack(processed_channels)
    return torch.FloatTensor(processed_data)

def predict_conditions(model, input_tensor):
    """Predict conditions using the model."""
    with torch.no_grad():
        outputs = model(input_tensor)
        # Get the logits from the model output
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

def main():
    # Load the model
    print("Loading HuBERT-ECG model...")
    model = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True)
    model.eval()  # Set to evaluation mode
    
    # Load validation data
    validation_dir = "validation"
    validation_files = load_validation_data(validation_dir)
    
    print(f"Found {len(validation_files)} validation files")
    
    # Initialize results storage
    all_predictions = []
    all_confidence_scores = []
    
    # Process each validation file
    for file_path in validation_files:
        print(f"\nProcessing {file_path}...")
        file_name = Path(file_path).stem
        
        # Load ECG data
        ecg_data = np.load(file_path)
        print(f"Input data shape: {ecg_data.shape}")
        
        # Preprocess data
        input_tensor = preprocess_ecg(ecg_data)
        print(f"Preprocessed tensor shape: {input_tensor.shape}")
        
        # Process each channel separately
        all_features = []
        for i in range(input_tensor.shape[0]):
            channel_tensor = input_tensor[i:i+1]  # Keep batch dimension
            print(f"Processing channel {i+1}, shape: {channel_tensor.shape}")
            
            # Make prediction
            probabilities = predict_conditions(model, channel_tensor)
            all_features.append(probabilities)
        
        # Combine features from all channels
        combined_features = torch.cat(all_features, dim=0)
        mean_features = combined_features.mean(dim=0)  # Average across channels
        
        # Store predictions and confidence scores
        predictions = mean_features.argmax(dim=-1).cpu().numpy()
        confidence_scores = mean_features.max(dim=-1)[0].cpu().numpy()
        
        all_predictions.append({
            'file_name': file_name,
            'predicted_condition': predictions.tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        all_confidence_scores.append({
            'file_name': file_name,
            'confidence_score': confidence_scores.tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Save results to CSV files
    predictions_df = pd.DataFrame(all_predictions)
    confidence_df = pd.DataFrame(all_confidence_scores)
    
    predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
    confidence_df.to_csv(CONFIDENCE_SCORES_CSV_PATH, index=False)
    
    print(f"\nResults saved to:")
    print(f"Predictions: {PREDICTIONS_CSV_PATH}")
    print(f"Confidence scores: {CONFIDENCE_SCORES_CSV_PATH}")

if __name__ == "__main__":
    main()
